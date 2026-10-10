#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASKS = ("glass",)
LANES = (0, 1, 2, 3)


def _isolated(*values: str) -> list[str]:
    return [
        "env",
        "-u",
        "PYTHONPATH",
        "-u",
        "PYTHONHOME",
        "-u",
        "PYTHONUSERBASE",
        "-u",
        "PIP_TARGET",
        "-u",
        "PIP_PREFIX",
        "PYTHONNOUSERSITE=1",
        "PIP_USER=0",
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "hpsafe-anchorboost",
        *values,
    ]


def _job(
    *,
    job_id: str,
    kind: str,
    task: str,
    outer: int,
    lane: int | None,
    command: list[str],
    dependencies: list[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job_id,
        "kind": kind,
        "task_name": task,
        "expert_name": "anchorboost" if task in TASKS else "stage1_export_validation",
        "outer_fold": outer,
        "inner_fold": None,
        "gpu_lane": lane,
        "estimated_gpu_memory_gb": 8 if lane is not None else 0,
        "request_path": None,
        "command": command,
        "dependencies": dependencies,
        "status": "ready",
        "blocked_reason": None,
    }
    if lane is not None:
        payload["timeout_seconds"] = 3600
    return payload


def render(
    *,
    output_root: Path,
    stage1_root: Path,
    export_root: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    stage1_root = stage1_root.expanduser().resolve()
    export_root = export_root.expanduser().resolve()
    if export_root == stage1_root or stage1_root in export_root.parents:
        raise ValueError("Export root may not overwrite frozen Stage1")

    preflight_id = "stage1_anchorboost_exports.preflight"
    jobs: list[dict[str, Any]] = [
        _job(
            job_id=preflight_id,
            kind="stage1_anchorboost_export_preflight",
            task="all",
            outer=-1,
            lane=None,
            command=_isolated(
                "python",
                "-s",
                "scripts/preflight_stage1_anchorboost_exports.py",
                "--stage1-root",
                str(stage1_root),
                "--export-root",
                str(export_root),
            ),
            dependencies=[],
        )
    ]
    export_ids: list[str] = []
    for task_index, task in enumerate(TASKS):
        for outer in range(5):
            lane = LANES[(task_index * 5 + outer) % len(LANES)]
            job_id = f"stage1_anchorboost_exports.{task}.o{outer}"
            export_ids.append(job_id)
            jobs.append(
                _job(
                    job_id=job_id,
                    kind="stage1_anchorboost_checkpoint_replay",
                    task=task,
                    outer=outer,
                    lane=lane,
                    command=_isolated(
                        "python",
                        "-s",
                        "scripts/export_stage1_anchorboost_aligned.py",
                        "--stage1-root",
                        str(stage1_root),
                        "--export-root",
                        str(export_root),
                        "--task",
                        task,
                        "--outer-fold",
                        str(outer),
                    ),
                    dependencies=[preflight_id],
                )
            )
    jobs.append(
        _job(
            job_id="stage1_anchorboost_exports.validation",
            kind="stage1_anchorboost_export_validation",
            task="all",
            outer=-1,
            lane=None,
            command=_isolated(
                "python",
                "-s",
                "scripts/validate_stage1_anchorboost_exports.py",
                "--stage1-root",
                str(stage1_root),
                "--export-root",
                str(export_root),
            ),
            dependencies=export_ids,
        )
    )
    graph = {
        "schema_version": 1,
        "protocol": "stage1_anchorboost_aligned_hidden_replay",
        "project_root": str(PROJECT_ROOT.resolve()),
        "output_root": str(output_root),
        "stage1_root": str(stage1_root),
        "export_root": str(export_root),
        "tasks": list(TASKS),
        "outer_folds": list(range(5)),
        "roles": ["train", "val", "test"],
        "expected_exports": 15,
        "stage1_retraining": False,
        "stage2_launch": False,
        "scheduler_profile": {
            "cpu_concurrency": 4,
            "preprocess_concurrency": 2,
            "work_stealing": {"enabled": True, "prefer_declared_lane": False},
            "gpu_lanes": {
                str(lane): {"max_concurrent_jobs": 2, "reserve_free_gb": 20}
                for lane in LANES
            },
        },
        "jobs": jobs,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "experiment_job_graph.json"
    path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render five frozen-fold replays and one 15-export validation."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/orchestration_stage1_anchorboost_exports",
    )
    parser.add_argument(
        "--stage1-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/stage1_glass_steels_h20",
    )
    parser.add_argument(
        "--export-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/stage1_anchorboost_aligned_exports",
    )
    args = parser.parse_args()
    graph = render(
        output_root=args.output_root,
        stage1_root=args.stage1_root,
        export_root=args.export_root,
    )
    counts = Counter(job["kind"] for job in graph["jobs"])
    if len(graph["jobs"]) != 7:
        raise RuntimeError(f"Expected 7 jobs, rendered {len(graph['jobs'])}")
    print(
        json.dumps(
            {
                "graph": str((args.output_root / "experiment_job_graph.json").resolve()),
                "jobs": len(graph["jobs"]),
                "by_kind": dict(sorted(counts.items())),
                "expected_exports": graph["expected_exports"],
                "gpu_lanes": graph["scheduler_profile"]["gpu_lanes"],
                "stage1_retraining": False,
                "stage2_launch": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
