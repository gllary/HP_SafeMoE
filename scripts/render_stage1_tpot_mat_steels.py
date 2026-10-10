#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_tpot.workflow import read_config  # noqa: E402
from hpsafe_sota.stage2_common.io import sha256_file  # noqa: E402


def _isolated(environment: str, threads: int, *command: str) -> list[str]:
    return [
        "env", "-u", "CUDA_VISIBLE_DEVICES", "-u", "CUDA_HOME", "-u", "CUDA_PATH",
        "-u", "LD_LIBRARY_PATH", "-u", "PYTHONPATH", "-u", "PYTHONHOME",
        "-u", "PYTHONUSERBASE", "-u", "PIP_TARGET", "-u", "PIP_PREFIX",
        "PYTHONNOUSERSITE=1", "PIP_USER=0", f"OMP_NUM_THREADS={threads}",
        f"MKL_NUM_THREADS={threads}", f"OPENBLAS_NUM_THREADS={threads}",
        "conda", "run", "--no-capture-output", "-n", environment,
        "python", "-s", *command,
    ]


def _job(
    job_id: str,
    kind: str,
    command: list[str],
    dependencies: list[str],
    result_path: Path,
    *,
    progress_path: Path | None = None,
    seed: int | None = None,
    outer_fold: int | None = None,
    timeout_seconds: float = 1800,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "kind": kind,
        "task_name": "steels",
        "expert_name": "tpot_mat_steels_anchor",
        "seed": seed,
        "outer_fold": outer_fold,
        "gpu_lane": None,
        "estimated_gpu_memory_gb": 0,
        "command": command,
        "dependencies": dependencies,
        "result_path": str(result_path),
        "progress_path": None if progress_path is None else str(progress_path),
        "timeout_seconds": timeout_seconds,
        "status": "ready",
        "blocked_reason": None,
    }


def render(
    *,
    output_root: Path,
    artifact_root: Path,
    cache_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    artifact_root = artifact_root.expanduser().resolve()
    cache_root = cache_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve(strict=True)
    config = read_config(config_path)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)
    environment = str(config["execution"]["environment"])
    threads = int(config["execution"]["threads_per_fit"])
    fit_timeout = float(config["execution"]["fit_timeout_minutes"]) * 60.0
    export_root = artifact_root / "stage2_exports"
    validation_root = artifact_root / "validation/stage1_tpot"
    jobs: list[dict[str, Any]] = []
    preflight_path = output_root / "preflight_receipt.json"
    jobs.append(_job(
        "stage1_tpot.preflight", "stage1_tpot_preflight",
        _isolated(environment, 1, "scripts/preflight_stage1_tpot_mat.py",
            "--config", str(config_path),
            "--cache-root", str(cache_root), "--output", str(preflight_path)),
        [], preflight_path, timeout_seconds=300,
    ))
    cache_receipt = cache_root / "receipt.json"
    jobs.append(_job(
        "stage1_tpot.feature_cache", "expert_preprocess",
        _isolated(environment, 1, "scripts/prepare_stage1_tpot_mat_features.py",
            "--output-root", str(cache_root)),
        ["stage1_tpot.preflight"], cache_receipt, timeout_seconds=300,
    ))
    folds: list[str] = []
    fixed_seed = int(config["fixed_protocol"]["random_seed"])
    for outer in range(5):
        job_id = f"stage1_tpot.fit.o{outer}"
        folds.append(job_id)
        root = artifact_root / "folds" / f"outer_{outer}" / "outer_test"
        progress = root / "progress.json"
        jobs.append(_job(
            job_id, "stage1_tpot_fixed_fold_fit",
            _isolated(environment, threads, "scripts/run_stage1_tpot_mat_fold.py",
                "--config", str(config_path), "--cache-root", str(cache_root),
                "--artifact-root", str(artifact_root),
                "--outer-fold", str(outer), "--progress", str(progress)),
            ["stage1_tpot.feature_cache"], root / "completed.json",
            progress_path=progress, seed=fixed_seed, outer_fold=outer,
            timeout_seconds=fit_timeout,
        ))
    exports: list[str] = []
    for outer in range(5):
        job_id = f"stage1_tpot.export.o{outer}"
        exports.append(job_id)
        receipt = export_root / "tpot_mat_steels_anchor/steels" / f"outer_{outer}/export_receipt.json"
        jobs.append(_job(
            job_id, "stage1_tpot_stage2_export",
            _isolated(environment, 1, "scripts/export_stage1_tpot_mat.py",
                "--config", str(config_path),
                "--cache-root", str(cache_root), "--artifact-root", str(artifact_root),
                "--export-root", str(export_root),
                "--outer-fold", str(outer)),
            [folds[outer]], receipt,
            outer_fold=outer, timeout_seconds=300,
        ))
    validation_path = validation_root / "stage1_tpot_mat_validation.json"
    jobs.append(_job(
        "stage1_tpot.validation", "stage1_tpot_final_validation",
        _isolated(environment, 1, "scripts/validate_stage1_tpot_mat.py",
            "--config", str(config_path), "--artifact-root", str(artifact_root),
            "--export-root", str(export_root),
            "--output-root", str(validation_root)),
        exports, validation_path, timeout_seconds=300,
    ))
    graph = {
        "schema_version": "tpot-mat-stage1-run",
        "graph_kind": "stage1_tpot_mat_official_selected_pipeline_steels_fe14",
        "project_root": str(PROJECT_ROOT.resolve()),
        "output_root": str(output_root),
        "artifact_root": str(artifact_root),
        "cache_root": str(cache_root),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "official_pipeline_sha256": config["official_source"]["pipeline_sha256"],
        "fit_jobs": 5,
        "gpu_jobs": 0,
        "protocol": {
            "fixed_seed": fixed_seed,
            "seed_scope": "all_outer_folds",
            "fit_scope": "outer_train",
            "pipeline_initialization": "official_topology_fresh_outer_fold_fit",
        },
        "scheduler_profile": {
            "cpu_concurrency": int(config["execution"]["cpu_concurrency"]),
            "preprocess_concurrency": int(config["execution"]["preprocess_concurrency"]),
            # The shared scheduler requires a validated lane registry even for
            # an all-CPU graph.  No job below is assigned a GPU lane.
            "gpu_lanes": {
                str(lane): {"max_concurrent_jobs": 1, "reserve_free_gb": 18}
                for lane in (0, 1, 2, 3)
            },
            "work_stealing": {"enabled": False, "prefer_declared_lane": True},
        },
        "jobs": jobs,
    }
    graph_path = output_root / "experiment_job_graph.json"
    graph_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Render official TPOT-Mat steels TPOT-Mat.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/orchestration_stage1_tpot_mat_steels")
    parser.add_argument("--artifact-root", type=Path, default=PROJECT_ROOT / "outputs/stage1_tpot_mat_steels")
    parser.add_argument("--cache-root", type=Path, default=PROJECT_ROOT / "data/external_features/tpot_mat/steels/official_fe14")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/experiments/stage1_tpot_mat_steels.yaml")
    args = parser.parse_args()
    graph = render(output_root=args.output_root, artifact_root=args.artifact_root,
        cache_root=args.cache_root, config_path=args.config)
    print(json.dumps({
        "graph": str((args.output_root / "experiment_job_graph.json").resolve()),
        "jobs": len(graph["jobs"]),
        "by_kind": dict(sorted(Counter(job["kind"] for job in graph["jobs"]).items())),
        "cpu_concurrency": graph["scheduler_profile"]["cpu_concurrency"],
        "gpu_jobs": graph["gpu_jobs"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
