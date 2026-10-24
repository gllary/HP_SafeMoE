#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage2.experiment import (  # noqa: E402
    ARCHITECTURE_CANDIDATES,
    OUTER_FOLDS,
    load_config,
)
from hpsafe_sota.stage2_common.io import read_json  # noqa: E402


def _python(environment: str, *args: str, threads: int) -> list[str]:
    return [
        "env",
        "-u",
        "CUDA_HOME",
        "-u",
        "CUDA_PATH",
        "-u",
        "LD_LIBRARY_PATH",
        "-u",
        "PYTHONPATH",
        "-u",
        "PYTHONHOME",
        "-u",
        "PYTHONUSERBASE",
        "PYTHONNOUSERSITE=1",
        f"OMP_NUM_THREADS={threads}",
        f"MKL_NUM_THREADS={threads}",
        f"OPENBLAS_NUM_THREADS={threads}",
        "TOKENIZERS_PARALLELISM=false",
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        environment,
        "python",
        "-s",
        *args,
    ]


def _job(
    job_id: str,
    kind: str,
    command: list[str],
    dependencies: list[str],
    *,
    result_path: Path,
    progress_path: Path | None = None,
    gpu_lane: int | None = None,
    timeout_seconds: float = 3600,
    memory: int = 0,
    outer: int | None = None,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "kind": kind,
        "command": command,
        "dependencies": dependencies,
        "gpu_lane": gpu_lane,
        "estimated_gpu_memory_gb": memory,
        "timeout_seconds": timeout_seconds,
        "result_path": str(result_path),
        "progress_path": None if progress_path is None else str(progress_path),
        "outer_fold": outer,
        "variant": "stage2_heterogeneous_providers_official_outer5",
        "seed": None,
    }


def render(
    *,
    feature_root: Path,
    output_root: Path,
    config_path: Path,
    resource_config: Path,
    gpu_lanes: list[int],
    environment: str,
) -> dict[str, Any]:
    feature_root = feature_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()
    config_path = config_path.expanduser().resolve(strict=True)
    resource_config = resource_config.expanduser().resolve(strict=True)
    config = load_config(config_path)
    lanes = tuple(map(int, gpu_lanes))
    if len(lanes) < 4 or len(lanes) != len(set(lanes)) or min(lanes) < 0:
        raise ValueError("HP-SafeMoE requires at least four distinct GPU lanes")
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < float(config["minimum_free_disk_gb"]) * 1024**3:
        raise RuntimeError("Insufficient free disk for HP-SafeMoE")
    orchestration = output_root / "orchestration"
    orchestration.mkdir(parents=True, exist_ok=True)
    threads = int(config["cpu_threads_per_job"])
    memory = int(config["estimated_gpu_memory_gb"])
    jobs_per_gpu = int(config["jobs_per_gpu"])
    reserve = int(config["gpu_reserve_free_gb"])
    jobs: list[dict[str, Any]] = []

    source_policy = output_root / "policy/frozen_stage1_sources.json"
    freeze_id = "stage2.freeze_heterogeneous_official_outer5_sources"
    jobs.append(
        _job(
            freeze_id,
            "stage2_freeze_sources",
            _python(
                environment,
                "scripts/run_stage2.py",
                "freeze",
                "--feature-root",
                str(feature_root),
                "--output-root",
                str(output_root),
                "--config",
                str(config_path),
                threads=threads,
            ),
            [],
            result_path=source_policy,
        )
    )

    training_ids: dict[tuple[int, str], str] = {}
    lane_cursor = 0
    for outer in OUTER_FOLDS:
        for candidate in ARCHITECTURE_CANDIDATES:
            job_id = f"stage2.train_calibrate.{candidate}.o{outer}"
            training_ids[(outer, candidate)] = job_id
            run_root = output_root / f"proposal_training/outer_{outer}/{candidate}"
            lane = lanes[lane_cursor % len(lanes)]
            lane_cursor += 1
            jobs.append(
                _job(
                    job_id,
                    "stage2_true_inner_oof_train_calibrate",
                    _python(
                        environment,
                        "scripts/run_stage2.py",
                        "train-calibrate",
                        "--source-policy",
                        str(source_policy),
                        "--output-root",
                        str(output_root),
                        "--outer-fold",
                        str(outer),
                        "--candidate",
                        candidate,
                        "--device",
                        "cuda",
                        "--cpu-threads",
                        str(threads),
                        threads=threads,
                    ),
                    [freeze_id],
                    result_path=run_root / "calibration_receipt.json",
                    progress_path=run_root / "progress.json",
                    gpu_lane=lane,
                    timeout_seconds=float(config["training_hard_timeout_seconds"]),
                    memory=memory,
                    outer=outer,
                )
            )

    lock_ids: dict[int, str] = {}
    lock_paths: dict[int, Path] = {}
    for outer in OUTER_FOLDS:
        lock_id = f"stage2.lock_profile_from_outer_train_oof.o{outer}"
        lock_path = output_root / f"policy/profile_locks/outer_{outer}.json"
        lock_ids[outer] = lock_id
        lock_paths[outer] = lock_path
        jobs.append(
            _job(
                lock_id,
                "stage2_fold_local_oof_profile_lock",
                _python(
                    environment,
                    "scripts/run_stage2.py",
                    "lock-profile",
                    "--source-policy",
                    str(source_policy),
                    "--output-root",
                    str(output_root),
                    "--outer-fold",
                    str(outer),
                    threads=threads,
                ),
                [training_ids[(outer, "proposal_full")]],
                result_path=lock_path,
                outer=outer,
            )
        )

    prediction_ids: list[str] = []
    for outer in OUTER_FOLDS:
        for candidate in ARCHITECTURE_CANDIDATES:
            job_id = f"stage2.predict_locked.{candidate}.o{outer}"
            prediction_ids.append(job_id)
            run_root = output_root / f"proposal_training/outer_{outer}/{candidate}"
            lane = lanes[lane_cursor % len(lanes)]
            lane_cursor += 1
            jobs.append(
                _job(
                    job_id,
                    "stage2_oof_locked_crossfit_ensemble_predict",
                    _python(
                        environment,
                        "scripts/run_stage2.py",
                        "predict",
                        "--source-policy",
                        str(source_policy),
                        "--profile-lock",
                        str(lock_paths[outer]),
                        "--output-root",
                        str(output_root),
                        "--outer-fold",
                        str(outer),
                        "--candidate",
                        candidate,
                        "--device",
                        "cuda",
                        "--cpu-threads",
                        str(threads),
                        threads=threads,
                    ),
                    [training_ids[(outer, candidate)], lock_ids[outer]],
                    result_path=run_root / "prediction_receipt.json",
                    gpu_lane=lane,
                    timeout_seconds=float(config["training_hard_timeout_seconds"]),
                    memory=memory,
                    outer=outer,
                )
            )

    barrier_path = output_root / "policy/global_prediction_barrier.json"
    barrier_id = "stage2.commit_all_oof_locked_target_free_predictions"
    jobs.append(
        _job(
            barrier_id,
            "stage2_global_prediction_barrier",
            _python(
                environment,
                "scripts/run_stage2.py",
                "commit-predictions",
                "--source-policy",
                str(source_policy),
                "--output-root",
                str(output_root),
                threads=threads,
            ),
            prediction_ids,
            result_path=barrier_path,
        )
    )

    score_ids: list[str] = []
    for outer in OUTER_FOLDS:
        job_id = f"stage2.score_after_barrier.o{outer}"
        score_ids.append(job_id)
        jobs.append(
            _job(
                job_id,
                "stage2_outer_score_evaluation_only",
                _python(
                    environment,
                    "scripts/run_stage2.py",
                    "score",
                    "--prediction-barrier",
                    str(barrier_path),
                    "--output-root",
                    str(output_root),
                    "--outer-fold",
                    str(outer),
                    threads=threads,
                ),
                [barrier_id],
                result_path=output_root / f"outer_scoring/outer_{outer}/outer_receipt.json",
                timeout_seconds=float(config["outer_scoring_hard_timeout_seconds"]),
                outer=outer,
            )
        )

    final_report = output_root / "reports/stage2_summary.md"
    jobs.append(
        _job(
            "stage2.summarize_locked_predictions",
            "stage2_publication_summary",
            _python(
                environment,
                "scripts/run_stage2.py",
                "summarize",
                "--prediction-barrier",
                str(barrier_path),
                "--output-root",
                str(output_root),
                threads=threads,
            ),
            score_ids,
            result_path=output_root / "reports/stage2_summary.json",
        )
    )

    render_spec = {
        "feature_root": str(feature_root),
        "output_root": str(output_root),
        "config": str(config_path),
        "resource_config": str(resource_config),
        "gpu_lanes": list(lanes),
        "environment": environment,
    }
    graph = {
        "schema_version": "hpsafemoe",
        "graph_kind": "stage2_heterogeneous_nested_oof_profile_lock_official_outer5",
        "project_root": str(PROJECT_ROOT.resolve()),
        "output_root": str(orchestration),
        "artifact_root": str(output_root),
        "resource_config": str(resource_config),
        "render_spec": render_spec,
        "final_report": str(final_report),
        "scientific_protocol": {
            "stage1_frozen": True,
            "joint_task_count": 13,
            "task_private_residual_output": True,
            "learned_task_source_relation": True,
            "evaluation_protocol": "official_matbench_five_fold",
            "fold_alignment": "same_outer_fold_stage1_stage2",
            "inner_oof_folds": int(config["inner_oof_folds"]),
            "meta_crossfit_partitions": int(config["meta_crossfit_partitions"]),
            "true_inner_oof_proposals": True,
            "deployment_model": "inner_crossfit_ensemble",
            "proposal_route_policy": "cross_expert_data_or_relation",
            "stage2_artifact_source": "current_run",
            "prediction_data_scope": "target_free_features",
            "all_outer_predictions_committed_before_scoring": True,
            "profile_selection_data_scope": "fold_local_outer_train_oof",
            "profile_lock_before_outer_test_prediction": True,
            "profile_selection_policy": "one_profile_for_all_tasks_within_each_outer_fold",
            "input_validation": "schema_alignment_and_manifest",
        },
        "scheduler_profile": {
            "cpu_concurrency": int(config["cpu_concurrency"]),
            "preprocess_concurrency": 1,
            "gpu_lanes": {
                str(lane): {
                    "max_concurrent_jobs": jobs_per_gpu,
                    "reserve_free_gb": reserve,
                }
                for lane in lanes
            },
            "work_stealing": {
                "enabled": True,
                "prefer_declared_lane": True,
                "allowed_lanes": list(lanes),
            },
        },
        "jobs": jobs,
    }
    expected_training_gpu = len(OUTER_FOLDS) * len(ARCHITECTURE_CANDIDATES)
    expected_prediction_gpu = expected_training_gpu
    expected_gpu = expected_training_gpu + expected_prediction_gpu
    expected_jobs = (
        1 + expected_training_gpu + len(OUTER_FOLDS) + expected_prediction_gpu + 1 + len(OUTER_FOLDS) + 1
    )
    if len(jobs) != expected_jobs or sum(j["gpu_lane"] is not None for j in jobs) != expected_gpu:
        raise AssertionError("HP-SafeMoE job graph size drifted")
    graph_path = orchestration / "experiment_job_graph.json"
    if graph_path.is_file() and read_json(graph_path).get("render_spec") != render_spec:
        raise RuntimeError(f"Refusing to replace a different graph: {graph_path}")
    temporary = graph_path.with_name(f".{graph_path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, graph_path)
    return graph


def main() -> None:
    parser = argparse.ArgumentParser(description="Render independent Stage2 HP-SafeMoE DAG.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/stage2.yaml",
    )
    parser.add_argument(
        "--resource-config",
        type=Path,
        default=PROJECT_ROOT / "configs/server_resources_h20_8gpu_stage2.yaml",
    )
    parser.add_argument("--gpu-lanes", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--environment", default="hpsafe-stage2")
    args = parser.parse_args()
    graph = render(
        feature_root=args.feature_root,
        output_root=args.output_root,
        config_path=args.config,
        resource_config=args.resource_config,
        gpu_lanes=args.gpu_lanes,
        environment=args.environment,
    )
    print(
        json.dumps(
            {
                "status": "ready",
                "graph": str(
                    Path(graph["output_root"])
                    / "orchestration"
                    / "experiment_job_graph.json"
                ),
                "jobs": len(graph["jobs"]),
                "gpu_jobs": sum(job["gpu_lane"] is not None for job in graph["jobs"]),
                "by_kind": dict(Counter(job["kind"] for job in graph["jobs"])),
                "final_report": graph["final_report"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
