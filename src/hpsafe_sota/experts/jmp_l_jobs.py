from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from hpsafe_sota.experts.interface import ExpertRequest
from hpsafe_sota.experts.jmp_l_bridge import (
    LATENT_CONTRACT_VERSION,
    LATENT_DIM,
    LATENT_SOURCE,
    LEARNING_RATE,
    MAX_TIME_DAYS,
    SUPPORTED_TASKS,
)
from hpsafe_sota.experts.jobs import ExpertJob
from hpsafe_sota.experts.registry import load_expert_runtime


def _yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def _conda(environment: str, *command: str) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", environment, *command]


def render_jmp_l_stage1_job_graph(
    project_root: Path,
    output_root: Path,
    *,
    tasks: Iterable[str] = SUPPORTED_TASKS,
    experiment_config: str | Path = "configs/experiments/jmp_l_stage1.yaml",
    resource_config: str | Path = "configs/server_resources.yaml",
) -> dict[str, Any]:
    """Render one official JMP fine-tune per Matbench outer fold, and nothing else."""

    project_root = project_root.resolve()
    output_root = output_root.resolve()
    selected_tasks = tuple(dict.fromkeys(tasks))
    unsupported = sorted(set(selected_tasks) - set(SUPPORTED_TASKS))
    if unsupported:
        raise ValueError(f"Unsupported JMP-L Stage1 tasks: {unsupported}")
    if not selected_tasks:
        raise ValueError("At least one JMP-L task is required")

    runtime = load_expert_runtime(project_root)["experts"]["jmp_l"]
    experiment_path = (project_root / experiment_config).resolve()
    resources_path = (project_root / resource_config).resolve()
    experiment = _yaml(experiment_path)
    resources = _yaml(resources_path)
    environment = str(_yaml(project_root / str(runtime["environment"]))["name"])
    configured_lr = float(experiment["optimization"]["learning_rate"])
    if configured_lr != LEARNING_RATE:
        raise ValueError(f"JMP-L learning rate must remain {LEARNING_RATE}, got {configured_lr}")
    if experiment.get("hyperparameter_policy") != "fixed_release_values":
        raise ValueError("JMP-L Stage1 requires the fixed release hyperparameters")
    if experiment.get("fit_protocol") != "single_outer_fold_fit":
        raise ValueError("JMP-L Stage1 requires one fit per outer fold")
    maximum_epochs = int(experiment["optimization"]["maximum_epochs"])
    if maximum_epochs != 500:
        raise ValueError("JMP-L Stage1 must remain a complete 500-epoch fine-tune")
    maximum_wall_time_days = int(
        experiment["optimization"].get("maximum_wall_time_days", 7)
    )
    if maximum_wall_time_days != MAX_TIME_DAYS:
        raise ValueError(f"JMP-L official max_time must remain {MAX_TIME_DAYS} days")
    training_seed = int(experiment["training_seed"])
    if training_seed != 42:
        raise ValueError("The official public JMP recipe uses seed 42")
    memory_gb = int(resources["job_memory_gb"]["jmp_l"])
    execution = experiment["execution"]
    if "physical_gpu_indices" in execution:
        gpu_lanes = tuple(int(value) for value in execution["physical_gpu_indices"])
    else:
        gpu_lanes = (int(execution["physical_gpu_index"]),)
    if not gpu_lanes or len(set(gpu_lanes)) != len(gpu_lanes):
        raise ValueError("JMP-L physical GPU indices must be unique and non-empty")
    allowed = {
        int(value)
        for key in ("allowed_indices", "isolated_allowed_indices")
        for value in resources["expected_gpu"].get(key, [])
    }
    unknown_lanes = sorted(set(gpu_lanes).difference(allowed))
    if unknown_lanes:
        raise ValueError(
            f"JMP-L GPU lanes are outside the selected resource allowlist: {unknown_lanes}"
        )
    max_jobs_per_lane = int(execution["max_concurrent_gpu_jobs"])
    if max_jobs_per_lane != 1:
        raise ValueError("Clean JMP-L execution requires exactly one fold job per GPU")
    configured_launch_order = tuple(execution.get("task_launch_order", selected_tasks))
    launch_order = tuple(
        task_name for task_name in configured_launch_order if task_name in selected_tasks
    )
    if len(launch_order) != len(selected_tasks) or set(launch_order) != set(selected_tasks):
        raise ValueError(
            "JMP-L task_launch_order must contain every selected task exactly once"
        )
    work_stealing = bool(execution.get("work_stealing", False))
    preferred_lanes = {
        (task_name, outer_fold): gpu_lanes[
            (task_index * 5 + outer_fold) % len(gpu_lanes)
        ]
        for task_index, task_name in enumerate(selected_tasks)
        for outer_fold in range(5)
    }

    jobs: list[ExpertJob] = []
    preflight_id = "jmp_l.stage1.preflight"
    jobs.append(
        ExpertJob(
            job_id=preflight_id,
            kind="jmp_l_preflight",
            task_name="all",
            expert_name="jmp_l",
            outer_fold=-1,
            inner_fold=None,
            gpu_lane=None,
            estimated_gpu_memory_gb=0,
            request_path=None,
            command=_conda(
                environment,
                "python",
                str(project_root / "scripts/preflight_jmp_l_runtime.py"),
                "--project-root",
                str(project_root),
                "--experiment-config",
                str(experiment_path),
                "--resource-config",
                str(resources_path),
            ),
            dependencies=[],
        )
    )

    for task_name in launch_order:
        task_config = experiment["tasks"][task_name]
        cache_id = f"jmp_l.{task_name}.cache"
        jobs.append(
            ExpertJob(
                job_id=cache_id,
                kind="jmp_l_preprocess",
                task_name=task_name,
                expert_name="jmp_l",
                outer_fold=-1,
                inner_fold=None,
                gpu_lane=None,
                estimated_gpu_memory_gb=0,
                request_path=None,
                command=_conda(
                    environment,
                    "python",
                    str(project_root / "scripts/prepare_jmp_l_cache.py"),
                    "--project-root",
                    str(project_root),
                    "--task",
                    task_name,
                ),
                dependencies=[preflight_id],
            )
        )
        fold_job_ids: list[str] = []
        for outer_fold in range(5):
            gpu_lane = preferred_lanes[(task_name, outer_fold)]
            output_dir = output_root / "jmp_l" / task_name / f"outer_{outer_fold}" / "outer_test"
            request = ExpertRequest(
                schema_version=1,
                project_root=str(project_root),
                expert_name="jmp_l",
                task_name=task_name,
                outer_fold=outer_fold,
                phase="official_finetune",
                output_dir=str(output_dir),
                seed=training_seed,
                gpu_id=gpu_lane,
                extra={
                    "learning_rate": LEARNING_RATE,
                    "maximum_epochs": maximum_epochs,
                    "maximum_wall_time_days": maximum_wall_time_days,
                    "batch_size": int(task_config["batch_size"]),
                    "precision": str(task_config["precision"]),
                    "graph_reduction": str(task_config["graph_reduction"]),
                    "num_workers": 8,
                    "hyperparameter_policy": "fixed_release_values",
                    "fit_protocol": "single_outer_fold_fit",
                    "workflow_stage": "stage1",
                    "latent_export": {
                        "schema_version": LATENT_CONTRACT_VERSION,
                        "source": LATENT_SOURCE,
                        "dimension": LATENT_DIM,
                        "pooling": str(task_config["graph_reduction"]),
                        "row_order": "exact_official_outer_test_manifest_order",
                        "checkpoint_state": "fitted_fold_checkpoint",
                        "inference_only": True,
                        "targets_accessed": False,
                    },
                },
            )
            request_path = output_dir / "request.json"
            request.write(request_path)
            job_id = f"jmp_l.{task_name}.o{outer_fold}.official_finetune"
            fold_job_ids.append(job_id)
            jobs.append(
                ExpertJob(
                    job_id=job_id,
                    kind="jmp_l_official_finetune",
                    task_name=task_name,
                    expert_name="jmp_l",
                    outer_fold=outer_fold,
                    inner_fold=None,
                    gpu_lane=gpu_lane,
                    estimated_gpu_memory_gb=memory_gb,
                    request_path=str(request_path),
                    command=_conda(
                        environment,
                        "python",
                        "-m",
                        "hpsafe_sota.experts.runner",
                        "--request",
                        str(request_path),
                    ),
                    dependencies=[cache_id],
                )
            )

        jobs.append(
            ExpertJob(
                job_id=f"jmp_l.{task_name}.stage1_certify",
                kind="jmp_l_stage1_certify",
                task_name=task_name,
                expert_name="jmp_l",
                outer_fold=-1,
                inner_fold=None,
                gpu_lane=None,
                estimated_gpu_memory_gb=0,
                request_path=None,
                command=_conda(
                    environment,
                    "python",
                    str(project_root / "scripts/certify_jmp_l_stage1.py"),
                    "--task",
                    task_name,
                    "--expert-root",
                    str(output_root),
                ),
                dependencies=fold_job_ids,
            )
        )

    graph = {
        "schema_version": 1,
        "protocol": "project_stage1_matbench_outer_5fold_single_jmp_l_finetune",
        "project_root": str(project_root),
        "output_root": str(output_root),
        "expert_name": "jmp_l",
        "tasks": list(selected_tasks),
        "task_launch_order": list(launch_order),
        "official_outer_folds": [0, 1, 2, 3, 4],
        "validation_split": None,
        "train_split": "complete_official_matbench_outer_train",
        "prediction_split": "official_matbench_outer_test",
        "learning_rate": LEARNING_RATE,
        "maximum_epochs": maximum_epochs,
        "maximum_wall_time_days": maximum_wall_time_days,
        "pretrained_checkpoint": "official_jmp_l",
        "hyperparameter_policy": "fixed_release_values",
        "fit_protocol": "single_outer_fold_fit",
        "workflow_stage": "stage1",
        "inner_oof_generated": False,
        "latents_exported": True,
        "latent_contract": {
            "schema_version": LATENT_CONTRACT_VERSION,
            "source": LATENT_SOURCE,
            "dimension": LATENT_DIM,
            "split": "official_matbench_outer_test",
            "checkpoint_state": "fitted_fold_checkpoint",
            "pooling": "task_official_graph_reduction",
            "inference_only": True,
        },
        "experiment_config": str(experiment_path),
        "resource_config": str(resources_path),
        "scheduler_profile": {
            "cpu_concurrency": int(execution["cpu_concurrency"]),
            "preprocess_concurrency": int(execution["preprocess_concurrency"]),
            "work_stealing": {
                "enabled": work_stealing,
                "prefer_declared_lane": True,
            },
            "gpu_lanes": {
                str(lane): {
                    "max_concurrent_jobs": max_jobs_per_lane,
                    "reserve_free_gb": int(execution["reserve_free_gpu_memory_gb"]),
                }
                for lane in gpu_lanes
            },
        },
        "jobs": [asdict(job) for job in jobs],
    }
    graph_path = output_root / "jmp_l_stage1_job_graph.json"
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.write_text(json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return graph
