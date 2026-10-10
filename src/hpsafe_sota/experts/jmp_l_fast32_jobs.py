from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from hpsafe_sota.experts.interface import ExpertRequest
from hpsafe_sota.experts.jmp_l_bridge import (
    FAST32_CHECKPOINT_EVERY_N_STEPS,
    FAST32_MAX_EPOCHS,
    FAST32_MAX_TIME_DAYS,
    FAST32_PROTOCOL,
    FAST32_TASK_SETTINGS,
    LATENT_CONTRACT_VERSION,
    LATENT_DIM,
    LATENT_SOURCE,
    LEARNING_RATE,
)
from hpsafe_sota.experts.jobs import ExpertJob
from hpsafe_sota.experts.registry import load_expert_runtime

FAST32_TASKS = ("mp_gap", "mp_e_form")
JMP_CONDA_ENVIRONMENT = "hpsafe-jmp-l"


def _yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def _conda(environment: str, *command: str) -> list[str]:
    return ["conda", "run", "--no-capture-output", "-n", environment, *command]


def render_jmp_l_fast32_job_graph(
    project_root: Path,
    output_root: Path,
    *,
    tasks: Iterable[str] = FAST32_TASKS,
    runtime_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    experiment_config: str | Path = (
        "configs/experiments/jmp_l_stage1_mp_tasks.yaml"
    ),
    resource_config: str | Path = "configs/server_resources_jmp_l.yaml",
) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_root = output_root.expanduser().resolve()
    selected_tasks = tuple(dict.fromkeys(tasks))
    if not selected_tasks or set(selected_tasks).difference(FAST32_TASKS):
        raise ValueError(f"The 32-epoch tasks must be a non-empty subset of {FAST32_TASKS}")

    experiment_path = (project_root / experiment_config).resolve()
    resource_path = (project_root / resource_config).resolve()
    experiment = _yaml(experiment_path)
    resources = _yaml(resource_path)
    if experiment.get("finetune_protocol") != FAST32_PROTOCOL:
        raise ValueError("The 32-epoch protocol identifier drifted")
    if float(experiment["optimization"]["learning_rate"]) != LEARNING_RATE:
        raise ValueError("The learning rate must remain 8e-5")
    if int(experiment["optimization"]["maximum_epochs"]) != FAST32_MAX_EPOCHS:
        raise ValueError("The recipe must run exactly 32 epochs")
    if int(experiment["optimization"]["maximum_wall_time_days"]) != FAST32_MAX_TIME_DAYS:
        raise ValueError("The recipe wall-time drifted")
    if experiment.get("hyperparameter_policy") != "fixed_release_values":
        raise ValueError("The recipe requires the fixed release hyperparameters")
    if experiment.get("fit_protocol") != "single_outer_fold_fit":
        raise ValueError("The recipe requires one fit per outer fold")
    if int(experiment["training_seed"]) != 42:
        raise ValueError("JMP-L seed must remain 42")
    for task_name in selected_tasks:
        configured = experiment["tasks"][task_name]
        expected = FAST32_TASK_SETTINGS[task_name]
        if {
            "batch_size": int(configured["batch_size"]),
            "precision": str(configured["precision"]),
            "reduction": str(configured["graph_reduction"]),
            "num_workers": int(configured["num_workers"]),
        } != expected:
            raise ValueError(f"Task settings drifted for {task_name}")

    runtime = load_expert_runtime(project_root)["experts"]["jmp_l"]
    resolved_runtime_root = (
        Path(runtime_root).expanduser().resolve()
        if runtime_root is not None
        else project_root
    )
    source_root = resolved_runtime_root / "external/JMP"
    checkpoint_path = resolved_runtime_root / "checkpoints/jmp-l.pt"
    resolved_cache_root = (
        Path(cache_root).expanduser().resolve()
        if cache_root is not None
        else output_root / "cache/jmp_l"
    )
    # The Stage2 code-only tree may deliberately omit envs/jmp_l.yml.  The
    # actual isolated environment has already been preflighted by name, so graph
    # rendering must not depend on that optional packaging file.
    environment = JMP_CONDA_ENVIRONMENT
    execution = experiment["execution"]
    gpu_lanes = tuple(int(value) for value in execution["physical_gpu_indices"])
    if gpu_lanes != tuple(range(8)):
        raise ValueError("The job graph requires configured physical GPUs 0..7")
    allowed = {int(value) for value in resources["expected_gpu"]["allowed_indices"]}
    if not set(gpu_lanes).issubset(allowed):
        raise ValueError("GPU lanes are outside the resource allowlist")
    launch_order = tuple(
        task for task in execution.get("task_launch_order", FAST32_TASKS) if task in selected_tasks
    )
    if set(launch_order) != set(selected_tasks) or len(launch_order) != len(selected_tasks):
        raise ValueError("task_launch_order must cover selected tasks exactly once")
    memory_gb = int(resources["job_memory_gb"]["jmp_l"])

    jobs: list[ExpertJob] = []
    preflight_id = "jmp_l.fast32.preflight"
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
                str(project_root / "scripts/preflight_jmp_l_h20_fp32.py"),
                "--project-root",
                str(project_root),
                "--experiment-config",
                str(experiment_path),
                "--resource-config",
                str(resource_path),
                "--jmp-runtime-root",
                str(resolved_runtime_root),
            ),
            dependencies=[],
        )
    )

    preferred_lanes = {
        (task_name, outer_fold): gpu_lanes[(task_index * 5 + outer_fold) % len(gpu_lanes)]
        for task_index, task_name in enumerate(launch_order)
        for outer_fold in range(5)
    }
    for task_name in launch_order:
        cache_id = f"jmp_l.fast32.{task_name}.cache"
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
                    "--source-root",
                    str(source_root),
                    "--cache-root",
                    str(resolved_cache_root),
                ),
                dependencies=[preflight_id],
            )
        )
        fold_ids: list[str] = []
        settings = FAST32_TASK_SETTINGS[task_name]
        for outer_fold in range(5):
            lane = preferred_lanes[(task_name, outer_fold)]
            output_dir = output_root / "jmp_l" / task_name / f"outer_{outer_fold}" / "outer_test"
            latent_contract = {
                "schema_version": LATENT_CONTRACT_VERSION,
                "source": LATENT_SOURCE,
                "dimension": LATENT_DIM,
                "pooling": settings["reduction"],
                "row_order": "exact_official_outer_test_manifest_order",
                "checkpoint_state": "fitted_fold_checkpoint",
                "inference_only": True,
                "targets_accessed": False,
            }
            request = ExpertRequest(
                schema_version=1,
                project_root=str(project_root),
                expert_name="jmp_l",
                task_name=task_name,
                outer_fold=outer_fold,
                phase="official_finetune",
                output_dir=str(output_dir),
                seed=42,
                gpu_id=lane,
                extra={
                    "finetune_protocol": FAST32_PROTOCOL,
                    "jmp_source_root": str(source_root),
                    "jmp_checkpoint": str(checkpoint_path),
                    "jmp_cache_root": str(resolved_cache_root),
                    "selection_source_override": "fixed_jmp_l_release_recipe",
                    "learning_rate": LEARNING_RATE,
                    "maximum_epochs": FAST32_MAX_EPOCHS,
                    "maximum_wall_time_days": FAST32_MAX_TIME_DAYS,
                    "batch_size": settings["batch_size"],
                    "precision": settings["precision"],
                    "graph_reduction": settings["reduction"],
                    "num_workers": int(settings["num_workers"]),
                    "checkpoint_every_n_train_steps": FAST32_CHECKPOINT_EVERY_N_STEPS,
                    "auto_resume": True,
                    "hyperparameter_policy": "fixed_release_values",
                    "fit_protocol": "single_outer_fold_fit",
                    "workflow_stage": "stage1",
                    "latent_export": latent_contract,
                },
            )
            request_path = output_dir / "request.json"
            request.write(request_path)
            job_id = f"jmp_l.fast32.{task_name}.o{outer_fold}.official_finetune"
            fold_ids.append(job_id)
            jobs.append(
                ExpertJob(
                    job_id=job_id,
                    kind="jmp_l_official_finetune",
                    task_name=task_name,
                    expert_name="jmp_l",
                    outer_fold=outer_fold,
                    inner_fold=None,
                    gpu_lane=lane,
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
                job_id=f"jmp_l.fast32.{task_name}.stage1_certify",
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
                    str(project_root / "scripts/certify_jmp_l_h20_fp32_stage1.py"),
                    "--task",
                    task_name,
                    "--expert-root",
                    str(output_root),
                ),
                dependencies=fold_ids,
            )
        )

    graph = {
        "schema_version": 1,
        "protocol": FAST32_PROTOCOL,
        "training_profile": "fp32_32epoch",
        "project_root": str(project_root),
        "output_root": str(output_root),
        "expert_name": "jmp_l",
        "tasks": list(selected_tasks),
        "task_launch_order": list(launch_order),
        "official_outer_folds": [0, 1, 2, 3, 4],
        "train_split": "complete_official_matbench_outer_train",
        "prediction_split": "official_matbench_outer_test",
        "validation_split": None,
        "learning_rate": LEARNING_RATE,
        "maximum_epochs": FAST32_MAX_EPOCHS,
        "maximum_wall_time_days": FAST32_MAX_TIME_DAYS,
        "pretrained_checkpoint": "official_jmp_l",
        "jmp_runtime_root": str(resolved_runtime_root),
        "jmp_source_root": str(source_root),
        "jmp_checkpoint": str(checkpoint_path),
        "jmp_cache_root": str(resolved_cache_root),
        "hyperparameter_policy": "fixed_release_values",
        "fit_protocol": "single_outer_fold_fit",
        "workflow_stage": "stage1",
        "inner_oof_generated": False,
        "latents_exported": True,
        "experiment_config": str(experiment_path),
        "resource_config": str(resource_path),
        "scheduler_profile": {
            "cpu_concurrency": int(execution["cpu_concurrency"]),
            "preprocess_concurrency": int(execution["preprocess_concurrency"]),
            "work_stealing": {"enabled": True, "prefer_declared_lane": True},
            "gpu_lanes": {
                str(lane): {
                    "max_concurrent_jobs": 1,
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
