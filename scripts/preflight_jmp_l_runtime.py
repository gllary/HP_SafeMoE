#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.experts.jmp_l_bridge import (  # noqa: E402
    LEARNING_RATE,
    OFFICIAL_COMMIT,
    SUPPORTED_TASKS,
    TASK_SETTINGS,
    _activate_source,
    _checkpoint_path,
    _verify_source_commit,
)


def _yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the isolated official JMP-L runtime.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/jmp_l_stage1.yaml"),
    )
    parser.add_argument(
        "--resource-config",
        type=Path,
        default=Path("configs/server_resources.yaml"),
    )
    parser.add_argument("--allow-no-cuda", action="store_true")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    runtime = _yaml(project_root / "configs/expert_runtime.yaml")["experts"]["jmp_l"]
    experiment_path = (
        args.experiment_config
        if args.experiment_config.is_absolute()
        else project_root / args.experiment_config
    )
    resource_path = (
        args.resource_config
        if args.resource_config.is_absolute()
        else project_root / args.resource_config
    )
    experiment = _yaml(experiment_path)
    resources = _yaml(resource_path)
    source_root = project_root / str(runtime["source_root"])
    source_commit = _verify_source_commit(source_root)
    if source_commit != OFFICIAL_COMMIT:
        raise RuntimeError(source_commit)
    checkpoint = _checkpoint_path(project_root, runtime, {})
    _activate_source(source_root)

    import lightning
    import torch
    import torch_geometric
    from jmp.models.gemnet.config import BackboneConfig
    from jmp.tasks.finetune.matbench import MatbenchConfig, MatbenchModel

    if not args.allow_no_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the JMP-L Conda environment")
    if torch.cuda.is_available():
        expected_gpu = resources["expected_gpu"]
        required_count = int(expected_gpu["count_at_least"])
        if torch.cuda.device_count() < required_count:
            raise RuntimeError(
                f"Expected at least {required_count} GPUs, found {torch.cuda.device_count()}"
            )
        selected_indices = tuple(
            int(value)
            for key in ("allowed_indices", "isolated_allowed_indices")
            for value in expected_gpu.get(key, [])
        )
        required_name = str(expected_gpu.get("name_contains", "")).lower()
        minimum_memory_gb = float(expected_gpu.get("memory_gb_minimum_each", 0))
        gpu_inventory = []
        for index in selected_indices:
            if index >= torch.cuda.device_count():
                raise RuntimeError(f"Configured GPU index {index} does not exist")
            properties = torch.cuda.get_device_properties(index)
            memory_gb = float(properties.total_memory) / (1024.0**3)
            if required_name and required_name not in properties.name.lower():
                raise RuntimeError(
                    f"GPU {index} is {properties.name!r}, expected name containing {required_name!r}"
                )
            if memory_gb < minimum_memory_gb:
                raise RuntimeError(
                    f"GPU {index} has {memory_gb:.2f} GiB, expected at least {minimum_memory_gb}"
                )
            gpu_inventory.append(
                {"index": index, "name": properties.name, "memory_gib": round(memory_gb, 2)}
            )
    else:
        gpu_inventory = []
    configured_lr = float(experiment["optimization"]["learning_rate"])
    if configured_lr != LEARNING_RATE:
        raise RuntimeError(f"Learning-rate drift: {configured_lr} != {LEARNING_RATE}")
    if int(experiment["training_seed"]) != 42:
        raise RuntimeError("Official JMP training seed must remain 42")
    if experiment.get("hyperparameter_policy") != "fixed_release_values":
        raise RuntimeError("JMP-L Stage1 requires the fixed release hyperparameters")
    if experiment.get("fit_protocol") != "single_outer_fold_fit":
        raise RuntimeError("JMP-L Stage1 requires one fit per outer fold")
    if experiment.get("target_normalization") != (
        "complete_official_outer_train_only"
    ):
        raise RuntimeError("JMP-L normalization must use complete outer-train targets only")
    if experiment["optimization"].get("reduce_on_plateau_validation_feedback") is not False:
        raise RuntimeError("Validation-dependent LR updates must remain disabled")
    execution = experiment["execution"]
    physical_gpu_indices = execution.get("physical_gpu_indices")
    if physical_gpu_indices is not None and len(physical_gpu_indices) > 1:
        if execution.get("task_launch_order") != [
            "mp_gap",
            "perovskites",
            "dielectric",
            "jdft2d",
            "phonons",
        ]:
            raise RuntimeError("A6000 task launch order must remain longest-task-first")
        if execution.get("work_stealing") is not True:
            raise RuntimeError("A6000 work stealing must remain enabled")
    if tuple(experiment["tasks"]) != SUPPORTED_TASKS:
        raise RuntimeError("The JMP-L experiment task order/set drifted from the bridge")
    for task_name, settings in TASK_SETTINGS.items():
        configured = experiment["tasks"][task_name]
        if int(configured["batch_size"]) != int(settings["batch_size"]):
            raise RuntimeError(f"Batch-size drift for {task_name}")
        if configured["precision"] != settings["precision"]:
            raise RuntimeError(f"Precision drift for {task_name}")
        if configured["graph_reduction"] != settings["reduction"]:
            raise RuntimeError(f"Readout drift for {task_name}")
    backbone = BackboneConfig.large()
    if backbone.num_blocks != 6 or backbone.emb_size_atom != 256:
        raise RuntimeError("Pinned JMP checkout does not expose the expected JMP-L backbone")
    print(
        json.dumps(
            {
                "status": "pass",
                "source_commit": source_commit,
                "checkpoint": str(checkpoint),
                "learning_rate": LEARNING_RATE,
                "tasks": list(SUPPORTED_TASKS),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "gpu_inventory": gpu_inventory,
                "experiment_config": str(experiment_path.resolve()),
                "resource_config": str(resource_path.resolve()),
                "torch": torch.__version__,
                "torch_geometric": torch_geometric.__version__,
                "lightning": lightning.__version__,
                "matbench_config": MatbenchConfig.__name__,
                "matbench_model": MatbenchModel.__name__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
