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
    FAST32_MAX_EPOCHS as H20_FP32_MAX_EPOCHS,
    FAST32_MAX_TIME_DAYS as H20_FP32_MAX_TIME_DAYS,
    FAST32_PROTOCOL as H20_FP32_PROTOCOL,
    FAST32_TASK_SETTINGS as H20_FP32_TASK_SETTINGS,
    LEARNING_RATE,
    _activate_source,
)


def _yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight the JMP-L FP32 Stage-1 run.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--experiment-config", type=Path, required=True)
    parser.add_argument("--resource-config", type=Path, required=True)
    parser.add_argument("--jmp-runtime-root", type=Path, default=None)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
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
    runtime_root = (
        args.jmp_runtime_root.expanduser().resolve()
        if args.jmp_runtime_root is not None
        else project_root
    )
    source_root = runtime_root / "external/JMP"
    checkpoint = runtime_root / "checkpoints/jmp-l.pt"
    if not (source_root / "src/jmp").is_dir():
        raise FileNotFoundError(source_root)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    experiment = _yaml(experiment_path)
    resources = _yaml(resource_path)
    if experiment.get("finetune_protocol") != H20_FP32_PROTOCOL:
        raise RuntimeError("H20 FP32 protocol identifier drifted")
    if float(experiment["optimization"]["learning_rate"]) != LEARNING_RATE:
        raise RuntimeError("Learning rate must remain 8e-5")
    if int(experiment["optimization"]["maximum_epochs"]) != H20_FP32_MAX_EPOCHS:
        raise RuntimeError("Epoch budget must remain 32")
    if int(experiment["optimization"]["maximum_wall_time_days"]) != H20_FP32_MAX_TIME_DAYS:
        raise RuntimeError("Wall-time ceiling drifted")
    if experiment.get("hyperparameter_policy") != "fixed_release_values":
        raise RuntimeError("The fixed release hyperparameters are required")
    if experiment.get("fit_protocol") != "single_outer_fold_fit":
        raise RuntimeError("One fit per outer fold is required")
    for task, expected in H20_FP32_TASK_SETTINGS.items():
        configured = experiment["tasks"][task]
        observed = {
            "batch_size": int(configured["batch_size"]),
            "precision": str(configured["precision"]),
            "reduction": str(configured["graph_reduction"]),
            "num_workers": int(configured["num_workers"]),
        }
        if observed != expected:
            raise RuntimeError(f"Task settings drifted for {task}: {observed}")

    _activate_source(source_root)
    import lightning
    import torch
    import torch_geometric
    from jmp.models.gemnet.config import BackboneConfig
    from jmp.tasks.finetune.matbench import MatbenchConfig, MatbenchModel

    if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
        raise RuntimeError(f"Eight CUDA GPUs required; found {torch.cuda.device_count()}")
    expected_gpu = resources["expected_gpu"]
    inventory = []
    for index in range(8):
        properties = torch.cuda.get_device_properties(index)
        memory_gib = float(properties.total_memory) / (1024.0**3)
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_gib = float(free_bytes) / (1024.0**3)
        if "h20" not in properties.name.lower():
            raise RuntimeError(f"GPU {index} is {properties.name!r}, expected H20")
        if memory_gib < float(expected_gpu["memory_gb_minimum_each"]):
            raise RuntimeError(f"GPU {index} has only {memory_gib:.1f} GiB")
        if free_gib < 125.0:
            raise RuntimeError(f"GPU {index} has only {free_gib:.1f} GiB free")
        inventory.append(
            {"index": index, "name": properties.name, "memory_gib": round(memory_gib, 2), "free_gib": round(free_gib, 2)}
        )
    backbone = BackboneConfig.large()
    if backbone.num_blocks != 6 or backbone.emb_size_atom != 256:
        raise RuntimeError("Unexpected JMP-L backbone")
    assert MatbenchConfig is not None and MatbenchModel is not None
    print(
        json.dumps(
            {
                "status": "pass",
                "protocol": H20_FP32_PROTOCOL,
                "tasks": H20_FP32_TASK_SETTINGS,
                "learning_rate": LEARNING_RATE,
                "maximum_epochs": H20_FP32_MAX_EPOCHS,
                "checkpoint": str(checkpoint),
                "source_root": str(source_root),
                "gpu_inventory": inventory,
                "torch": torch.__version__,
                "torch_geometric": torch_geometric.__version__,
                "lightning": lightning.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
