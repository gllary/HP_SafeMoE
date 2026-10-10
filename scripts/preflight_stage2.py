#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage2.experiment import (  # noqa: E402
    ARCHITECTURE_CANDIDATES,
    SAFETY_PROFILES,
    load_config,
)
from hpsafe_sota.stage2_common.protocol import CANONICAL_TASKS, load_expert_pool  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight independent Stage2 HP-SafeMoE.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resource-config", type=Path, required=True)
    parser.add_argument("--gpu-lanes", type=int, nargs="+", default=list(range(8)))
    args = parser.parse_args()

    feature_root = args.feature_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    pool_path = PROJECT_ROOT / str(config["pool_config_relative_path"])
    pool = load_expert_pool(pool_path)
    if tuple(spec.source_task for spec in pool.experts) != CANONICAL_TASKS:
        raise ValueError("Frozen provider pool is not ordered as the canonical 13 tasks")
    if set(pool.anchor_map) != set(CANONICAL_TASKS) or len(pool.expert_ids) != 13:
        raise ValueError("HP-SafeMoE requires exactly one frozen specialist Anchor per task")
    providers = Counter(spec.provider for spec in pool.experts)
    if providers.get("jmp_l_official_checkpoint_consistent") != 3:
        raise ValueError("Three official JMP-L specialists are required")
    if providers.get("jmp_l_frozen_outer_checkpoint_replay") != 2:
        raise ValueError("Two Fast-32 JMP-L specialists are required")
    if pool.anchor_for("steels") != "tpot_mat_steels_anchor":
        raise ValueError("The steels specialist must be TPOT-Mat")
    validation_path = feature_root / str(config["input_validation_relative_path"])
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    input_validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if (
        input_validation.get("status") != "pass"
        or input_validation.get("target_values_accessed_by_validator") is not False
        or tuple(input_validation.get("task_order", ())) != CANONICAL_TASKS
    ):
        raise ValueError("Stage-1 input validation does not satisfy the release label-free contract")

    resources = yaml.safe_load(args.resource_config.expanduser().resolve(strict=True).read_text())
    allowed = set(map(int, resources["expected_gpu"]["allowed_indices"]))
    requested = set(args.gpu_lanes)
    if len(requested) < 4 or not requested <= allowed:
        raise ValueError(f"GPU lanes outside resource allowlist: {sorted(requested - allowed)}")
    if not torch.cuda.is_available() or torch.cuda.device_count() <= max(requested):
        raise RuntimeError("Requested CUDA devices are not visible")
    if shutil.disk_usage(output_root).free < float(config["minimum_free_disk_gb"]) * 1024**3:
        raise RuntimeError("Insufficient free disk")

    print(
        json.dumps(
            {
                "status": "pass",
                "method": config["method_name"],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_lanes": args.gpu_lanes,
                "tasks": list(CANONICAL_TASKS),
                "experts": list(pool.expert_ids),
                "provider_counts": dict(providers),
                "trained_proposal_architectures": len(ARCHITECTURE_CANDIDATES),
                "global_safety_profiles": list(SAFETY_PROFILES),
                "stage1_frozen": True,
                "joint_13_task_training": True,
                "task_private_residual_output": True,
                "learned_task_source_relation": True,
                "evaluation_protocol": "official_matbench_five_fold",
                "fold_alignment": "same_outer_fold_stage1_stage2",
                "inner_oof_folds": int(config["inner_oof_folds"]),
                "meta_crossfit_partitions": int(config["meta_crossfit_partitions"]),
                "deployment_model": "inner_crossfit_ensemble",
                "proposal_route_policy": "cross_expert_data_or_relation",
                "stage2_artifact_source": "current_run",
                "training_data_scope": "outer_train",
                "profile_selection_data_scope": "fold_local_outer_train_oof",
                "profile_lock_before_outer_test_prediction": True,
                "profile_selection_policy": "one_profile_for_all_tasks_within_each_outer_fold",
                "input_validation": "schema_alignment_and_manifest",
            },
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
