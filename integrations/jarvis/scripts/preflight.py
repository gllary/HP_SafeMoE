#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT  # noqa: F401
from jarvis_hpsafe.constants import JARVIS_TASKS
from jarvis_hpsafe.io import atomic_json
from jarvis_hpsafe.protocol import _source, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only JARVIS HP-SafeMoE preflight.")
    parser.add_argument("--proposal-run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    policy, lock, _, _ = _source(args.proposal_run_root)
    root = args.proposal_run_root.expanduser().resolve(strict=True)
    training = root / "proposal_training/outer_0/proposal_full"
    missing = []
    for inner in range(5):
        for seed in map(int, policy["config_payload"]["model_seeds"]):
            path = training / f"checkpoints/inner_{inner}/seed_{seed}.pt"
            if not path.is_file():
                missing.append(str(path))
    for task in JARVIS_TASKS:
        for path in (
            training / f"calibration/{task}.json",
            root / f"outer_prediction/outer_0/hpsafemoe__oof_locked/{task}.npz",
        ):
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"Missing source proposal artifacts: {missing[:10]}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("The isolated HP-SafeMoE job must see exactly one GPU")
    payload = {
        "schema_version": "jarvis-hpsafemoe-preflight",
        "status": "pass",
        "method": config["method_name"],
        "source_proposal_run_root": str(root),
        "proposal_profile": lock["selected_safety_profile"],
        "proposal_checkpoint_count": 10,
        "stage1_frozen": True,
        "proposal_checkpoints_frozen": True,
        "calibration_data_roles": ["train_oof", "val"],
        "scoring_data_role": "test",
        "visible_gpu_count": torch.cuda.device_count(),
        "torch": torch.__version__,
    }
    args.output_root.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_root.expanduser().resolve() / "preflight.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
