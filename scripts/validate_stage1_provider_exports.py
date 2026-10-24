#!/usr/bin/env python3
"""Validate the frozen Stage-1 provider pool using target-free exports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_anchorboost.anchor_replay import (  # noqa: E402
    LabelFreeSplitManifest,
)
from hpsafe_sota.stage2_common.cache import (  # noqa: E402
    export_directory,
    validate_expert_export,
)
from hpsafe_sota.stage2_common.io import atomic_json  # noqa: E402
from hpsafe_sota.stage2_common.protocol import (  # noqa: E402
    CANONICAL_TASKS,
    load_expert_pool,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate label-free Stage-1 exports for all five Matbench folds."
    )
    parser.add_argument(
        "--pool-config",
        type=Path,
        default=PROJECT_ROOT / "configs/reference/stage1_provider_pool.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "stage1_exports/input_validation.json",
    )
    parser.add_argument("--verify-hashes", action="store_true")
    args = parser.parse_args()

    pool = load_expert_pool(args.pool_config)
    if tuple(spec.source_task for spec in pool.experts) != CANONICAL_TASKS:
        raise ValueError("Provider pool order differs from the canonical task order")

    records: list[dict[str, object]] = []
    present_exports = 0
    expected_slots = 0
    for outer_fold in range(5):
        for task in CANONICAL_TASKS:
            manifest = LabelFreeSplitManifest(
                PROJECT_ROOT / "data/manifests" / f"{task}.npz"
            )
            manifest.require_valid()
            anchor_id = pool.anchor_for(task)
            for role in ("train", "val", "test"):
                anchor_available = False
                role_present = 0
                role_available_rows = 0
                for spec in pool.experts:
                    expected_slots += 1
                    export_root = export_directory(spec, task, outer_fold, role)
                    if not (export_root / "metadata.json").is_file():
                        continue
                    export = validate_expert_export(
                        export_root,
                        manifest=manifest,
                        spec=spec,
                        outer_fold=outer_fold,
                        role=role,
                        verify_hashes=args.verify_hashes,
                    )
                    present_exports += 1
                    role_present += 1
                    role_available_rows += int(export.available.sum())
                    if spec.expert_id == anchor_id:
                        if not bool(export.available.all()):
                            raise ValueError(
                                f"Target anchor has unavailable rows: {task}/outer_{outer_fold}/{role}"
                            )
                        anchor_available = True
                if not anchor_available:
                    raise FileNotFoundError(
                        f"Missing target anchor export: {task}/outer_{outer_fold}/{role}/{anchor_id}"
                    )
                records.append(
                    {
                        "task": task,
                        "outer_fold": outer_fold,
                        "role": role,
                        "n_samples": len(manifest.ids({
                            "train": "holdout_train",
                            "val": "holdout_val",
                            "test": "outer_test",
                        }[role], outer_fold)),
                        "anchor_expert": anchor_id,
                        "present_experts": role_present,
                        "available_expert_rows": role_available_rows,
                    }
                )

    payload = {
        "schema_version": "stage1-export-validation",
        "status": "pass",
        "pool_config": str(args.pool_config.expanduser().resolve()),
        "task_order": list(CANONICAL_TASKS),
        "expert_order": list(pool.expert_ids),
        "provider_counts": dict(Counter(spec.provider for spec in pool.experts)),
        "outer_folds": list(range(5)),
        "roles": ["train", "val", "test"],
        "expected_export_slots": expected_slots,
        "present_export_slots": present_exports,
        "hashes_verified": bool(args.verify_hashes),
        "target_values_accessed_by_validator": False,
        "records": records,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
