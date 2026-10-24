#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_anchorboost.anchor_replay import (  # noqa: E402
    LabelFreeSplitManifest,
    TASKS,
    load_label_free_artifact,
)
from hpsafe_sota.stage2_common.io import read_json, sha256_file  # noqa: E402
from hpsafe_sota.stage1_anchorboost.cache import AnchorBoostDescriptorCache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight label-free AnchorBoost train/val/test hidden replay."
    )
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    args = parser.parse_args()

    stage1_root = args.stage1_root.expanduser().resolve()
    export_root = args.export_root.expanduser().resolve()
    if export_root == stage1_root or stage1_root in export_root.parents:
        raise ValueError("Export root may not overwrite or nest inside frozen Stage1")
    bindings: dict[str, dict[str, dict[str, str]]] = {}
    for task in TASKS:
        certification_path = stage1_root / "anchorboost" / task / "certification.json"
        certification = read_json(certification_path)
        if (
            certification.get("status") != "certified_anchor"
            or certification.get("task_name") != task
            or certification.get("expert_name") != "anchorboost"
            or len(certification.get("fold_scores", [])) != 5
        ):
            raise ValueError(f"Invalid fivefold AnchorBoost certification: {task}")
        manifest = LabelFreeSplitManifest(
            PROJECT_ROOT / "data/manifests" / f"{task}.npz"
        )
        manifest.require_valid()
        cache = AnchorBoostDescriptorCache(PROJECT_ROOT, task)
        if cache.receipt.get("manifest_digest") != manifest.digest:
            raise ValueError(f"{task}: descriptor cache manifest changed")
        bindings[task] = {}
        by_outer = {
            int(record["outer_fold"]): record
            for record in certification.get("fold_receipts", [])
        }
        if set(by_outer) != set(range(5)):
            raise ValueError(f"{task}: incomplete certified folds")
        for outer in range(5):
            base = stage1_root / "anchorboost" / task / f"outer_{outer}"
            holdout = load_label_free_artifact(
                base / "validation",
                manifest=manifest,
                outer_fold=outer,
                expected_role="holdout_val",
            )
            test = load_label_free_artifact(
                base / "outer_test",
                manifest=manifest,
                outer_fold=outer,
                expected_role="outer_test",
            )
            if test.metadata_sha256 != by_outer[outer].get("metadata_sha256"):
                raise ValueError(f"{task}/outer_{outer}: certification metadata changed")
            bindings[task][str(outer)] = {
                "manifest_file_sha256": sha256_file(manifest.path),
                "manifest_semantic_digest": manifest.digest,
                "holdout_checkpoint_sha256": holdout.checkpoint_sha256,
                "outer_checkpoint_sha256": test.checkpoint_sha256,
                "holdout_metadata_sha256": holdout.metadata_sha256,
                "outer_metadata_sha256": test.metadata_sha256,
                "descriptor_cache_receipt_sha256": sha256_file(cache.receipt_path),
            }

    free_gib = shutil.disk_usage(export_root.parent).free / (1024.0**3)
    if free_gib < 2.0:
        raise RuntimeError(f"At least 2 GiB free disk is required; observed {free_gib:.2f}")
    receipt = {
        "schema_version": 1,
        "status": "pass",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "tasks": list(TASKS),
        "outer_folds": list(range(5)),
        "expected_exports": 15,
        "stage1_root": str(stage1_root),
        "export_root": str(export_root),
        "free_disk_gib": round(free_gib, 2),
        "bindings": bindings,
        "feature_data_scope": "target_free",
        "checkpoint_source": "supplied_stage1_checkpoint",
    }
    export_root.mkdir(parents=True, exist_ok=True)
    path = export_root / "preflight_receipt.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
