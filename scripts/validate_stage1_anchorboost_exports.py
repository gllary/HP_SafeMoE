#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_anchorboost.anchor_replay import (  # noqa: E402
    HIDDEN_DIM,
    LabelFreeSplitManifest,
    PROVIDER,
    TASKS,
    expert_spec,
)
from hpsafe_sota.stage2_common.cache import validate_expert_export  # noqa: E402
from hpsafe_sota.stage2_common.io import atomic_json, read_json, sha256_file  # noqa: E402


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", suffix=".yaml.tmp", delete=False
        ) as handle:
            temporary = handle.name
            yaml.safe_dump(payload, handle, sort_keys=False)
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate 15 label-free glass AnchorBoost Stage-2 exports."
    )
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    args = parser.parse_args()
    stage1_root = args.stage1_root.expanduser().resolve()
    export_root = args.export_root.expanduser().resolve()

    records: dict[str, Any] = {}
    manifest_bindings: dict[str, dict[str, str]] = {}
    total_exports = 0
    total_rows = 0
    for task in TASKS:
        manifest = LabelFreeSplitManifest(
            PROJECT_ROOT / "data/manifests" / f"{task}.npz"
        )
        manifest.require_valid()
        manifest_bindings[task] = {
            "path": str(manifest.path),
            "file_sha256": sha256_file(manifest.path),
            "semantic_digest": manifest.digest,
        }
        spec = expert_spec(export_root, task)
        records[task] = {}
        for outer in range(5):
            receipt_path = spec.export_root / task / f"outer_{outer}" / "export_receipt.json"
            receipt = read_json(receipt_path)
            if (
                receipt.get("status") != "complete"
                or receipt.get("feature_data_scope") != "target_free"
                or receipt.get("checkpoint_source") != "supplied_stage1_checkpoint"
            ):
                raise ValueError(f"Invalid replay receipt: {receipt_path}")
            fold_record: dict[str, Any] = {
                "receipt": str(receipt_path),
                "receipt_sha256": sha256_file(receipt_path),
                "roles": {},
            }
            checkpoint_hashes: dict[str, str] = {}
            for role in ("train", "val", "test"):
                path = spec.export_root / task / f"outer_{outer}" / role
                export = validate_expert_export(
                    path,
                    manifest=manifest,
                    spec=spec,
                    outer_fold=outer,
                    role=role,
                    verify_hashes=True,
                )
                producer = export.metadata.get("producer", {})
                if (
                    producer.get("adapter")
                    != "anchorboost_native_512d_checkpoint_replay"
                    or producer.get("target_values_accessed_by_exporter") is not False
                    or producer.get("source_target_arrays_loaded") is not False
                    or producer.get("relation_screening") != "none"
                    or not export.available.all()
                    or export.hidden.shape[1] != HIDDEN_DIM
                ):
                    raise ValueError(f"Invalid AnchorBoost export contract: {path}")
                expected_checkpoint_role = "holdout" if role in {"train", "val"} else "outer"
                expected_fit_role = "holdout_train" if role in {"train", "val"} else "outer_train"
                if (
                    producer.get("checkpoint_role") != expected_checkpoint_role
                    or producer.get("checkpoint_fit_role") != expected_fit_role
                ):
                    raise ValueError(f"Wrong checkpoint role for {path}")
                checkpoint_hash = str(export.metadata["expert"]["checkpoint_sha256"])
                checkpoint_hashes[role] = checkpoint_hash
                fold_record["roles"][role] = {
                    "path": str(path),
                    "metadata_sha256": sha256_file(path / "metadata.json"),
                    "rows": len(export.sample_ids),
                    "hidden_dim": export.hidden.shape[1],
                    "checkpoint_sha256": checkpoint_hash,
                }
                total_exports += 1
                total_rows += len(export.sample_ids)
            if checkpoint_hashes["train"] != checkpoint_hashes["val"]:
                raise ValueError(f"{task}/outer_{outer}: train and val checkpoints differ")
            records[task][str(outer)] = fold_record

    if total_exports != 15:
        raise ValueError(f"Expected 15 aligned exports, observed {total_exports}")
    validation_root = export_root / "validation/stage1_anchorboost"
    provider_map_path = validation_root / "anchor_providers.yaml"
    provider_map = {
        "schema_version": 1,
        "status": "validated_provider_exports",
        "protocol": "stage1_anchor_provider_exports",
        "anchor_providers": {
            task: {
                "expert_id": expert_spec(export_root, task).expert_id,
                "source_task": task,
                "provider": PROVIDER,
                "hidden_dim": HIDDEN_DIM,
                "input_modality": "enriched_composition_descriptors",
                "representation_kind": "task_trained_hidden",
                "export_root": str(expert_spec(export_root, task).export_root),
            }
            for task in TASKS
        },
    }
    _atomic_yaml(provider_map_path, provider_map)
    payload = {
        "schema_version": 1,
        "status": "pass",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol": "stage1_aligned_train_validation_test_anchorboost_replay",
        "tasks": list(TASKS),
        "outer_folds": list(range(5)),
        "roles": ["train", "val", "test"],
        "exports_validated": total_exports,
        "rows_validated": total_rows,
        "hidden_dim": HIDDEN_DIM,
        "provider": PROVIDER,
        "manifest_bindings": manifest_bindings,
        "stage1_root": str(stage1_root),
        "export_root": str(export_root),
        "provider_map": str(provider_map_path),
        "provider_map_sha256": sha256_file(provider_map_path),
        "data_scope": {
            "features": "target_free",
            "train_and_val_checkpoint": "holdout fit on holdout_train only",
            "test_checkpoint": "outer-refit fit on outer_train",
        },
        "records": records,
    }
    validation_root.mkdir(parents=True, exist_ok=True)
    validation_path = validation_root / "export_validation.json"
    atomic_json(validation_path, payload)
    lines = [
        "# Glass AnchorBoost hidden-representation export validation",
        "",
        "- Status: PASS",
        "- Stage-1 retraining: no",
        "- Stage-2 launch: no",
        "- Validated exports: 15/15 (1 task x 5 folds x train/val/test)",
        f"- Validated rows: {total_rows}",
        "- Hidden representation: native task-trained 512-dimensional vector",
        "- train/val use one holdout checkpoint; test uses the outer-refit checkpoint",
        "- Label firewall: PASS; the exporter neither loads nor stores target values.",
        "- The provider map records the AnchorBoost glass provider used by Stage-2.",
        "",
        "| Task | Folds | Exports | Hidden | Status |",
        "|---|---:|---:|---:|:---:|",
        "| glass | 5 | 15 | 512 | PASS |",
    ]
    (validation_root / "export_validation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
