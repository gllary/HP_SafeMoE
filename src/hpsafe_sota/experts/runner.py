from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from hpsafe_sota.artifacts import write_fold_artifact
from hpsafe_sota.data.manifest import TaskManifest
from hpsafe_sota.experts.events import EventLogger
from hpsafe_sota.experts.interface import ExpertBridge, ExpertRequest
from hpsafe_sota.experts.registry import load_expert_runtime, require_supported


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _environment_receipt(path: Path) -> str:
    packages = sorted(
        {
            f"{distribution.metadata.get('Name', 'unknown')}=={distribution.version}"
            for distribution in importlib.metadata.distributions()
        },
        key=str.lower,
    )
    payload = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }
    _atomic_json(path, payload)
    return _sha256(path)


def _bridge(
    request: ExpertRequest,
    manifest: TaskManifest,
    config: dict[str, Any],
) -> ExpertBridge:
    bridge = config["bridge"]
    if bridge == "tpot_mat_selected_pipeline":
        raise ValueError(
            "TPOT-Mat uses the fixed-fold workflow under scripts/run_stage1_tpot_mat_fold.py"
        )
    if bridge == "anchorboost":
        from hpsafe_sota.stage1_anchorboost.anchorboost import AnchorBoostBridge

        return AnchorBoostBridge(request, manifest, config)
    if bridge == "mattervial_modnet_multitask":
        from hpsafe_sota.experts.mattervial_bridge import MatterVialMODNetBridge

        return MatterVialMODNetBridge(request, manifest, config)
    if bridge == "jmp_l_official_finetune":
        from hpsafe_sota.experts.jmp_l_bridge import JMPLStage1Bridge

        return JMPLStage1Bridge(request, manifest, config)
    raise ValueError(f"No bridge implementation for {bridge!r}")


def run_request(request: ExpertRequest) -> Path:
    request.validate()
    selection_receipt: dict[str, Any] | None = None
    if request.selected_hyperparameters:
        selection_path = Path(request.selected_hyperparameters)
        if not selection_path.is_file():
            raise FileNotFoundError(f"Missing holdout selection receipt: {selection_path}")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("outer_test_used") is not False:
            raise ValueError("Selection receipt does not prove outer-test lockout")
        request = replace(
            request,
            extra={**selection.get("selected_parameters", {}), **request.extra},
        )
        selection_receipt = selection
    request.output_path.mkdir(parents=True, exist_ok=True)
    logger = EventLogger(request.output_path / "events.jsonl")
    logger.emit("job_started", expert=request.expert_name, task=request.task_name, phase=request.phase)
    runtime = load_expert_runtime(request.project_path)
    config = require_supported(runtime, request.expert_name, request.task_name)
    manifest = TaskManifest(request.project_path / "data/manifests" / f"{request.task_name}.npz")
    manifest.require_valid()
    bridge: ExpertBridge | None = None
    try:
        bridge = _bridge(request, manifest, config)
        logger.emit("bridge_resolved", description=bridge.describe())
        output = bridge.run()
        logger.emit("native_run_complete", n_predictions=len(output.y_pred))
        checkpoint_hash = _sha256(output.checkpoint_path) if output.checkpoint_path is not None else None
        environment_hash = _environment_receipt(request.output_path / "environment.json")
        preprocessing_hash = _sha256(output.preprocessing_manifest)
        pretraining = {
            "uses_pretraining": bool(config.get("pretrained", False)),
            "expert_name": request.expert_name,
            "overlap_audit_required": bool(config.get("overlap_audit_required", False)),
        }
        write_fold_artifact(
            request.output_path,
            manifest=manifest,
            model_name=request.expert_name,
            model_family=str(config["bridge"]),
            outer_fold=request.outer_fold,
            inner_fold=request.inner_fold,
            split_role=bridge.split.prediction_role,
            sample_ids=output.sample_ids,
            y_true=output.y_true,
            y_pred=output.y_pred,
            y_uncertainty=output.y_uncertainty,
            latents=output.latents,
            available=output.available,
            run_kind="production",
            checkpoint_sha256=checkpoint_hash,
            environment_lock_sha256=environment_hash,
            preprocessing_manifest_sha256=preprocessing_hash,
            pretrained_disclosure=pretraining,
            selection={
                "source": (
                    str(selection_receipt.get("selection_source"))
                    if selection_receipt and selection_receipt.get("selection_source")
                    else (
                        "fixed_jmp_l_recipe_on_project_stage1_folds"
                        if request.phase == "official_finetune"
                        else (
                            "single_outer_train_holdout"
                            if request.phase
                            in {"holdout_validation", "outer_refit", "artifact_import"}
                            else "stacking_oof_generation_only"
                        )
                    )
                ),
                "outer_test_used": False,
                "selected_hyperparameters": request.selected_hyperparameters,
                "selection_receipt_sha256": (
                    _sha256(Path(request.selected_hyperparameters))
                    if request.selected_hyperparameters
                    else None
                ),
            },
            extra_metadata={**output.extra_metadata, "request_extra": request.extra},
        )
        completed = {
            "schema_version": 1,
            "status": "complete",
            "artifact": str(request.output_path),
            "metadata_sha256": _sha256(request.output_path / "metadata.json"),
        }
        _atomic_json(request.output_path / "completed.json", completed)
        finalize_progress = getattr(bridge, "finalize_progress", None)
        if callable(finalize_progress):
            finalize_progress()
        logger.emit("job_complete")
        return request.output_path
    except Exception as exc:
        if bridge is not None:
            finalize_progress = getattr(bridge, "finalize_progress", None)
            if callable(finalize_progress):
                finalize_progress(exc)
        logger.emit("job_failed", error=repr(exc), traceback=traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one expert job inside its pinned environment.")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    run_request(ExpertRequest.read(args.request))


if __name__ == "__main__":
    main()
