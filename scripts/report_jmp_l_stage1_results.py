#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

TASKS = ("phonons", "mp_gap", "perovskites", "dielectric", "jdft2d", "mp_e_form")
LATENT_DIM = 256


def _number(value: float) -> str:
    return f"{value:.8f}" if abs(value) < 10 else f"{value:.6f}"


def _task_report(root: Path, task: str) -> dict[str, object]:
    task_root = root / "jmp_l" / task
    certification_path = task_root / "certification.json"
    if not certification_path.is_file():
        raise FileNotFoundError(f"Missing certification: {certification_path}")
    certification = json.loads(certification_path.read_text(encoding="utf-8"))
    allowed_statuses = {
        "certified_jmp_l_on_project_stage1_matbench_fivefold",
        "certified_jmp_l_h20_fast32_on_project_stage1_matbench_fivefold",
    }
    if certification.get("status") not in allowed_statuses:
        raise ValueError(f"{task}: invalid certification status")
    fold_scores = [float(value) for value in certification["fold_scores"]]
    if len(fold_scores) != 5 or not all(math.isfinite(value) for value in fold_scores):
        raise ValueError(f"{task}: certification does not contain five finite scores")
    observed_mean = float(np.mean(np.asarray(fold_scores, dtype=np.float64)))
    observed_std = float(np.std(np.asarray(fold_scores, dtype=np.float64)))
    if not math.isclose(observed_mean, float(certification["mean"]), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{task}: certified mean does not match fold scores")
    if not math.isclose(observed_std, float(certification["std"]), rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{task}: certified std does not match fold scores")
    receipts_by_fold = {
        int(receipt["outer_fold"]): receipt for receipt in certification["fold_receipts"]
    }
    if set(receipts_by_fold) != set(range(5)):
        raise ValueError(f"{task}: fold receipts do not cover official outer folds 0..4")

    fold_rows: list[int] = []
    checkpoint_bytes = 0
    latent_hashes: list[str] = []
    for outer_fold in range(5):
        fold_root = task_root / f"outer_{outer_fold}" / "outer_test"
        latent_path = fold_root / "latents.npz"
        prediction_path = fold_root / "predictions.npz"
        checkpoint_path = fold_root / "jmp_l_official_fold.ckpt"
        for path in (latent_path, prediction_path, checkpoint_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        with np.load(latent_path, allow_pickle=False) as latent_payload:
            latents = latent_payload["latents"]
            sample_ids = latent_payload["sample_ids"]
            available = latent_payload["available"]
            if latents.shape != (len(sample_ids), LATENT_DIM):
                raise ValueError(f"{task}/o{outer_fold}: invalid latent shape {latents.shape}")
            if len(available) != len(sample_ids) or not bool(np.all(available)):
                raise ValueError(f"{task}/o{outer_fold}: incomplete latent availability")
            fold_rows.append(len(sample_ids))
        checkpoint_bytes += checkpoint_path.stat().st_size
        receipt = receipts_by_fold[outer_fold]
        latent_hashes.append(str(receipt["latents_sha256"]))

    return {
        "task": task,
        "metric": str(certification["metric"]),
        "direction": str(certification["direction"]),
        "fold_scores": fold_scores,
        "mean": observed_mean,
        "std": observed_std,
        "outer_test_rows": sum(fold_rows),
        "fold_rows": fold_rows,
        "latent_dimension": LATENT_DIM,
        "all_latents_available": True,
        "checkpoint_count": 5,
        "checkpoint_bytes": checkpoint_bytes,
        "certification": str(certification_path.resolve()),
        "latent_sha256": latent_hashes,
        "certification_status": certification["status"],
        "protocol": certification.get("protocol", "jmp_l_official_500_or_7d"),
        "training_profile": str(certification.get("training_profile", "official_finetune")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display and locally validate certified JMP-L Stage1 results."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    tasks = args.tasks or [
        task for task in TASKS if (root / "jmp_l" / task / "certification.json").is_file()
    ]
    if not tasks:
        raise SystemExit(f"No JMP-L certifications found below {root}")
    reports = [_task_report(root, task) for task in tasks]
    if args.json:
        print(json.dumps({"status": "pass", "tasks": reports}, indent=2, sort_keys=True))
        return

    print("JMP-L Stage1 certified Matbench outer-test results")
    print(
        f"{'task':<15} {'metric':<9} {'mean':>13} {'std':>13} "
        f"{'rows':>8} {'latent':>8} {'ckpts':>7}"
    )
    print("-" * 82)
    for report in reports:
        print(
            f"{report['task']:<15} {str(report['metric']).upper():<9} "
            f"{_number(float(report['mean'])):>13} "
            f"{_number(float(report['std'])):>13} "
            f"{int(report['outer_test_rows']):>8} "
            f"{int(report['latent_dimension']):>8} "
            f"{int(report['checkpoint_count']):>7}"
        )
        folds = ", ".join(
            f"o{index}={_number(score)}"
            for index, score in enumerate(report["fold_scores"])
        )
        print(f"  folds: {folds}")
        print(f"  protocol: {report['protocol']} ({report['training_profile']})")
    print("status=pass; all outer-test embeddings are 256-D and fully available")


if __name__ == "__main__":
    main()
