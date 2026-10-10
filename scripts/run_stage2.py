#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage2.experiment import (  # noqa: E402
    commit_prediction_barrier,
    freeze_sources,
    lock_profile_for_outer_fold,
    predict_candidate,
    score_outer_fold,
    summarize,
    train_calibrate_candidate,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one official-fold Stage2 HP-SafeMoE phase."
    )
    sub = parser.add_subparsers(dest="action", required=True)

    freeze = sub.add_parser("freeze")
    freeze.add_argument("--feature-root", type=Path, required=True)
    freeze.add_argument("--output-root", type=Path, required=True)
    freeze.add_argument("--config", type=Path, required=True)

    train = sub.add_parser("train-calibrate")
    train.add_argument("--source-policy", type=Path, required=True)
    train.add_argument("--output-root", type=Path, required=True)
    train.add_argument("--outer-fold", type=int, required=True)
    train.add_argument("--candidate", required=True)
    train.add_argument("--device", default="cuda")
    train.add_argument("--cpu-threads", type=int, default=4)

    lock = sub.add_parser("lock-profile")
    lock.add_argument("--source-policy", type=Path, required=True)
    lock.add_argument("--output-root", type=Path, required=True)
    lock.add_argument("--outer-fold", type=int, required=True)

    predict = sub.add_parser("predict")
    predict.add_argument("--source-policy", type=Path, required=True)
    predict.add_argument("--profile-lock", type=Path, required=True)
    predict.add_argument("--output-root", type=Path, required=True)
    predict.add_argument("--outer-fold", type=int, required=True)
    predict.add_argument("--candidate", required=True)
    predict.add_argument("--device", default="cuda")
    predict.add_argument("--cpu-threads", type=int, default=4)

    barrier = sub.add_parser("commit-predictions")
    barrier.add_argument("--source-policy", type=Path, required=True)
    barrier.add_argument("--output-root", type=Path, required=True)

    score = sub.add_parser("score")
    score.add_argument("--prediction-barrier", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument("--outer-fold", type=int, required=True)

    report = sub.add_parser("summarize")
    report.add_argument("--prediction-barrier", type=Path, required=True)
    report.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    if hasattr(args, "cpu_threads"):
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
        torch.set_num_threads(args.cpu_threads)

    if args.action == "freeze":
        result = freeze_sources(
            project_root=PROJECT_ROOT,
            feature_root=args.feature_root,
            output_root=args.output_root,
            config_path=args.config,
        )
    elif args.action == "train-calibrate":
        result = train_calibrate_candidate(
            source_policy_path=args.source_policy,
            output_root=args.output_root,
            outer_fold=args.outer_fold,
            candidate=args.candidate,
            device=torch.device(args.device),
        )
    elif args.action == "lock-profile":
        result = lock_profile_for_outer_fold(
            source_policy_path=args.source_policy,
            output_root=args.output_root,
            outer_fold=args.outer_fold,
        )
    elif args.action == "predict":
        result = predict_candidate(
            source_policy_path=args.source_policy,
            profile_lock_path=args.profile_lock,
            output_root=args.output_root,
            outer_fold=args.outer_fold,
            candidate=args.candidate,
            device=torch.device(args.device),
        )
    elif args.action == "commit-predictions":
        result = commit_prediction_barrier(
            source_policy_path=args.source_policy,
            output_root=args.output_root,
        )
    elif args.action == "score":
        result = score_outer_fold(
            prediction_barrier_path=args.prediction_barrier,
            output_root=args.output_root,
            outer_fold=args.outer_fold,
        )
    else:
        result = summarize(
            prediction_barrier_path=args.prediction_barrier,
            output_root=args.output_root,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
