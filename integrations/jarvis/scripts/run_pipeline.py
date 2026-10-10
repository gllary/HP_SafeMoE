#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from _bootstrap import PROJECT_ROOT  # noqa: F401
from jarvis_hpsafe.protocol import (
    calibrate_validation,
    commit_barrier,
    predict_test,
    score_test,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JARVIS5 deployment-aligned HP-SafeMoE.")
    sub = parser.add_subparsers(dest="action", required=True)
    calibrate = sub.add_parser("calibrate-validation")
    calibrate.add_argument("--proposal-run-root", type=Path, required=True)
    calibrate.add_argument("--output-root", type=Path, required=True)
    calibrate.add_argument("--config", type=Path, required=True)
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--cpu-threads", type=int, default=4)
    for action in ("predict-test", "commit-barrier"):
        item = sub.add_parser(action)
        item.add_argument("--proposal-run-root", type=Path, required=True)
        item.add_argument("--output-root", type=Path, required=True)
        item.add_argument("--lock", type=Path, required=True)
    score = sub.add_parser("score-test")
    score.add_argument("--proposal-run-root", type=Path, required=True)
    score.add_argument("--output-root", type=Path, required=True)
    score.add_argument("--barrier", type=Path, required=True)
    report = sub.add_parser("summarize")
    report.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if hasattr(args, "cpu_threads"):
        os.environ.setdefault("OMP_NUM_THREADS", str(args.cpu_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
        torch.set_num_threads(args.cpu_threads)
    if args.action == "calibrate-validation":
        result = calibrate_validation(
            proposal_run_root=args.proposal_run_root,
            output_root=args.output_root,
            config_path=args.config,
            device=torch.device(args.device),
        )
    elif args.action == "predict-test":
        result = predict_test(
            proposal_run_root=args.proposal_run_root,
            output_root=args.output_root,
            lock_path=args.lock,
        )
    elif args.action == "commit-barrier":
        result = commit_barrier(
            proposal_run_root=args.proposal_run_root,
            output_root=args.output_root,
            lock_path=args.lock,
        )
    elif args.action == "score-test":
        result = score_test(
            proposal_run_root=args.proposal_run_root,
            output_root=args.output_root,
            barrier_path=args.barrier,
        )
    else:
        result = summarize(output_root=args.output_root)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

