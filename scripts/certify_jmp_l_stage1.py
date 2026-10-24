#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.experts.jmp_l_bridge import SUPPORTED_TASKS  # noqa: E402
from hpsafe_sota.experts.jmp_l_certify import certify_jmp_l_stage1  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify official JMP-L Stage1 fivefold predictions.")
    parser.add_argument("--task", required=True, choices=SUPPORTED_TASKS)
    parser.add_argument(
        "--expert-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/jmp_l_stage1_gpu3",
    )
    args = parser.parse_args()
    result = certify_jmp_l_stage1(PROJECT_ROOT, args.expert_root.resolve(), args.task)
    print(f"{result['status']} {args.task}: mean={result['mean']:.8g}")


if __name__ == "__main__":
    main()
