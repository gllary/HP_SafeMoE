#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.experts.jmp_l_bridge import FAST32_TASK_SETTINGS  # noqa: E402
from hpsafe_sota.experts.jmp_l_fast32_certify import (  # noqa: E402
    certify_jmp_l_fast32_stage1,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Certify JMP-L H20 FP32 fivefold artifacts.")
    parser.add_argument("--task", required=True, choices=tuple(FAST32_TASK_SETTINGS))
    parser.add_argument("--expert-root", type=Path, required=True)
    args = parser.parse_args()
    result = certify_jmp_l_fast32_stage1(
        PROJECT_ROOT, args.expert_root.expanduser().resolve(), args.task
    )
    print(f"{result['status']} {args.task}: mean={result['mean']:.8g}")


if __name__ == "__main__":
    main()
