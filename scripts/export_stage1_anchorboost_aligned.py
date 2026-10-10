#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_anchorboost.anchor_replay import (  # noqa: E402
    TASKS,
    export_fold,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one frozen AnchorBoost fold into aligned Stage2 exports."
    )
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--outer-fold", type=int, choices=range(5), required=True)
    args = parser.parse_args()
    result = export_fold(
        project_root=PROJECT_ROOT,
        stage1_root=args.stage1_root,
        export_root=args.export_root,
        task=args.task,
        outer_fold=args.outer_fold,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
