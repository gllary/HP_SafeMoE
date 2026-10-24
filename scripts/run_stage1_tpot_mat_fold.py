#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_tpot.workflow import train_fixed_fold  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit one fixed TPOT-Mat outer fold.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    args = parser.parse_args()
    result = train_fixed_fold(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        cache_root=args.cache_root,
        artifact_root=args.artifact_root,
        outer_fold=args.outer_fold,
        progress_path=args.progress,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
