#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_tpot.workflow import preflight  # noqa: E402
from hpsafe_sota.stage2_common.io import atomic_json  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight official TPOT-Mat steels TPOT-Mat.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        cache_root=args.cache_root,
    )
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
