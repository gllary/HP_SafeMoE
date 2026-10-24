#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_tpot.features import build_feature_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build label-free TPOT-Mat steels features.")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_feature_cache(PROJECT_ROOT, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
