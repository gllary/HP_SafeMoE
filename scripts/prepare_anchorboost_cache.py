#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_anchorboost.cache import prepare_anchorboost_cache  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare target-free Anchor AnchorBoost descriptor and graph caches.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--descriptors-only", action="store_true")
    parser.add_argument("--skip-raw-hash", action="store_true")
    args = parser.parse_args()
    result = prepare_anchorboost_cache(
        PROJECT_ROOT,
        args.task,
        descriptors=True,
        graphs=not args.descriptors_only,
        verify_raw_hash=not args.skip_raw_hash,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
