#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.stage1_tpot.workflow import validate_exports  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate fixed TPOT-Mat score and Stage-2 exports.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--export-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate_exports(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        artifact_root=args.artifact_root,
        export_root=args.export_root,
        output_root=args.output_root,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
