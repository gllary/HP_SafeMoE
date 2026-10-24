#!/usr/bin/env python3
"""Build immutable Matbench split manifests from official inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.data.manifest import build_all_manifests  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify official Matbench files and build the five-fold manifests used "
            "by every Stage-1 provider and HP-SafeMoE Stage 2."
        )
    )
    parser.add_argument(
        "--official-validation",
        type=Path,
        required=True,
        help="Official matbench_v0.1_validation.json file",
    )
    parser.add_argument(
        "--task",
        action="append",
        dest="tasks",
        help="Optional task identifier; repeat to prepare a subset",
    )
    args = parser.parse_args()
    report = build_all_manifests(
        PROJECT_ROOT,
        args.official_validation,
        task_names=args.tasks,
        verify_file_hash=True,
    )
    print(json.dumps({"status": "pass", "tasks": report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
