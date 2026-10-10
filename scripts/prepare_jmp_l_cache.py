#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.experts.jmp_l_bridge import (  # noqa: E402
    SUPPORTED_TASKS,
    materialize_jmp_cache,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build target-free official-JMP LMDB caches.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--task", action="append", choices=SUPPORTED_TASKS)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--cache-root", default="data/cache/jmp_l")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    source_root = (
        args.source_root.resolve()
        if args.source_root is not None
        else project_root / "external/JMP"
    )
    for task_name in args.task or list(SUPPORTED_TASKS):
        path = materialize_jmp_cache(
            project_root,
            task_name,
            source_root=source_root,
            cache_root=args.cache_root,
        )
        print(f"READY {task_name} {path}")


if __name__ == "__main__":
    main()
