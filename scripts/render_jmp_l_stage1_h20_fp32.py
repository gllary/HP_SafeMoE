#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.experts.jmp_l_h20_fp32_jobs import (  # noqa: E402
    H20_FP32_TASKS,
    render_jmp_l_h20_fp32_job_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the JMP-L FP32 Stage-1 jobs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--jmp-runtime-root", type=Path, required=True)
    parser.add_argument("--jmp-cache-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=H20_FP32_TASKS, default=list(H20_FP32_TASKS))
    args = parser.parse_args()
    graph = render_jmp_l_h20_fp32_job_graph(
        PROJECT_ROOT,
        args.output_root,
        tasks=tuple(args.tasks),
        runtime_root=args.jmp_runtime_root,
        cache_root=args.jmp_cache_root,
    )
    counts = Counter(job["kind"] for job in graph["jobs"])
    graph_path = args.output_root.expanduser().resolve() / "jmp_l_stage1_job_graph.json"
    print(f"WROTE {graph_path}")
    print(f"protocol={graph['protocol']} tasks={graph['tasks']} jobs={dict(counts)}")
    print("outer_folds=0..4 lr=8e-5 epochs=32 precision=32-true")


if __name__ == "__main__":
    main()
