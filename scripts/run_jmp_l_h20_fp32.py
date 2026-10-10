#!/usr/bin/env python3
from __future__ import annotations

# Apply process-level safety before importing torch, JMP, or the project runner.
import os
import resource
import sys
from pathlib import Path

os.environ["TORCH_BLAS_PREFER_CUBLASLT"] = "0"
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hpsafe_sota.experts.jmp_l_bridge import FAST32_PROTOCOL  # noqa: E402


def main() -> None:
    import argparse

    from hpsafe_sota.experts.interface import ExpertRequest

    parser = argparse.ArgumentParser(description="Run one fresh JMP-L H20 FP32 fold.")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = ExpertRequest.read(args.request)
    if request.task_name not in {"mp_gap", "mp_e_form"}:
        raise ValueError(f"Unsupported H20 FP32 task: {request.task_name!r}")
    if request.extra.get("finetune_protocol") != FAST32_PROTOCOL:
        raise ValueError("Request does not use the H20 FP32 protocol")
    from hpsafe_sota.experts.runner import run_request

    run_request(request)


if __name__ == "__main__":
    main()
