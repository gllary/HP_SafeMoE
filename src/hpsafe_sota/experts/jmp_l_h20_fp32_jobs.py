from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from hpsafe_sota.experts import jmp_l_fast32_jobs as base_jobs
from hpsafe_sota.experts.jmp_l_bridge import (
    FAST32_MAX_EPOCHS as H20_FP32_MAX_EPOCHS,
    FAST32_MAX_TIME_DAYS as H20_FP32_MAX_TIME_DAYS,
    FAST32_PROTOCOL as H20_FP32_PROTOCOL,
    FAST32_TASK_SETTINGS as H20_FP32_TASK_SETTINGS,
)


H20_FP32_TASKS = ("mp_gap", "mp_e_form")
JMP_ENV_PYTHON = os.environ.get("JMP_ENV_PYTHON", sys.executable)
DEFAULT_EXPERIMENT_CONFIG = Path(
    "configs/experiments/jmp_l_stage1_mp_tasks.yaml"
)
DEFAULT_RESOURCE_CONFIG = Path("configs/server_resources_jmp_l.yaml")


def _direct_python(command: list[str]) -> list[str]:
    if command[:6] != [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "hpsafe-jmp-l",
        "python",
    ]:
        raise ValueError(f"Unexpected generated command prefix: {command[:6]}")
    return [JMP_ENV_PYTHON, *command[6:]]


def render_jmp_l_h20_fp32_job_graph(
    project_root: Path,
    output_root: Path,
    *,
    tasks: Iterable[str] = H20_FP32_TASKS,
    runtime_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    experiment_config: str | Path = DEFAULT_EXPERIMENT_CONFIG,
    resource_config: str | Path = DEFAULT_RESOURCE_CONFIG,
) -> dict[str, Any]:
    """Render the two-task FP32 Stage-1 job graph."""

    selected = tuple(dict.fromkeys(tasks))
    if not selected or set(selected).difference(H20_FP32_TASKS):
        raise ValueError(f"Tasks must be a non-empty subset of {H20_FP32_TASKS}")

    saved = {
        "protocol": base_jobs.FAST32_PROTOCOL,
        "epochs": base_jobs.FAST32_MAX_EPOCHS,
        "days": base_jobs.FAST32_MAX_TIME_DAYS,
        "settings": base_jobs.FAST32_TASK_SETTINGS,
    }
    try:
        base_jobs.FAST32_PROTOCOL = H20_FP32_PROTOCOL
        base_jobs.FAST32_MAX_EPOCHS = H20_FP32_MAX_EPOCHS
        base_jobs.FAST32_MAX_TIME_DAYS = H20_FP32_MAX_TIME_DAYS
        base_jobs.FAST32_TASK_SETTINGS = {
            task: dict(settings) for task, settings in H20_FP32_TASK_SETTINGS.items()
        }
        graph = base_jobs.render_jmp_l_fast32_job_graph(
            project_root,
            output_root,
            tasks=selected,
            runtime_root=runtime_root,
            cache_root=cache_root,
            experiment_config=experiment_config,
            resource_config=resource_config,
        )
    finally:
        base_jobs.FAST32_PROTOCOL = str(saved["protocol"])
        base_jobs.FAST32_MAX_EPOCHS = int(saved["epochs"])
        base_jobs.FAST32_MAX_TIME_DAYS = int(saved["days"])
        base_jobs.FAST32_TASK_SETTINGS = saved["settings"]  # type: ignore[assignment]

    project_root = project_root.resolve()
    for job in graph["jobs"]:
        kind = job["kind"]
        if kind == "jmp_l_official_finetune":
            job["command"] = [
                JMP_ENV_PYTHON,
                str(project_root / "scripts/run_jmp_l_h20_fp32.py"),
                "--request",
                str(job["request_path"]),
            ]
        else:
            command = _direct_python(job["command"])
            if kind == "jmp_l_preflight":
                command[1] = str(project_root / "scripts/preflight_jmp_l_h20_fp32.py")
            elif kind == "jmp_l_stage1_certify":
                command[1] = str(
                    project_root / "scripts/certify_jmp_l_h20_fp32_stage1.py"
                )
            job["command"] = command

    graph.update(
        {
            "protocol": H20_FP32_PROTOCOL,
            "precision_policy": "32-true_for_both_tasks",
            "initialization": "official_jmp_l_pretrained_checkpoint",
        }
    )
    graph_path = Path(graph["output_root"]) / "jmp_l_stage1_job_graph.json"
    graph_path.write_text(
        json.dumps(graph, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return graph
