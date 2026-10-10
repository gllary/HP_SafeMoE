from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_expert_runtime(project_root: Path) -> dict[str, Any]:
    path = project_root / "configs/expert_runtime.yaml"
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def require_supported(runtime: dict[str, Any], expert_name: str, task_name: str) -> dict[str, Any]:
    try:
        expert = runtime["experts"][expert_name]
    except KeyError as exc:
        raise KeyError(f"Unknown expert: {expert_name}") from exc
    if task_name not in expert["supported_tasks"]:
        raise ValueError(f"{expert_name} does not declare support for {task_name}")
    return expert
