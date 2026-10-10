"""Shared job-record schema for the Stage-1 provider launchers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertJob:
    job_id: str
    kind: str
    task_name: str
    expert_name: str
    outer_fold: int
    inner_fold: int | None
    gpu_lane: int | None
    estimated_gpu_memory_gb: int
    request_path: str | None
    command: list[str]
    dependencies: list[str]
    status: str = "ready"
    blocked_reason: str | None = None
