from __future__ import annotations

import argparse
import asyncio
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import uuid
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hpsafe_sota.runtime.cache import asset_environment


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", suffix=".json", delete=False
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def job_signatures(jobs: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Bind resumable state to each job and its complete upstream specification."""

    signatures: dict[str, str] = {}
    active: set[str] = set()

    def resolve(job_id: str) -> str:
        if job_id in signatures:
            return signatures[job_id]
        if job_id in active:
            raise ValueError(f"Cycle detected at {job_id}")
        active.add(job_id)
        job = jobs[job_id]
        dependencies = list(job.get("dependencies", []))
        payload = {
            "schema_version": 1,
            "job": job,
            "dependency_signatures": {dependency: resolve(dependency) for dependency in dependencies},
        }
        active.remove(job_id)
        signatures[job_id] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return signatures[job_id]

    for value in jobs:
        resolve(value)
    return signatures


def marker_matches(path: Path, signature: str) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("job_signature") == signature


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recoverable_expert_artifact(job: dict[str, Any], attempt: dict[str, Any]) -> dict[str, Any] | None:
    """Validate a finished expert artifact left by an interrupted scheduler.

    The expert child writes ``completed.json`` only after predictions, metadata, hashes and
    environment receipts are durable.  This lets a replacement scheduler recover the scheduler
    marker without rerunning a successfully finished training job.
    """

    if job.get("kind") not in {
        "expert_validation",
        "expert_stacking_oof",
        "expert_outer",
        "jmp_l_official_finetune",
        "jmp_l_inner_oof",
    }:
        return None
    request_path = job.get("request_path")
    if not request_path:
        return None
    try:
        request = json.loads(Path(str(request_path)).read_text(encoding="utf-8"))
        output = Path(str(request["output_dir"])).resolve()
        completed_path = output / "completed.json"
        completed = json.loads(completed_path.read_text(encoding="utf-8"))
        completed_mtime = completed_path.stat().st_mtime
        metadata = output / "metadata.json"
        started = datetime.fromisoformat(str(attempt["started_at"])).timestamp()
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return None
    if (
        completed.get("status") != "complete"
        or Path(str(completed.get("artifact", ""))).resolve() != output
        or not metadata.is_file()
        or completed.get("metadata_sha256") != _sha256(metadata)
        or completed_mtime + 2.0 < started
    ):
        return None
    return {
        "artifact": str(output),
        "artifact_completion": str(completed_path),
        "metadata_sha256": completed["metadata_sha256"],
    }


def process_start_ticks(pid: int) -> int | None:
    """Return the Linux process start token used to reject PID reuse."""

    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        closing = raw.rfind(")")
        fields = raw[closing + 2 :].split()
        if closing < 0 or len(fields) <= 19 or fields[0] == "Z":
            return None
        return int(fields[19])
    except (OSError, ValueError):
        return None


def process_matches_attempt(attempt: dict[str, Any]) -> bool:
    """Check liveness and, for new receipts, bind the PID to its start token."""

    try:
        pid = int(attempt.get("pid", -1))
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    observed = process_start_ticks(pid)
    if observed is None:
        if Path("/proc").is_dir():
            return False
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return attempt.get("process_start_ticks") is None
    expected = attempt.get("process_start_ticks")
    if expected is None:
        return True
    try:
        return observed == int(expected)
    except (TypeError, ValueError):
        return False


def job_signature_digest(signatures: dict[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(signatures, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def scheduler_runtime_status(
    output_root: Path,
    graph_path: Path,
    signatures: dict[str, str],
    *,
    stale_after_seconds: float = 60.0,
) -> dict[str, Any]:
    """Read the scheduler heartbeat from the durable event-log state."""

    receipt_path = output_root / "scheduler_runtime.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("runtime receipt must be a mapping")
    except (OSError, ValueError, json.JSONDecodeError):
        return {
            "receipt_present": False,
            "healthy": False,
            "alive": False,
            "stalled": False,
            "status": "unavailable",
            "phase": None,
            "pid": None,
            "heartbeat_age_seconds": None,
            "graph_matches": None,
            "graph_hash_matches": None,
            "signature_matches": None,
        }
    heartbeat = receipt.get("heartbeat_at")
    try:
        heartbeat_age = max(0.0, time.time() - datetime.fromisoformat(str(heartbeat)).timestamp())
    except ValueError:
        heartbeat_age = None
    graph = graph_path.resolve()
    graph_matches = Path(str(receipt.get("graph", ""))).resolve() == graph
    try:
        graph_hash_matches = receipt.get("graph_sha256") == _sha256(graph)
    except OSError:
        graph_hash_matches = False
    signature_matches = receipt.get("job_signature_digest") == job_signature_digest(signatures)
    alive = process_matches_attempt(receipt)
    stalled = bool(
        alive
        and receipt.get("status") == "running"
        and heartbeat_age is not None
        and heartbeat_age > stale_after_seconds
    )
    healthy = bool(
        alive
        and receipt.get("status") == "running"
        and graph_matches
        and graph_hash_matches
        and signature_matches
        and not stalled
    )
    return {
        **receipt,
        "receipt_present": True,
        "healthy": healthy,
        "alive": alive,
        "stalled": stalled,
        "heartbeat_age_seconds": None if heartbeat_age is None else round(heartbeat_age, 1),
        "graph_matches": graph_matches,
        "graph_hash_matches": graph_hash_matches,
        "signature_matches": signature_matches,
    }


def _gpu_free_gb(*, timeout_seconds: float = 15.0) -> dict[int, float]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {}
    output: dict[int, float] = {}
    for line in result.stdout.splitlines():
        index, memory = [value.strip() for value in line.split(",", 1)]
        output[int(index)] = float(memory) / 1024.0
    return output


class DAGScheduler:
    def __init__(
        self,
        graph_path: Path,
        *,
        retry_limit: int | None = None,
        job_timeout_seconds: float | None = None,
    ) -> None:
        self.graph_path = graph_path.resolve()
        self.graph = json.loads(self.graph_path.read_text(encoding="utf-8"))
        self.project_root = Path(self.graph["project_root"])
        self.output_root = Path(self.graph["output_root"])
        resource_config_path = Path(
            str(self.graph.get("resource_config", "configs/server_resources.yaml"))
        )
        if not resource_config_path.is_absolute():
            resource_config_path = self.project_root / resource_config_path
        resource_config_path = resource_config_path.resolve()
        try:
            resource_config_path.relative_to(self.project_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Graph resource_config must stay inside the project: {resource_config_path}"
            ) from exc
        resource_config = yaml.safe_load(resource_config_path.read_text(encoding="utf-8"))
        if not isinstance(resource_config, dict):
            raise ValueError(f"Invalid resource configuration: {resource_config_path}")
        self.resource_config_path = resource_config_path
        config = deepcopy(resource_config["scheduler"])
        profile = self.graph.get("scheduler_profile")
        if profile is not None:
            if not isinstance(profile, dict):
                raise ValueError("scheduler_profile must be a mapping")
            expected_gpu = resource_config["expected_gpu"]
            allowed = {
                int(value)
                for key in ("allowed_indices", "isolated_allowed_indices")
                for value in expected_gpu.get(key, [])
            }
            requested_lanes = {
                int(key): deepcopy(value)
                for key, value in dict(profile.get("gpu_lanes", {})).items()
            }
            if not requested_lanes:
                raise ValueError("scheduler_profile.gpu_lanes must not be empty")
            unknown = sorted(set(requested_lanes).difference(allowed))
            if unknown:
                raise ValueError(
                    f"scheduler_profile requests GPU lanes outside the server allowlist: {unknown}"
                )
            for lane, lane_value in requested_lanes.items():
                maximum = int(lane_value.get("max_concurrent_jobs", 0))
                reserve = float(lane_value.get("reserve_free_gb", -1))
                if maximum < 1 or reserve < 0:
                    raise ValueError(
                        f"Invalid scheduler_profile GPU lane {lane}: "
                        "max_concurrent_jobs must be positive and reserve_free_gb non-negative"
                    )
            config["gpu_lanes"] = requested_lanes
            for key in ("cpu_concurrency", "preprocess_concurrency"):
                if key in profile:
                    value = int(profile[key])
                    if value < 1:
                        raise ValueError(f"scheduler_profile.{key} must be positive")
                    config[key] = value
            if "work_stealing" in profile:
                work_stealing_profile = profile["work_stealing"]
                if not isinstance(work_stealing_profile, dict):
                    raise ValueError("scheduler_profile.work_stealing must be a mapping")
                config["work_stealing"] = {
                    **dict(config.get("work_stealing", {})),
                    **work_stealing_profile,
                }
        self.poll_seconds = float(config["poll_seconds"])
        self.retry_limit = int(config["retry_limit"] if retry_limit is None else retry_limit)
        if job_timeout_seconds is not None and job_timeout_seconds <= 0:
            raise ValueError("job_timeout_seconds must be positive")
        self.job_timeout_seconds = job_timeout_seconds
        self.termination_grace_seconds = 10.0
        self.cpu_concurrency = int(config["cpu_concurrency"])
        self.preprocess_concurrency = int(config.get("preprocess_concurrency", 2))
        self.lane_config = {int(key): value for key, value in config["gpu_lanes"].items()}
        work_stealing = config.get("work_stealing", {})
        self.work_stealing_enabled = bool(work_stealing.get("enabled", False))
        self.prefer_declared_lane = bool(work_stealing.get("prefer_declared_lane", True))
        self.jobs = {job["job_id"]: job for job in self.graph["jobs"]}
        self.state_root = self.output_root / "state"
        self.log_root = self.output_root / "logs"
        self.event_path = self.output_root / "scheduler_events.jsonl"
        self.runtime_path = self.output_root / "scheduler_runtime.json"
        self.running: dict[str, asyncio.subprocess.Process] = {}
        self.finish_tasks: set[asyncio.Task[None]] = set()
        self.finish_errors: list[BaseException] = []
        self.running_lanes: Counter[int | None] = Counter()
        self.running_kinds: Counter[str] = Counter()
        self.attempts: Counter[str] = Counter()
        self._validate_graph()
        self.job_signatures = job_signatures(self.jobs)
        self.job_signature_digest = job_signature_digest(self.job_signatures)
        self.run_id = uuid.uuid4().hex
        self.started_at: str | None = None
        self._recover_attempts()

    def _timeout_seconds(self, job: dict[str, Any]) -> float | None:
        value = job.get("timeout_seconds", self.job_timeout_seconds)
        if value is None:
            return None
        timeout = float(value)
        if timeout <= 0:
            raise ValueError(f"{job['job_id']}: timeout_seconds must be positive")
        return timeout

    def _recover_attempts(self) -> None:
        for path in self.state_root.glob("*.attempt.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                job_id = str(payload["job_id"])
                if job_id not in self.jobs:
                    continue
                if payload.get("job_signature") != self.job_signatures[job_id]:
                    continue
                self.attempts[job_id] = max(self.attempts[job_id], int(payload["attempt"]))
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue

    def _alive_orphan_attempts(self) -> list[dict[str, Any]]:
        alive: list[dict[str, Any]] = []
        for job_id in self.jobs:
            if self._complete(job_id) or self._failed(job_id):
                continue
            path = self._marker(job_id, "attempt")
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("job_signature") != self.job_signatures[job_id]:
                    continue
                if not process_matches_attempt(payload):
                    continue
                pid = int(payload["pid"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            alive.append({"job_id": job_id, "pid": pid, "attempt": payload.get("attempt")})
        return alive

    def _validate_graph(self) -> None:
        for job_id, job in self.jobs.items():
            if len(job.get("command", [])) == 0:
                raise ValueError(f"{job_id}: empty command")
            missing = [value for value in job.get("dependencies", []) if value not in self.jobs]
            if missing:
                raise ValueError(f"{job_id}: unknown dependencies {missing}")
            lane = job.get("gpu_lane")
            if lane is not None and int(lane) not in self.lane_config:
                raise ValueError(
                    f"{job_id}: GPU lane {lane} is outside this graph's scheduler profile"
                )
            self._timeout_seconds(job)
        visited: set[str] = set()
        active: set[str] = set()

        def visit(job_id: str) -> None:
            if job_id in active:
                raise ValueError(f"Cycle detected at {job_id}")
            if job_id in visited:
                return
            active.add(job_id)
            for dependency in self.jobs[job_id].get("dependencies", []):
                visit(dependency)
            active.remove(job_id)
            visited.add(job_id)

        for job_id in self.jobs:
            visit(job_id)

    def _marker(self, job_id: str, suffix: str) -> Path:
        return self.state_root / f"{job_id}.{suffix}.json"

    def _complete(self, job_id: str) -> bool:
        return marker_matches(self._marker(job_id, "complete"), self.job_signatures[job_id])

    def _failed(self, job_id: str) -> bool:
        return marker_matches(self._marker(job_id, "failed"), self.job_signatures[job_id])

    def _recover_finished_expert_attempts(self) -> list[str]:
        recovered: list[str] = []
        for job_id, job in self.jobs.items():
            if self._complete(job_id) or self._failed(job_id):
                continue
            attempt = {}
            attempt_path = self._marker(job_id, "attempt")
            try:
                attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
                pid = int(attempt.get("pid", -1))
                if attempt.get("job_signature") != self.job_signatures[job_id]:
                    continue
                if pid > 0:
                    try:
                        os.kill(pid, 0)
                    except OSError:
                        pass
                    else:
                        continue
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            artifact = recoverable_expert_artifact(job, attempt)
            if artifact is None:
                continue
            payload = {
                "schema_version": 1,
                "job_id": job_id,
                "job_signature": self.job_signatures[job_id],
                "status": "complete",
                "return_code": 0,
                "attempt": attempt.get("attempt"),
                "started_at": attempt.get("started_at"),
                "finished_at": _now(),
                "log": attempt.get("log"),
                "command": job["command"],
                "preferred_gpu_lane": job.get("gpu_lane"),
                "assigned_gpu_lane": attempt.get("assigned_gpu_lane", job.get("gpu_lane")),
                "recovered_from_finished_expert_artifact": True,
                **artifact,
            }
            _atomic_json(self._marker(job_id, "complete"), payload)
            self._emit("job_recovered_complete", job_id=job_id, attempt=attempt.get("attempt"))
            recovered.append(job_id)
        return recovered

    def _emit(self, event: str, **payload: Any) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {"time": _now(), "event": event, "run_id": self.run_id, **payload},
                    sort_keys=True,
                )
                + "\n"
            )

    def _write_runtime(self, *, status: str, phase: str, **extra: Any) -> None:
        payload = {
            "schema_version": 1,
            "run_id": self.run_id,
            "pid": os.getpid(),
            "process_start_ticks": process_start_ticks(os.getpid()),
            "status": status,
            "phase": phase,
            "started_at": self.started_at,
            "heartbeat_at": _now(),
            "graph": str(self.graph_path),
            "graph_sha256": _sha256(self.graph_path),
            "graph_job_count": len(self.jobs),
            "job_signature_digest": self.job_signature_digest,
            "running_jobs": sorted(self.running),
            "counts": self.status(),
            **extra,
        }
        _atomic_json(self.runtime_path, payload)

    def _track_finish_task(self, task: asyncio.Task[None]) -> None:
        self.finish_tasks.discard(task)
        if task.cancelled():
            return
        exception = task.exception()
        if exception is not None:
            self.finish_errors.append(exception)

    def _raise_finish_errors(self) -> None:
        if not self.finish_errors:
            return
        exception = self.finish_errors.pop(0)
        raise RuntimeError("A scheduler job finalizer failed") from exception

    def _lane_has_capacity(self, job: dict[str, Any], lane: int, free: dict[int, float]) -> bool:
        config = self.lane_config[lane]
        if self.running_lanes[lane] >= int(config["max_concurrent_jobs"]):
            return False
        if not free:
            return True
        required = float(job.get("estimated_gpu_memory_gb", 0)) + float(config["reserve_free_gb"])
        return free.get(lane, 0.0) >= required

    def _resource_assignment(self, job: dict[str, Any], free: dict[int, float]) -> tuple[bool, int | None]:
        preprocess_kinds = {"expert_preprocess", "mattervial_feature_cache"}
        if (
            job.get("kind") in preprocess_kinds
            and sum(self.running_kinds[kind] for kind in preprocess_kinds)
            >= self.preprocess_concurrency
        ):
            return False, None
        preferred = job.get("gpu_lane")
        if preferred is None:
            return self.running_lanes[None] < self.cpu_concurrency, None
        preferred = int(preferred)
        candidates = [preferred]
        if self.work_stealing_enabled:
            alternatives = [lane for lane in self.lane_config if lane != preferred]
            alternatives.sort(
                key=lambda lane: (
                    self.running_lanes[lane] / int(self.lane_config[lane]["max_concurrent_jobs"]),
                    -free.get(lane, 0.0),
                    lane,
                )
            )
            if self.prefer_declared_lane:
                candidates.extend(alternatives)
            else:
                candidates = sorted(
                    [preferred, *alternatives],
                    key=lambda lane: (
                        self.running_lanes[lane] / int(self.lane_config[lane]["max_concurrent_jobs"]),
                        -free.get(lane, 0.0),
                        lane,
                    ),
                )
        for lane in candidates:
            if self._lane_has_capacity(job, lane, free):
                return True, lane
        return False, None

    async def _launch(self, job: dict[str, Any], assigned_lane: int | None) -> None:
        job_id = job["job_id"]
        preferred_lane = job.get("gpu_lane")
        timeout_seconds = self._timeout_seconds(job)
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        environment["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
        environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        environment.update(asset_environment(self.project_root))
        if assigned_lane is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(assigned_lane)
        self.log_root.mkdir(parents=True, exist_ok=True)
        log_path = self.log_root / f"{job_id}.attempt_{self.attempts[job_id] + 1}.log"
        log_handle = log_path.open("ab")
        self.attempts[job_id] += 1
        attempt_payload = {
            "schema_version": 1,
            "job_id": job_id,
            "job_signature": self.job_signatures[job_id],
            "attempt": self.attempts[job_id],
            "started_at": _now(),
            "pid": None,
            "command": job["command"],
            "log": str(log_path),
            "preferred_gpu_lane": preferred_lane,
            "assigned_gpu_lane": assigned_lane,
            "work_stolen": assigned_lane != preferred_lane,
            "timeout_seconds": timeout_seconds,
        }
        _atomic_json(self._marker(job_id, "attempt"), attempt_payload)
        self._emit(
            "job_started",
            job_id=job_id,
            attempt=self.attempts[job_id],
            gpu_lane=assigned_lane,
            preferred_gpu_lane=preferred_lane,
            work_stolen=assigned_lane != preferred_lane,
            command=job["command"],
            log=str(log_path),
            timeout_seconds=timeout_seconds,
        )
        process = await asyncio.create_subprocess_exec(
            *job["command"],
            cwd=self.project_root,
            env=environment,
            stdout=log_handle,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
        self.running[job_id] = process
        self.running_lanes[assigned_lane] += 1
        self.running_kinds[str(job.get("kind", "unknown"))] += 1
        _atomic_json(
            self._marker(job_id, "attempt"),
            {
                **attempt_payload,
                "pid": process.pid,
                "process_start_ticks": process_start_ticks(process.pid),
                "scheduler_run_id": self.run_id,
            },
        )

        async def finish() -> None:
            timed_out = False
            wait_task = asyncio.create_task(process.wait())
            try:
                if timeout_seconds is None:
                    return_code = await wait_task
                else:
                    done, _ = await asyncio.wait({wait_task}, timeout=timeout_seconds)
                    if wait_task in done:
                        return_code = wait_task.result()
                    else:
                        timed_out = True
                        self._emit(
                            "job_timeout",
                            job_id=job_id,
                            attempt=self.attempts[job_id],
                            timeout_seconds=timeout_seconds,
                            pid=process.pid,
                        )
                        try:
                            os.killpg(process.pid, signal.SIGTERM)
                        except OSError:
                            pass
                        try:
                            return_code = await asyncio.wait_for(
                                asyncio.shield(wait_task), timeout=self.termination_grace_seconds
                            )
                        except asyncio.TimeoutError:
                            try:
                                os.killpg(process.pid, signal.SIGKILL)
                            except OSError:
                                pass
                            return_code = await wait_task
                payload = {
                    "schema_version": 1,
                    "job_id": job_id,
                    "job_signature": self.job_signatures[job_id],
                    "return_code": return_code,
                    "attempt": self.attempts[job_id],
                    "finished_at": _now(),
                    "log": str(log_path),
                    "command": job["command"],
                    "preferred_gpu_lane": preferred_lane,
                    "assigned_gpu_lane": assigned_lane,
                    "work_stolen": assigned_lane != preferred_lane,
                    "timeout_seconds": timeout_seconds,
                    "timed_out": timed_out,
                    "scheduler_run_id": self.run_id,
                }
                if return_code == 0:
                    _atomic_json(
                        self._marker(job_id, "complete"), {**payload, "status": "complete"}
                    )
                    self._emit("job_complete", job_id=job_id, attempt=self.attempts[job_id])
                elif self.attempts[job_id] > self.retry_limit:
                    reason = "timeout" if timed_out else "nonzero_return_code"
                    _atomic_json(
                        self._marker(job_id, "failed"),
                        {**payload, "status": "failed", "reason": reason},
                    )
                    self._emit(
                        "job_failed",
                        job_id=job_id,
                        attempt=self.attempts[job_id],
                        reason=reason,
                    )
                else:
                    self._emit("job_retry_pending", job_id=job_id, attempt=self.attempts[job_id])
            finally:
                log_handle.close()
                self.running.pop(job_id, None)
                self.running_lanes[assigned_lane] -= 1
                self.running_kinds[str(job.get("kind", "unknown"))] -= 1

        task = asyncio.create_task(finish())
        self.finish_tasks.add(task)
        task.add_done_callback(self._track_finish_task)

    def status(self) -> dict[str, int]:
        values = Counter()
        for job_id, job in self.jobs.items():
            if self._complete(job_id):
                values["complete"] += 1
            elif self._failed(job_id):
                values["failed"] += 1
            elif job_id in self.running:
                values["running"] += 1
            elif job.get("status") == "blocked":
                values["blocked"] += 1
            else:
                values["pending"] += 1
        return dict(values)

    async def run(self) -> dict[str, int]:
        self.state_root.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_root / "scheduler.lock"
        with lock_path.open("w", encoding="utf-8") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                owner = scheduler_runtime_status(
                    self.output_root, self.graph_path, self.job_signatures
                )
                raise RuntimeError(
                    f"Another scheduler holds {lock_path}; runtime={json.dumps(owner, sort_keys=True)}"
                ) from exc
            self.started_at = _now()
            self._write_runtime(status="running", phase="starting")
            try:
                orphan_attempts = self._alive_orphan_attempts()
                if orphan_attempts:
                    raise RuntimeError(
                        "Scheduler child processes are active; duplicate launch prevented: "
                        f"{orphan_attempts}"
                    )
                self._recover_finished_expert_attempts()
                self._emit("scheduler_started", jobs=len(self.jobs), graph=str(self.graph_path))
                while True:
                    self._raise_finish_errors()
                    self._write_runtime(status="running", phase="probing_gpu_resources")
                    free = _gpu_free_gb()
                    launched = 0
                    for job_id, job in self.jobs.items():
                        if self._complete(job_id) or self._failed(job_id) or job_id in self.running:
                            continue
                        if job.get("status") == "blocked":
                            continue
                        if self.attempts[job_id] > self.retry_limit:
                            _atomic_json(
                                self._marker(job_id, "failed"),
                                {
                                    "schema_version": 1,
                                    "job_id": job_id,
                                    "job_signature": self.job_signatures[job_id],
                                    "status": "failed",
                                    "reason": "retry_limit_exhausted_before_scheduler_restart",
                                    "attempt": self.attempts[job_id],
                                    "finished_at": _now(),
                                },
                            )
                            continue
                        dependencies = job.get("dependencies", [])
                        if any(self._failed(value) for value in dependencies):
                            _atomic_json(
                                self._marker(job_id, "failed"),
                                {
                                    "schema_version": 1,
                                    "job_id": job_id,
                                    "job_signature": self.job_signatures[job_id],
                                    "status": "failed",
                                    "reason": "dependency_failed",
                                    "finished_at": _now(),
                                },
                            )
                            continue
                        if not all(self._complete(value) for value in dependencies):
                            continue
                        ready, assigned_lane = self._resource_assignment(job, free)
                        if ready:
                            await self._launch(job, assigned_lane)
                            launched += 1
                            if assigned_lane is not None and free:
                                free[assigned_lane] -= float(job.get("estimated_gpu_memory_gb", 0))
                    self._raise_finish_errors()
                    status = self.status()
                    self._write_runtime(
                        status="running",
                        phase="waiting_for_jobs",
                        gpu_free_gb=free,
                        launched_this_cycle=launched,
                    )
                    if not self.running and status.get("pending", 0) == 0:
                        self._emit("scheduler_finished", status=status)
                        self._write_runtime(status="finished", phase="complete")
                        return status
                    if not self.running and launched == 0 and status.get("pending", 0) > 0:
                        raise RuntimeError(f"DAG cannot progress; status={status}")
                    await asyncio.sleep(self.poll_seconds)
            except BaseException as exc:
                try:
                    self._emit(
                        "scheduler_failed",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    self._write_runtime(
                        status="failed",
                        phase="exited_with_error",
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                except OSError:
                    pass
                raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run resumable jobs across configured memory-aware GPU lanes."
    )
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--retry-limit", type=int)
    parser.add_argument(
        "--job-timeout-minutes",
        type=float,
        help="Fallback hard timeout per job; a graph job timeout_seconds takes precedence.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-jobs", action="store_true", help="Include the complete DAG in dry-run output")
    args = parser.parse_args()
    timeout_seconds = (
        None if args.job_timeout_minutes is None else args.job_timeout_minutes * 60.0
    )
    scheduler = DAGScheduler(
        args.graph,
        retry_limit=args.retry_limit,
        job_timeout_seconds=timeout_seconds,
    )
    if args.dry_run:
        payload: dict[str, Any] = {"jobs": len(scheduler.jobs), "status": scheduler.status()}
        if args.show_jobs:
            payload["dag"] = [
                {
                    "job_id": job["job_id"],
                    "kind": job.get("kind"),
                    "dependencies": job.get("dependencies", []),
                    "gpu_lane": job.get("gpu_lane"),
                    "estimated_gpu_memory_gb": job.get("estimated_gpu_memory_gb", 0),
                    "timeout_seconds": scheduler._timeout_seconds(job),
                    "status": job.get("status", "ready"),
                }
                for job in scheduler.jobs.values()
            ]
        print(json.dumps(payload, indent=2))
        return
    print(json.dumps(asyncio.run(scheduler.run()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
