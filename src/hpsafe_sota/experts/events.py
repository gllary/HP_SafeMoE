from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: Any) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "event": event,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class ProgressHeartbeat:
    """Atomically publish liveness for long external-expert operations."""

    def __init__(self, path: Path, *, interval_seconds: float = 30.0) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = max(1.0, float(interval_seconds))
        self._started = time.monotonic()
        self._stage = "initializing"
        self._detail = ""
        self._completed: int | None = None
        self._total: int | None = None
        self._extra: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False

    def _payload(self, status: str = "running") -> dict[str, Any]:
        with self._lock:
            completed = self._completed
            total = self._total
            payload = {
                "schema_version": 1,
                "status": status,
                "pid": os.getpid(),
                "heartbeat_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": round(time.monotonic() - self._started, 3),
                "stage": self._stage,
                "detail": self._detail,
                "completed": completed,
                "total": total,
                "percent": (
                    None
                    if completed is None or total in {None, 0}
                    else round(100.0 * completed / total, 2)
                ),
                **self._extra,
            }
        return payload

    def _write(self, status: str = "running") -> None:
        payload = self._payload(status=status)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, suffix=".tmp", delete=False
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            temp_path = Path(handle.name)
        os.replace(temp_path, self.path)

    def update(
        self,
        stage: str,
        *,
        detail: str = "",
        completed: int | None = None,
        total: int | None = None,
        **extra: Any,
    ) -> None:
        with self._lock:
            self._stage = str(stage)
            self._detail = str(detail)
            self._completed = completed
            self._total = total
            self._extra.update(extra)
        self._write()

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write()

    def __enter__(self) -> ProgressHeartbeat:
        self._write()
        self._thread = threading.Thread(target=self._loop, name="progress-heartbeat", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=min(5.0, self.interval_seconds))
        with self._lock:
            if exc is None:
                self._stage = "complete"
                self._detail = ""
            else:
                self._stage = "failed"
                self._detail = repr(exc)
        self._write(status="complete" if exc is None else "failed")
