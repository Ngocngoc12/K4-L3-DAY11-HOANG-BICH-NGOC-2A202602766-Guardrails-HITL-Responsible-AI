"""
Assignment 11 — Audit Log (implemented).

Records every interaction for forensics. Never blocks by itself —
other layers catch attacks; this layer makes them reviewable.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or your pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, float] = {}  # request_id -> start_time

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Store input + start timestamp keyed by request_id/user_id."""
        key = request_id or user_id
        self._open[key] = time.time()
        # We will append the full entry in record_output; save partial info for now
        # (some callers only call record_input — store a pending entry)
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "input": text,
            "timestamp": utc_now_iso(),
            "output": None,
            "blocked": None,
            "layer": None,
            "latency_ms": None,
            "_pending": True,
        })

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Store output, layer decision, latency; append (or update pending) in self.logs."""
        key = request_id or user_id
        start = self._open.pop(key, None)
        latency_ms = round((time.time() - start) * 1000, 2) if start else None

        # Try to update an existing pending entry for this key
        for entry in reversed(self.logs):
            if entry.get("_pending") and entry.get("user_id") == user_id:
                entry["output"] = text
                entry["blocked"] = blocked
                entry["layer"] = layer
                entry["latency_ms"] = latency_ms
                entry.pop("_pending", None)
                return

        # If no pending entry, append a new record
        self.logs.append({
            "request_id": request_id,
            "user_id": user_id,
            "input": None,
            "output": text,
            "blocked": blocked,
            "layer": layer,
            "timestamp": utc_now_iso(),
            "latency_ms": latency_ms,
        })

    def export_json(self, filepath: str | None = None):
        """Write logs to disk (JSON array) under repo-root ``outputs/`` by default."""
        path = filepath or default_audit_log_path()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Clean up any still-pending entries before export
        clean_logs = []
        for entry in self.logs:
            e = dict(entry)
            e.pop("_pending", None)
            clean_logs.append(e)
        out.write_text(json.dumps(clean_logs, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
