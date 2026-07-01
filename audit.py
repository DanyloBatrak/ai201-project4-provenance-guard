"""
Audit log helper.

Appends newline-delimited JSON events to a local file and reads them
back for the /log endpoint. Good enough to prove the "every submission
gets logged" contract now; swap the storage backend (SQLite, DB table,
log pipeline) later without touching callers -- they only depend on
log_event(dict) / get_log(limit).
"""

import json
import os
from datetime import datetime, timezone

AUDIT_LOG_PATH = os.environ.get("AUDIT_LOG_PATH", "audit_log.jsonl")


def log_event(event: dict) -> dict:
    """Write one structured entry. Adds an ISO-8601 UTC timestamp if the
    caller didn't supply one. Returns the record actually written.
    """
    record = dict(event)
    record.setdefault(
        "timestamp",
        datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
    )
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def get_log(limit: int = 20) -> list:
    """Return the most recent `limit` entries, newest first."""
    if not os.path.exists(AUDIT_LOG_PATH):
        return []
    with open(AUDIT_LOG_PATH) as f:
        lines = [line.strip() for line in f if line.strip()]
    entries = [json.loads(line) for line in lines]
    return list(reversed(entries[-limit:]))