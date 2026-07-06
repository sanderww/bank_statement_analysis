"""Persistent activity log: what happened, when, in which area.

Server actions append JSONL entries to activity_log.jsonl in the data root
(gitignored — it may reference statement filenames). The UI shows the log in
an Activity panel, filterable by category, so action feedback is durable
instead of a one-off message at the bottom of the page.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from . import config

# Known areas; the UI renders these as filter chips.
CATEGORIES = (
    "extract", "categorise", "handoff", "review", "export",
    "training", "prompts", "settings",
)


def log_path():
    return config.data_root() / "activity_log.jsonl"


def log_event(category: str, message: str, level: str = "info") -> dict[str, Any]:
    """Append one entry. Unknown categories are kept (shown under 'other')."""
    entry = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "category": category if category in CATEGORIES else "other",
        "level": level,  # 'info' | 'error'
        "message": message,
    }
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def recent(limit: int = 100, category: str | None = None) -> list[dict[str, Any]]:
    """Most recent entries, newest first, optionally filtered by category."""
    path = log_path()
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if category and e.get("category") != category:
                continue
            entries.append(e)
    return list(reversed(entries))[:limit]
