"""Machine-local app settings, persisted as settings.json (gitignored).

Holds the choices the user makes in the UI's Setup section: which prompt
version and model version are active, the low-confidence threshold, and the
default OpenAI model. Everything has a code default so a missing/partial file
is always fine.
"""
from __future__ import annotations

import json
from typing import Any

from . import config

DEFAULTS: dict[str, Any] = {
    "active_prompt_version": config.DEFAULT_PROMPT_VERSION,  # e.g. "v2"
    "active_model_version": None,  # int, None until a model is trained
    "confidence_threshold": config.DEFAULT_CONFIDENCE_THRESHOLD,
    "openai_model": config.DEFAULT_OPENAI_MODEL,
}


def load() -> dict[str, Any]:
    """Read settings.json merged over the defaults. Unknown keys are kept."""
    merged = dict(DEFAULTS)
    path = config.settings_path()
    if path.exists():
        try:
            merged.update(json.loads(path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            pass  # corrupt/unreadable file -> defaults
    return merged


def save(values: dict[str, Any]) -> dict[str, Any]:
    """Merge `values` into the stored settings and persist. Returns the result."""
    merged = load()
    merged.update(values)
    path = config.settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    return merged


def get(key: str) -> Any:
    return load().get(key, DEFAULTS.get(key))
