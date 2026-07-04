"""Versioned categorisation prompts, stored as prompts/v{N}.txt files.

Files stay the storage format (they're reviewable in git and easy to edit by
hand); this module lists versions, reads the active one (from settings), and
creates new versions. The active version is whatever settings.json points at.
"""
from __future__ import annotations

import re
from typing import Any

from . import config, settings

_VERSION_RE = re.compile(r"^v(\d+)$")


def _version_number(name: str) -> int | None:
    m = _VERSION_RE.match(name)
    return int(m.group(1)) if m else None


def list_versions() -> list[dict[str, Any]]:
    """All prompt versions, newest first: {version, name, text, active}."""
    pdir = config.prompts_dir()
    active = active_version()
    versions: list[dict[str, Any]] = []
    if pdir.exists():
        for f in pdir.glob("v*.txt"):
            n = _version_number(f.stem)
            if n is None:
                continue
            versions.append({
                "version": f.stem,
                "number": n,
                "text": f.read_text(encoding="utf-8"),
                "active": f.stem == active,
            })
    versions.sort(key=lambda v: v["number"], reverse=True)
    return versions


def active_version() -> str:
    return str(settings.get("active_prompt_version"))


def read(version: str) -> str:
    path = config.prompts_dir() / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def active_text() -> str:
    return read(active_version())


def set_active(version: str) -> None:
    if not (config.prompts_dir() / f"{version}.txt").exists():
        raise FileNotFoundError(f"Prompt version does not exist: {version}")
    settings.save({"active_prompt_version": version})


def update(version: str, text: str) -> None:
    """Overwrite an existing prompt version's text in place."""
    if not text.strip():
        raise ValueError("Prompt text is empty.")
    path = config.prompts_dir() / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt version does not exist: {version}")
    path.write_text(text.strip() + "\n", encoding="utf-8")


def add_version(text: str, activate: bool = True) -> str:
    """Save `text` as the next v{N}.txt. Returns the new version name."""
    if not text.strip():
        raise ValueError("Prompt text is empty.")
    pdir = config.prompts_dir()
    pdir.mkdir(parents=True, exist_ok=True)
    numbers = [n for f in pdir.glob("v*.txt") if (n := _version_number(f.stem)) is not None]
    next_version = f"v{max(numbers, default=0) + 1}"
    (pdir / f"{next_version}.txt").write_text(text.strip() + "\n", encoding="utf-8")
    if activate:
        settings.save({"active_prompt_version": next_version})
    return next_version
