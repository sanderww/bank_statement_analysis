"""Shared fixtures. Every test runs against a throwaway data root so nothing
touches the real bank_statements/, output/, models/ or settings.json."""
from __future__ import annotations

import pytest

from bank_statement_analysis import config


@pytest.fixture()
def data_root(tmp_path, monkeypatch):
    """Point BSA_PROJECT_DATA (and prompts) at a tmp dir and create the layout."""
    monkeypatch.setenv("BSA_PROJECT_DATA", str(tmp_path))
    monkeypatch.setenv("BSA_PROMPTS_DIR", str(tmp_path / "prompts"))
    config.ensure_dirs()
    return tmp_path
