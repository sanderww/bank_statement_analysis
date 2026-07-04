"""config.py resolves everything under the (overridable) data root."""
from __future__ import annotations

from bank_statement_analysis import config


def test_paths_follow_data_root_override(data_root):
    assert config.data_root() == data_root
    assert config.bank_statements_dir() == data_root / "bank_statements"
    assert config.extracted_raw_dir() == data_root / "output" / "extracted_raw"
    assert config.categorised_dir() == data_root / "output" / "categorised"
    assert config.training_data_dir() == data_root / "models" / "training_data"
    assert config.settings_path() == data_root / "settings.json"
    assert config.prompts_dir() == data_root / "prompts"


def test_ensure_dirs_creates_layout(data_root):
    for d in (
        config.bank_statements_dir(),
        config.extracted_raw_dir(),
        config.categorised_dir(),
        config.models_dir(),
        config.training_data_dir(),
        config.prompts_dir(),
    ):
        assert d.is_dir()


def test_defaults_without_override():
    # No env override in this test (no data_root fixture): everything hangs off
    # the repo checkout.
    assert config.data_root() == config.project_root()
    assert config.prompts_dir() == config.project_root() / "prompts"
