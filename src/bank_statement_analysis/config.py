"""Central paths and tunable defaults.

Every data folder hangs off the project root by default, but the whole data
layout can be relocated with the BSA_PROJECT_DATA environment variable (tests
point it at a tmp dir; the other machine can point it anywhere). Prompts are
part of the repo but can be overridden separately with BSA_PROMPTS_DIR.

Paths are exposed as functions (not module constants) so environment overrides
always take effect, regardless of import order.
"""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def data_root() -> Path:
    """Root for all user data (statements, output, models, settings)."""
    env = os.environ.get("BSA_PROJECT_DATA")
    return Path(env) if env else project_root()


def bank_statements_dir() -> Path:
    return data_root() / "bank_statements"


def output_dir() -> Path:
    return data_root() / "output"


def extracted_raw_dir() -> Path:
    return output_dir() / "extracted_raw"


def categorised_dir() -> Path:
    return output_dir() / "categorised"


def models_dir() -> Path:
    return data_root() / "models"


def training_data_dir() -> Path:
    return models_dir() / "training_data"


def prompts_dir() -> Path:
    env = os.environ.get("BSA_PROMPTS_DIR")
    return Path(env) if env else project_root() / "prompts"


def settings_path() -> Path:
    return data_root() / "settings.json"


def ensure_dirs() -> None:
    """Create the runtime folder layout if missing. Safe to call repeatedly."""
    for d in (
        bank_statements_dir(),
        extracted_raw_dir(),
        categorised_dir(),
        models_dir(),
        training_data_dir(),
        prompts_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)


# --- Tunable defaults (overridable via settings.json, see settings.py) ------

# Model predictions below this probability are flagged for review.
DEFAULT_CONFIDENCE_THRESHOLD = 0.70

# Direction inference: |balance delta| vs amount magnitude tolerance.
BALANCE_MATCH_TOLERANCE = 0.05

# Prompt version used when settings.json doesn't specify one.
DEFAULT_PROMPT_VERSION = "v2"

# OpenAI model default for LLM categorisation.
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
