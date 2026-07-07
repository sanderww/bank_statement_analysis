"""LLM categorisation via the OpenAI structured-output API.

Design notes (2026-07-07 quality pass):
- Transactions are sent in BATCHES (one API call per chunk, not per row).
- The payload includes `direction` and the SIGNED amount — prompt v2 describes
  amounts that way, but the old code sent a positive magnitude with no
  direction, so the LLM had to guess income vs expense.
- The model returns only what it decides (index, category, controllable);
  dates/amounts/labels are never echoed back, which removes a whole class of
  hallucinated-field errors. Labels are derived locally from categories.py.
- A failed chunk falls back to Unknown for its rows instead of aborting.

Local-model categorisation lives in model_store.predict_rows.
"""
from __future__ import annotations

import logging
import os
from typing import Any, List, Optional, Sequence

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from . import config
from .categories import CATEGORY_LABELS, Category, default_controllable, label as category_label

# Load env from project root .env
load_dotenv()

logger = logging.getLogger(__name__)

BATCH_SIZE = 25


class CategoryResult(BaseModel):
    """The LLM's decision for one transaction, matched back by index."""
    index: int = Field(..., description="The 'index' of the transaction this result is for")
    category: Category = Field(..., description="Category enum 0..10 (0=Unknown, 10=Income)")
    controllable: bool = Field(
        ...,
        description="True if this is a controllable/consumption cost the account "
                    "holder can influence short-term (eating out, petrol, "
                    "electricity usage, groceries); False for fixed/committed "
                    "costs (mortgage, school fees, insurance) and for income.",
    )


class CategoryResults(BaseModel):
    results: List[CategoryResult]


def _build_system_prompt(version: str) -> str:
    prompt_path = config.prompts_dir() / f"{version}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text().strip()


def _payload(rows: Sequence[dict[str, Any]], offset: int) -> str:
    import json

    items = [
        {
            "index": offset + i,
            "date": r.get("date", ""),
            "direction": r.get("direction", ""),
            "amount": float(r.get("signed_amount") or 0.0),
            "description": r.get("description", ""),
        }
        for i, r in enumerate(rows)
    ]
    return (
        "Categorise every transaction below. Return one result per transaction, "
        "matched by 'index'.\n" + json.dumps(items, ensure_ascii=False)
    )


def _mark_unknown(row: dict[str, Any]) -> None:
    row["category"] = int(Category.UNKNOWN)
    row["category_label"] = CATEGORY_LABELS[Category.UNKNOWN]
    row["controllable"] = "yes" if default_controllable(int(Category.UNKNOWN)) else "no"
    row["source"] = "openai"


def categorize_rows(
    rows: Sequence[dict[str, Any]],
    model: str = config.DEFAULT_OPENAI_MODEL,
    prompt_version: Optional[str] = None,
    client: Any = None,
    batch_size: int = BATCH_SIZE,
) -> None:
    """Categorise `rows` in place via the OpenAI API: sets category,
    category_label, controllable and source='openai' per row. Rows need
    date, description, direction and signed_amount (the enriched schema).

    `client` is injectable for tests; by default an OpenAI client is created
    (requires OPENAI_API_KEY)."""
    if not rows:
        return
    if client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

    version = prompt_version or config.DEFAULT_PROMPT_VERSION
    system_prompt = _build_system_prompt(version)
    logger.info("Categorizing %d transactions with model '%s', prompt '%s', batches of %d",
                len(rows), model, version, batch_size)

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        try:
            resp = client.responses.parse(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": _payload(chunk, start)},
                ],
                text_format=CategoryResults,
            )
            by_index = {r.index: r for r in resp.output_parsed.results}
        except Exception as e:
            logger.warning("OpenAI parse failed for rows %d-%d: %s. Marking chunk Unknown.",
                           start, start + len(chunk) - 1, e)
            by_index = {}

        for i, row in enumerate(chunk):
            res = by_index.get(start + i)
            if res is None:
                _mark_unknown(row)
                continue
            code = int(res.category)
            row["category"] = code
            row["category_label"] = category_label(code)
            # income/unknown are never controllable, whatever the LLM says
            controllable = res.controllable and code not in (0, 10)
            row["controllable"] = "yes" if controllable else "no"
            row["source"] = "openai"
