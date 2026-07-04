"""The curated training set, managed separately from day-to-day output.

Training data lives in models/training_data/ (gitignored) and only grows
deliberately: reviewed statements are *promoted* into it from the Review step.
Models are trained from this folder — never silently from raw categorised
output — so what the model learns from is always an explicit, inspectable set
of files. This is the fine-tune-over-time loop until the local model is good
enough.

A training CSV needs `description`, `category` and an amount column —
`signed_amount` preferred, `amount` (signed) accepted for legacy files.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

from . import config
from .categories import VALID_CODES
from .dedup import dedup_key

TRAINING_FILE = "training_data.csv"
TRAINING_COLUMNS = ["date", "description", "amount", "balance", "signed_amount", "category"]


def list_training_csvs() -> list[Path]:
    tdir = config.training_data_dir()
    if not tdir.exists():
        return []
    return sorted(tdir.glob("*.csv"))


def read_training_csv(path: str | Path) -> list[dict[str, Any]]:
    """Read one CSV into training rows {description, signed_amount, category}.
    Skips rows missing a description or a valid category code."""
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            desc = (raw.get("description") or "").strip()
            cat = _to_int(raw.get("category"))
            if not desc or cat is None or cat not in VALID_CODES:
                continue
            signed = raw.get("signed_amount")
            amount = _to_float(signed if signed not in (None, "") else raw.get("amount"))
            rows.append({"description": desc, "signed_amount": amount, "category": cat})
    return rows


def load_training_rows(paths: Iterable[str | Path] | None = None) -> list[dict[str, Any]]:
    """Concatenate training rows from the given CSVs (default: every CSV in the
    training data folder)."""
    if paths is None:
        paths = list_training_csvs()
    rows: list[dict[str, Any]] = []
    for p in paths:
        rows.extend(read_training_csv(p))
    return rows


def promote_rows(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Append reviewed, categorised rows to the training set, deduplicated on
    the transaction dedup key. Rows without a valid category are skipped.
    Returns {added, skipped_duplicate, skipped_invalid}."""
    config.ensure_dirs()
    path = config.training_data_dir() / TRAINING_FILE

    existing_keys: set[str] = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for raw in csv.DictReader(f):
                existing_keys.add(dedup_key(
                    raw.get("date") or "", raw.get("description") or "",
                    _to_float(raw.get("amount")), _to_float(raw.get("balance")),
                ))

    added = dup = invalid = 0
    file_exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRAINING_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for r in rows:
            cat = _to_int(r.get("category"))
            if not (r.get("description") or "").strip() or cat is None or cat not in VALID_CODES:
                invalid += 1
                continue
            key = dedup_key(r.get("date") or "", r["description"],
                            _to_float(r.get("amount")), _to_float(r.get("balance")))
            if key in existing_keys:
                dup += 1
                continue
            existing_keys.add(key)
            writer.writerow({
                "date": r.get("date", ""),
                "description": r["description"],
                "amount": _to_float(r.get("amount")),
                "balance": _to_float(r.get("balance")),
                "signed_amount": _to_float(r.get("signed_amount") or r.get("amount")),
                "category": cat,
            })
            added += 1
    return {"added": added, "skipped_duplicate": dup, "skipped_invalid": invalid}


def stats() -> dict[str, Any]:
    """Row/category counts per training CSV plus totals (for the Setup UI)."""
    files = []
    total_rows = 0
    categories: set[int] = set()
    for p in list_training_csvs():
        rows = read_training_csv(p)
        cats = {r["category"] for r in rows}
        files.append({"name": p.name, "rows": len(rows), "categories": len(cats)})
        total_rows += len(rows)
        categories |= cats
    return {"files": files, "total_rows": total_rows, "total_categories": len(categories)}


def _to_int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float:
    try:
        return float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return 0.0
