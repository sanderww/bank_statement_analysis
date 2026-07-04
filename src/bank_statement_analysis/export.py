"""Export the final, decoupled CSV for a downstream budget app.

Reads a reviewed categorised CSV and writes a self-contained file to
output/final/ — no balances, working flags or confidences, just what a budget
app needs. Only categorised rows are exported; each row keeps its
source_statement for traceability. Ported in spirit from the prototype's
export.py.
"""
from __future__ import annotations

import csv
from datetime import datetime

from . import config
from .categories import label as category_label
from .io_utils import read_csv

FINAL_COLUMNS = [
    "date", "description", "amount", "direction",
    "category", "category_label", "source_statement",
]


def export_final(categorised_filename: str) -> dict:
    """Write the categorised rows of one reviewed file to output/final/.
    Returns {path, name, rows, skipped_uncategorised}."""
    source = config.categorised_dir() / categorised_filename
    if not source.exists():
        raise FileNotFoundError(f"Categorised CSV not found: {categorised_filename}")

    rows = read_csv(str(source))
    config.ensure_dirs()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = config.final_dir() / f"final_{stamp}.csv"

    written = skipped = 0
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(FINAL_COLUMNS)
        for r in rows:
            cat = r.get("category")
            if cat in (None, ""):
                skipped += 1
                continue
            cat = int(float(cat))
            signed = r.get("signed_amount") or r.get("amount") or 0
            writer.writerow([
                r.get("date", ""),
                r.get("description", ""),
                f"{float(signed):.2f}",
                r.get("direction", ""),
                cat,
                category_label(cat),
                r.get("source_statement", ""),
            ])
            written += 1

    return {"path": str(out_path), "name": out_path.name,
            "rows": written, "skipped_uncategorised": skipped}
