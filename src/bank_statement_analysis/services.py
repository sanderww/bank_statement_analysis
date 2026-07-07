import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from . import config, model_store, prompt_store
from .dedup import dedup_key
from . import direction
from .extract import clean_description, extract_transactions_from_pdf
from .categorize import categorize_rows

logger = logging.getLogger(__name__)


def enrich_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalise one statement's extracted rows and derive direction.

    - amount becomes a positive magnitude (direction carries the sign)
    - rows with an empty description or zero amount are dropped (extraction noise)
    - each row gains 'direction' ('in'/'out') and 'signed_amount' (+in/-out)

    Must be called per statement: direction inference uses the running balance.
    """
    cleaned = [
        {
            **r,
            "description": clean_description(str(r["description"])),
            "amount": round(abs(float(r["amount"])), 2),
            "balance": round(float(r["balance"]), 2),
        }
        for r in rows
        if r.get("description") and abs(float(r.get("amount") or 0)) > 0
    ]
    # cleaning can empty a description that was pure noise — drop those rows
    cleaned = [r for r in cleaned if r["description"]]
    return direction.derive(cleaned)


def dedup_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop duplicate transactions across statements (overlapping statement
    periods), keeping the first-seen row. Key: date|description|amount|balance."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = dedup_key(r["date"], r["description"], float(r["amount"]), float(r["balance"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def extract_data(pdf_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Extract transactions from the provided PDFs. Each statement's rows are
    normalised and direction-tagged individually; the combined result is
    deduplicated so overlapping statements never double-count.
    """
    all_rows: List[Dict[str, Any]] = []
    for p in pdf_paths:
        if not p.exists():
            logger.warning(f"File not found: {p}")
            continue

        logger.info(f"Extracting from {p} ...")

        try:
            rows = enrich_rows(extract_transactions_from_pdf(str(p)))
            for r in rows:
                r["source_statement"] = p.name  # traceability back to the PDF
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"Error extracting from {p}: {e}")
            raise e

    return dedup_rows(all_rows)

def load_csv_data(csv_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Iterates over provided CSV paths and loads transactions.
    """
    from .io_utils import read_csv
    
    all_rows: List[Dict[str, Any]] = []
    for p in csv_paths:
        if not p.exists():
            logger.warning(f"File not found: {p}")
            continue
            
        logger.info(f"Loading from {p} ...")
        try:
            rows = read_csv(str(p))
            # Ensure numeric fields are floats
            for r in rows:
                if "amount" in r:
                    r["amount"] = float(r["amount"])
                if "balance" in r:
                    r["balance"] = float(r["balance"])
            # Older extracted CSVs predate direction/signed_amount — derive them
            # (per file: direction inference follows one running balance).
            if rows and not rows[0].get("direction"):
                rows = enrich_rows(rows)
            else:
                for r in rows:
                    r["signed_amount"] = float(r.get("signed_amount") or 0.0)
            for r in rows:
                if not r.get("source_statement"):
                    r["source_statement"] = p.name
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"Error loading {p}: {e}")
            raise e

    return dedup_rows(all_rows)

def categorize_data(
    rows: List[Dict[str, Any]],
    mode: str = "openai",
    model: str = "gpt-5-mini",
    prompt_version: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Categorizes the extracted transaction rows in place: sets 'category',
    'category_label', 'source' and (for the local model) 'confidence'.

    Local mode uses the active model version (settings.json); OpenAI mode uses
    the active prompt version unless one is given explicitly.
    """
    if not rows:
        return rows

    if mode.lower() == "local":
        # Rows from extract_data/load_csv_data always carry signed_amount.
        model_store.predict_rows(rows)
        return rows

    if mode.lower() != "openai":
        raise ValueError("categorize_mode must be 'openai' or 'local'")

    version = prompt_version or prompt_store.active_version()
    categorize_rows(rows, model=model, prompt_version=version)
    return rows
