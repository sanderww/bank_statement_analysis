import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from . import config
from .dedup import dedup_key
from . import direction
from .extract import extract_transactions_from_pdf
from .categorize import Transaction, categorize_transactions, categorize_transactions_local

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
            "amount": round(abs(float(r["amount"])), 2),
            "balance": round(float(r["balance"]), 2),
        }
        for r in rows
        if r.get("description") and abs(float(r.get("amount") or 0)) > 0
    ]
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
            rows = extract_transactions_from_pdf(str(p))
            all_rows.extend(enrich_rows(rows))
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
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"Error loading {p}: {e}")
            raise e

    return dedup_rows(all_rows)

def categorize_data(
    rows: List[Dict[str, Any]],
    mode: str = "openai",
    model: str = "gpt-5-mini",
    prompt_version: str = config.DEFAULT_PROMPT_VERSION,
    local_model_path: Optional[Path] = None
) -> List[Dict[str, Any]]:
    """
    Categorizes the extracted transaction rows.
    Updates the rows in-place (or returns them) with 'category' and 'category_label'.
    """
    if not rows:
        return rows

    # Convert to Transaction objects
    tx_models = [
        Transaction(
            date=r["date"],
            description=r["description"],
            amount=float(r["amount"]),
            balance=float(r["balance"]),
        )
        for r in rows
    ]

    if mode.lower() == "local":
        categorized = categorize_transactions_local(tx_models, model_path=local_model_path)
    elif mode.lower() == "openai":
        categorized = categorize_transactions(tx_models, model=model, prompt_version=prompt_version)
    else:
        raise ValueError("categorize_mode must be 'openai' or 'local'")

    # Merge category back
    for i, c in enumerate(categorized):
        rows[i]["category"] = int(c.category)
        rows[i]["category_label"] = c.category_label
        rows[i]["source"] = mode.lower()

    return rows
