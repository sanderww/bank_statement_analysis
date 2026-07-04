import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from . import config
from .extract import extract_transactions_from_pdf
from .categorize import Transaction, categorize_transactions, categorize_transactions_local

logger = logging.getLogger(__name__)

def extract_data(pdf_paths: List[Path]) -> List[Dict[str, Any]]:
    """
    Iterates over the provided PDF paths and extracts transactions.
    Returns a list of transaction dictionaries.
    """
    all_rows: List[Dict[str, Any]] = []
    for p in pdf_paths:
        if not p.exists():
            logger.warning(f"File not found: {p}")
            continue
        
        # Log to console or logger depending on configuration. 
        # For CLI usage, we might want to print, but for now let's stick to logging 
        # or just let the caller handle progress indication if needed.
        # Since the original CLI printed "Extracting from ...", we might lose that 
        # specific per-file print unless we pass a callback or just log it.
        logger.info(f"Extracting from {p} ...")
        
        try:
            rows = extract_transactions_from_pdf(str(p))
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"Error extracting from {p}: {e}")
            # In server.py we raised 500. In main.py we just crashed? 
            # main.py didn't have try/except block around extract call.
            # We'll re-raise for now to be safe or just log. 
            # Given the server logic, it might be better to propagate the error 
            # if we want to stop, or continue if we want partial results.
            # The server logic raised HTTPException.
            raise e 

    return all_rows

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
            all_rows.extend(rows)
        except Exception as e:
            logger.error(f"Error loading {p}: {e}")
            raise e
            
    return all_rows

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
    
    return rows
