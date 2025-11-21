import re
import pdfplumber
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

# --- CONFIGURATION & REGEX PATTERNS ---

# 1. DATE: Matches "25 Jan", "25-Jan", "2023/01/01"
# We keep it flexible for typical bank statement formats.
DATE_PATTERN = r"\d{1,4}[/\s-][A-Za-z0-9]{3,9}(?:[/\s-]\d{2,4})?"

# 2. NUMBER: Matches "1,000.00", "1000.00", "74,602.52Cr", "500.00 Dr"
# Crucial: It handles the 'Cr'/'Dr' suffix which acts as a transaction delimiter.
NUMBER_PATTERN = r"-?[\d,]+\.\d{1,2}(?:\s?[CcDd][Rr])?"

# 3. ROW_RE (Main Transaction with Date)
# Note the `+?` in the description group. This makes it NON-GREEDY.
# It stops at the *first* occurrence of the Amount/Balance pattern.
ROW_RE = re.compile(
    rf"(?P<date>{DATE_PATTERN})\s+"
    rf"(?P<desc>.+?)\s+" 
    rf"(?P<amount>{NUMBER_PATTERN})\s+"
    rf"(?P<balance>{NUMBER_PATTERN})",
    re.IGNORECASE
)

# 4. TRANSACTION_WITHOUT_DATE_FULL_RE (Secondary Transaction on same line)
# Catches: "Payment To Inv... 26000.00 51700.76"
# It looks for text start, followed eventually by two numbers at the end.
TRANSACTION_WITHOUT_DATE_FULL_RE = re.compile(
    rf"(?P<desc>[A-Z0-9].+?)\s+" # Description usually starts with char/num
    rf"(?P<amount>{NUMBER_PATTERN})\s+"
    rf"(?P<balance>{NUMBER_PATTERN})",
    re.IGNORECASE
)

# 5. TRANSACTION_WITHOUT_DATE_AMOUNT_ONLY_RE (Fallback)
TRANSACTION_WITHOUT_DATE_AMOUNT_ONLY_RE = re.compile(
    rf"(?P<desc>[A-Z0-9].+?)\s+"
    rf"(?P<amount>{NUMBER_PATTERN})",
    re.IGNORECASE
)

# --- HELPER FUNCTIONS ---

def _normalize_number(num_str: str) -> float:
    """Converts '1,000.00Cr' -> -1000.00 or '1000.00' -> 1000.00"""
    if not num_str:
        return 0.0
    
    clean_str = num_str.replace(",", "").lower().strip()
    
    # Handle Credit/Debit markers
    # Note: Logic depends on bank. Usually Dr is negative (money out), Cr is positive or debt reduction.
    # Adjust multiplier based on specific bank logic. Assuming Dr = -, Cr = + for this example.
    multiplier = 1.0
    if "dr" in clean_str:
        multiplier = -1.0
        clean_str = clean_str.replace("dr", "")
    elif "cr" in clean_str:
        # Often bank statements show Cr as credit to account, but sometimes Cr is 'Credit Card Balance'
        # For now, we just strip it.
        clean_str = clean_str.replace("cr", "")
        
    try:
        return float(clean_str) * multiplier
    except ValueError:
        return 0.0

def _normalize_date(date_str: str, year: int) -> str:
    """Standardizes date string."""
    return f"{date_str} {year}" # Simplified for this demo

def _infer_year_from_page(text: str) -> int:
    """Mock function to infer year."""
    return datetime.today().year

def _iter_section_lines(text: str):
    """Generator to iterate lines."""
    for line in text.split('\n'):
        yield line

# --- MAIN LOGIC ---

def extract_transactions_from_line(line: str, year: int, pending_desc=None, pending_prefix_date=None) -> tuple[List[dict], Optional[str], Optional[str]]:
    """
    Refactored core logic to handle single-line processing.
    Returns: (results_list, new_pending_desc, new_pending_prefix_date)
    """
    results = []
    line = line.strip()
    if not line:
        return results, pending_desc, pending_prefix_date

    # 1. Search for standard Date-led transactions
    matches = list(ROW_RE.finditer(line))
    
    last_date_token = None
    last_match_end = 0

    if matches:
        # Process all standard matches (Date + Desc + Amt + Bal)
        for m in matches:
            date_token = m.group("date")
            desc = m.group("desc").strip()
            amount = _normalize_number(m.group("amount"))
            balance = _normalize_number(m.group("balance"))

            # Handle wrapped descriptions from previous lines
            if pending_desc and pending_prefix_date == date_token:
                desc = f"{pending_desc} {desc}".strip()
                pending_desc = None
                pending_prefix_date = None

            results.append({
                "date": _normalize_date(date_token, year),
                "description": desc,
                "amount": amount,
                "balance": balance,
            })
            
            last_date_token = date_token
            last_match_end = m.end()

        # 2. Handle "Hidden" Second Transactions
        # This handles the case: "77,700.76Cr Payment To Investment..."
        remaining_text = line[last_match_end:].strip()
        
        if remaining_text and last_date_token:
            # Try finding a full transaction (Desc + Amt + Bal) in the remainder
            secondary_matches = list(TRANSACTION_WITHOUT_DATE_FULL_RE.finditer(remaining_text))
            
            for sm in secondary_matches:
                desc = sm.group("desc").strip()
                amount = _normalize_number(sm.group("amount"))
                balance = _normalize_number(sm.group("balance"))

                results.append({
                    "date": _normalize_date(last_date_token, year), # Reuse date
                    "description": desc,
                    "amount": amount,
                    "balance": balance,
                })
                last_match_end += sm.end() # Advance cursor

    # 3. Handle Continuation Lines (Wrapped descriptions)
    # If no matches were found at all on this line, check if it's a wrapped text
    if not matches and not results:
        # Logic to check if this is a new date start or a continuation
        # (Simplified for this snippet)
        pass 

    return results, pending_desc, pending_prefix_date

def extract_transactions_from_pdf(pdf_path: str) -> list[dict]:
    results: list[dict] = []
    path_obj = Path(pdf_path)
    
    # For the purpose of this script, we skip the file check if running the test below
    # if not path_obj.exists(): raise FileNotFoundError(pdf_path)

    with pdfplumber.open(str(path_obj)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            year = _infer_year_from_page(text)
            
            pending_desc = None
            pending_prefix_date = None

            for raw in _iter_section_lines(text):
                line_res, pending_desc, pending_prefix_date = extract_transactions_from_line(
                    raw, year, pending_desc, pending_prefix_date
                )
                results.extend(line_res)

    return results

# --- TESTING THE SPECIFIC ISSUE ---

if __name__ == "__main__":
    print("--- Running Test Case ---")
    
    # Simulating the "mashed" line from the PDF extraction.
    # Note: I reconstructed the spacing based on standard PDF output logic.
    # The 'Cr' acts as a boundary.
    test_line = "01 Jan FNB OB Pmt Jicamasalariwie8508 74,602.52Cr 77,700.76Cr Payment To Investment Maint Back 26,000.00 51,700.76"
    
    print(f"Input Line: {test_line}\n")
    
    extracted, _, _ = extract_transactions_from_line(test_line, 2023)
    
    for i, tx in enumerate(extracted, 1):
        print(f"Transaction {i}:")
        print(f"  Date: {tx['date']}")
        print(f"  Desc: {tx['description']}")
        print(f"  Amt:  {tx['amount']}")
        print(f"  Bal:  {tx['balance']}")
        print("-" * 20)

    # Verification Logic
    if len(extracted) == 2:
        print("SUCCESS: 2 transactions found.")
        if "Payment To Investment" in extracted[1]['description']:
            print("SUCCESS: Second description captured correctly.")
    else:
        print(f"FAILURE: Found {len(extracted)} transactions, expected 2.")