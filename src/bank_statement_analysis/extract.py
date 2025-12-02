import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, List, Dict

import pdfplumber

# --- CONFIGURATION & REGEX PATTERNS ---

# 1. HEADER & SECTION MARKERS
DATE_HEADER_RE = re.compile(r"Statement (Period|Date)\s*:\s*(.+)", re.IGNORECASE)
SECTION_START_RE = re.compile(r"^\s*Transactions in RAND", re.IGNORECASE)
HEADER_RE = re.compile(r"Date\s+Description\s+Amount\s+Balance", re.IGNORECASE)

# NEW: End of section markers to stop processing when hitting the footer
SECTION_END_RE = re.compile(r"^\s*(?:Closing Balance|Turnover for Statement Period)", re.IGNORECASE)

# 2. CORE PATTERNS
# Matches "25 Jan", "25-Jan"
DATE_PATTERN = r"\d{1,2}[\s-][A-Za-z0-9]{3}(?:[\s-]\d{2,4})?"

# Matches "1,000.00", "74,602.52Cr", "500.00 Dr"
NUMBER_PATTERN = r"-?[\d,]+\.\d{2}(?:\s?[CcDd][Rr])?"

# 3. ROW_RE (Main Transaction with Date)
ROW_RE = re.compile(
    rf"(?P<date>{DATE_PATTERN})\s+"
    rf"(?P<desc>.+?)\s+" 
    rf"(?P<amount>{NUMBER_PATTERN})\s*"
    rf"(?P<balance>{NUMBER_PATTERN})",
    re.IGNORECASE
)

# 4. SECONDARY TRANSACTIONS (Hidden on same line)
TRANSACTION_WITHOUT_DATE_FULL_RE = re.compile(
    rf"(?P<desc>[A-Z0-9].+?)\s+"
    rf"(?P<amount>{NUMBER_PATTERN})\s+"
    rf"(?P<balance>{NUMBER_PATTERN})",
    re.IGNORECASE
)

# Fallback: Just Amount + Description
TRANSACTION_WITHOUT_DATE_DESC_AMOUNT_RE = re.compile(
    rf"(?P<desc>[A-Z0-9].+?)\s+"
    rf"(?P<amount>{NUMBER_PATTERN})",
    re.IGNORECASE
)

# 5. CONTINUATION CHECK
DATE_START_RE = re.compile(rf"^(?P<date>{DATE_PATTERN})\s+(?P<desc>.+)$")


# --- HELPER FUNCTIONS ---

def _infer_year_from_page(text: str) -> Optional[int]:
    for line in text.splitlines():
        m = DATE_HEADER_RE.search(line)
        if not m:
            continue
        # Try to find a 4-digit year in the captured text
        years = re.findall(r"(20\d{2})", m.group(2))
        if years:
            return int(years[-1])
    return None


def _normalize_number(num_str: str) -> float:
    """
    Converts '1,000.00Cr' -> 1000.00 (or -1000.00 depending on logic).
    """
    if not num_str:
        return 0.0
    
    clean_str = num_str.replace(",", "").lower().strip()
    
    # Handle Credit/Debit markers if present
    multiplier = 1.0
    if "dr" in clean_str:
        multiplier = -1.0
        clean_str = clean_str.replace("dr", "")
    elif "cr" in clean_str:
        clean_str = clean_str.replace("cr", "")
        
    try:
        return float(clean_str) * multiplier
    except ValueError:
        return 0.0


def _normalize_date(day_mon: str, year: int) -> str:
    try:
        clean_dm = day_mon.replace("-", " ").strip()
        dt = datetime.strptime(f"{clean_dm} {year}", "%d %b %Y")
        return dt.strftime("%d-%m-%Y")
    except ValueError:
        return f"{day_mon} {year}"


def _iter_section_lines(page_text: str) -> Iterable[str]:
    lines = page_text.splitlines()
    in_section = False
    saw_header = False
    for line in lines:
        if not in_section and SECTION_START_RE.search(line):
            in_section = True
            continue
        if in_section and not saw_header and HEADER_RE.search(line):
            saw_header = True
            continue
        if in_section and saw_header:
            # Check for end of section markers (Closing Balance / Turnover info)
            if SECTION_END_RE.search(line):
                in_section = False
                saw_header = False
                continue
                
            yield line


# --- MAIN EXTRACTION LOGIC ---

def extract_transactions_from_pdf(pdf_path: str) -> list[dict]:
    results: list[dict] = []
    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(pdf_path)

    with pdfplumber.open(str(path_obj)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            year = _infer_year_from_page(text) or datetime.today().year

            pending_desc: Optional[str] = None
            pending_prefix_date: Optional[str] = None

            for raw in _iter_section_lines(text):
                line = raw.strip()
                if not line:
                    continue

                # 1. Attempt to find standard transactions (Date + Desc + Amt + Bal)
                matches = list(ROW_RE.finditer(line))
                
                if matches:
                    last_date_token = None
                    last_match_end = 0
                    
                    # Process all primary matches on this line
                    for m in matches:
                        date_token = m.group("date")
                        desc = m.group("desc").strip()
                        amount = _normalize_number(m.group("amount"))
                        balance = _normalize_number(m.group("balance"))

                        # Merge with previous wrapped description if applicable
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
                    
                    # 2. Handle "Hidden" Transactions in Remaining Text
                    remaining_text = line[last_match_end:].strip()
                    
                    if remaining_text and last_date_token:
                        # Attempt to find full transactions (Desc + Amt + Bal)
                        additional_matches = list(TRANSACTION_WITHOUT_DATE_FULL_RE.finditer(remaining_text))
                        processed_positions = set()
                        
                        for m in additional_matches:
                            desc = m.group("desc").strip()
                            amount = _normalize_number(m.group("amount"))
                            balance = _normalize_number(m.group("balance"))
                            
                            results.append({
                                "date": _normalize_date(last_date_token, year),
                                "description": desc,
                                "amount": amount,
                                "balance": balance,
                            })
                            processed_positions.add((m.start(), m.end()))
                        
                        # Attempt fallback for transactions without balance shown (Desc + Amt)
                        if not additional_matches:
                            amount_only_matches = list(TRANSACTION_WITHOUT_DATE_DESC_AMOUNT_RE.finditer(remaining_text))
                            for m in amount_only_matches:
                                desc = m.group("desc").strip()
                                amount = _normalize_number(m.group("amount"))
                                
                                prev_balance = results[-1]["balance"] if results else 0.0
                                estimated_balance = prev_balance + amount
                                
                                results.append({
                                    "date": _normalize_date(last_date_token, year),
                                    "description": desc,
                                    "amount": amount,
                                    "balance": estimated_balance,
                                })

                    continue

                # 3. Handle Continuation Lines
                if pending_desc is None:
                    m2 = DATE_START_RE.match(line)
                    if m2:
                        pending_prefix_date = m2.group("date")
                        pending_desc = m2.group("desc").strip()
                        continue
                    else:
                        if results and not DATE_START_RE.match(line):
                            results[-1]["description"] = (
                                f"{results[-1]['description']} {line}".strip()
                            )
                        continue
                else:
                    m2 = DATE_START_RE.match(line)
                    if m2:
                        pending_prefix_date = m2.group("date")
                        pending_desc = m2.group("desc").strip()
                    else:
                        pending_desc = f"{pending_desc} {line}".strip()

    return results