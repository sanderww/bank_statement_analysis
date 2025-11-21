import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pdfplumber


DATE_HEADER_RE = re.compile(r"Statement (Period|Date)\s*:\s*(.+)", re.IGNORECASE)
SECTION_START_RE = re.compile(r"^\s*Transactions in RAND", re.IGNORECASE)
HEADER_RE = re.compile(r"Date\s+Description\s+Amount\s+Balance", re.IGNORECASE)
# Match a complete transaction: date, description, amount, balance
# The lookahead ensures we stop before a new date pattern or end of string
# We use a more specific pattern: description should not end with a date pattern
ROW_RE = re.compile(
    r"(?P<date>\d{1,2}\s\w{3})\s+(?P<desc>(?:(?!\d{1,2}\s\w{3}\s).)+?)\s+(?P<amount>-?[\d,]+\.\d{2})\s+(?P<balance>-?[\d,]+\.\d{2})(?:\s*(?:Cr|Dr))?(?=\s+\d{1,2}\s\w{3}\s|$)"
)
# Pattern to match a transaction without date prefix (uses same date as previous transaction)
# Pattern 1: amount + description + balance (full transaction)
# Pattern 2: amount + description (charge/fee, no balance shown)
TRANSACTION_WITHOUT_DATE_FULL_RE = re.compile(
    r"(?P<amount>-?[\d,]+\.\d{2})\s+(?P<desc>.+?)\s+(?P<balance>-?[\d,]+\.\d{2})(?:\s*(?:Cr|Dr))?(?=\s+\d{1,2}\s\w{3}\s|\s+-?[\d,]+\.\d{2}\s|$)"
)
# Pattern for transactions with just amount + description (no balance)
# This typically happens for bank charges/fees that appear after the main transaction
# Description starts with capital letter and continues until we see a date pattern or another amount
TRANSACTION_WITHOUT_DATE_AMOUNT_ONLY_RE = re.compile(
    r"(?P<amount>-?[\d,]+\.\d{2})\s+(?P<desc>(?:(?!\d{1,2}\s\w{3}\s|\s+-?[\d,]+\.\d{2}\s).)+?)(?=\s+\d{1,2}\s\w{3}\s|\s+-?[\d,]+\.\d{2}\s|$)"
)
# Pattern to detect if a line starts with a date (for continuation detection)
DATE_START_RE = re.compile(r"^(?P<date>\d{1,2}\s\w{3})\s+(?P<desc>.+)$")


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
    return float(num_str.replace(",", ""))


def _normalize_date(day_mon: str, year: int) -> str:
    # day_mon like "23 Apr"
    dt = datetime.strptime(f"{day_mon} {year}", "%d %b %Y")
    return dt.strftime("%d-%m-%Y")


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
            yield line


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

                # Find ALL complete transactions on this line
                # We need to be careful: the description might contain numbers that look like amounts
                # So we look for the pattern: date + description + amount + balance
                # where amount and balance are followed by either Cr/Dr, whitespace+date, or end of line
                matches = list(ROW_RE.finditer(line))
                if matches:
                    last_date_token = None
                    last_match_end = 0
                    
                    # Process all complete transactions found on this line
                    for idx, m in enumerate(matches):
                        date_token = m.group("date")
                        desc = m.group("desc").strip()
                        amount = _normalize_number(m.group("amount"))
                        balance = _normalize_number(m.group("balance"))

                        # Clean up description: remove any trailing amounts/balances that might have been captured
                        # This can happen if the regex is too greedy
                        desc = desc.strip()
                        
                        # If a previous wrapped description exists, merge
                        if pending_desc and pending_prefix_date == date_token:
                            desc = f"{pending_desc} {desc}".strip()
                            pending_desc = None
                            pending_prefix_date = None

                        results.append(
                            {
                                "date": _normalize_date(date_token, year),
                                "description": desc,
                                "amount": amount,
                                "balance": balance,
                            }
                        )
                        
                        last_date_token = date_token
                        last_match_end = m.end()
                    
                    # After processing all transactions with dates, check if there's remaining text
                    # that might contain additional transactions without date prefixes
                    remaining_text = line[last_match_end:].strip()
                    if remaining_text and last_date_token:
                        # Strategy: Look for patterns that indicate a new transaction
                        # After a balance (with Cr/Dr), if we see an amount pattern, it's likely a new transaction
                        # Try to find full transactions (amount + description + balance) without date prefixes first
                        additional_matches = list(TRANSACTION_WITHOUT_DATE_FULL_RE.finditer(remaining_text))
                        processed_positions = set()
                        
                        for m in additional_matches:
                            desc = m.group("desc").strip()
                            amount = _normalize_number(m.group("amount"))
                            balance = _normalize_number(m.group("balance"))
                            
                            results.append(
                                {
                                    "date": _normalize_date(last_date_token, year),
                                    "description": desc,
                                    "amount": amount,
                                    "balance": balance,
                                }
                            )
                            # Track the position to avoid double-processing
                            processed_positions.add((m.start(), m.end()))
                        
                        # Also check for transactions with just amount + description (no balance shown)
                        # These typically appear after a balance with Cr/Dr
                        # Look for pattern: amount followed by text that looks like a description
                        # The description should start with a capital letter or common transaction keywords
                        amount_only_matches = list(TRANSACTION_WITHOUT_DATE_AMOUNT_ONLY_RE.finditer(remaining_text))
                        for m in amount_only_matches:
                            # Skip if this position overlaps with a full transaction match
                            overlaps = any(
                                not (m.end() <= start or m.start() >= end)
                                for start, end in processed_positions
                            )
                            if overlaps:
                                continue
                            
                            desc = m.group("desc").strip()
                            amount = _normalize_number(m.group("amount"))
                            
                            # Clean up description - remove any trailing numbers that might have been captured
                            # Description should end before any date pattern or amount pattern
                            desc = desc.strip()
                            
                            # For transactions without balance, calculate it from the previous balance
                            # Balance change = previous balance + amount (amount can be negative for debits)
                            prev_balance = results[-1]["balance"] if results else 0.0
                            estimated_balance = prev_balance + amount
                            
                            results.append(
                                {
                                    "date": _normalize_date(last_date_token, year),
                                    "description": desc,
                                    "amount": amount,
                                    "balance": estimated_balance,
                                }
                            )
                    continue

                # Continuation line (likely wrapped description)
                if pending_desc is None:
                    # Try to detect a date at the beginning even if amounts are missing
                    m2 = DATE_START_RE.match(line)
                    if m2:
                        pending_prefix_date = m2.group("date")
                        pending_desc = m2.group("desc").strip()
                        continue
                    else:
                        # No date: if last row exists, append to its description
                        # But only if it doesn't look like a new transaction
                        # Check if line looks like it might be a continuation (no date pattern at start)
                        if results and not DATE_START_RE.match(line):
                            results[-1]["description"] = (
                                f"{results[-1]['description']} {line}".strip()
                            )
                        continue
                else:
                    # Check if this line starts a new transaction
                    m2 = DATE_START_RE.match(line)
                    if m2:
                        # This looks like a new transaction, save the pending one if it has amounts
                        # (but it shouldn't have amounts if we're in pending state)
                        # Just update pending with new date/desc
                        pending_prefix_date = m2.group("date")
                        pending_desc = m2.group("desc").strip()
                    else:
                        pending_desc = f"{pending_desc} {line}".strip()

    return results


