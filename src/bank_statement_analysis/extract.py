import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional, List, Dict

# --- CONFIGURATION & REGEX PATTERNS ---

# 1. HEADER & SECTION MARKERS
DATE_HEADER_RE = re.compile(r"Statement (Period|Date)\s*:\s*(.+)", re.IGNORECASE)
PERIOD_RE = re.compile(
    r"Statement Period\s*:\s*(\d{1,2}\s+[A-Za-z]+\s+\d{4})\s+to\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
    re.IGNORECASE,
)
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


# --- DESCRIPTION CLEANING ---

# Page furniture that pdfplumber sometimes glues onto transaction lines /
# wrapped descriptions. Everything from the first marker onwards is dropped —
# crucially this includes bank headers carrying the ACCOUNT NUMBER, which must
# never leak into extracted descriptions.
_NOISE_CUT_RE = re.compile(
    r"\s*(?:Page\s*\d+\s*of\s*\d+|Delivery Method\b|Branch Number\b|"
    r"Statement Period\s*:|Statement Date\s*:|Tax Invoice\b|VAT Registration\b).*$",
    re.IGNORECASE,
)
# Trailing table-column tokens that bleed into the description column.
_TRAILING_NOISE_RE = re.compile(r"\s+(?:Charges|Accrued|Accrued Bank Charges)$", re.IGNORECASE)
# Leftover amount/balance tokens from a mashed-up previous transaction.
_LEADING_AMOUNTS_RE = re.compile(rf"^(?:{NUMBER_PATTERN}\s+)+")


def clean_description(desc: str) -> str:
    """Strip extraction noise from a transaction description: page furniture
    (incl. account-number headers), leading leftover amount tokens, trailing
    column-header tokens, and redundant whitespace."""
    desc = _NOISE_CUT_RE.sub("", desc)
    desc = _LEADING_AMOUNTS_RE.sub("", desc)
    prev = None
    while prev != desc:  # several trailing tokens can stack up
        prev = desc
        desc = _TRAILING_NOISE_RE.sub("", desc)
    return " ".join(desc.split())


# --- YEAR INFERENCE (document-level) ---

class YearContext:
    """Resolve the year for a day+month token using the statement period.

    The old per-page inference broke multi-page statements: pages without a
    'Statement Period' header silently fell back to *today's* year, splitting
    one statement across two years. The context is now built once from the
    whole document. When the period spans a year boundary (Dec -> Jan), each
    transaction gets the year that puts it inside the period.
    """

    def __init__(self, start: Optional[date] = None, end: Optional[date] = None,
                 fallback_year: Optional[int] = None):
        self.start = start
        self.end = end
        self.fallback_year = fallback_year or (start.year if start else None) \
            or datetime.today().year

    def resolve(self, day: int, month: int) -> int:
        if self.start and self.end:
            slack = timedelta(days=7)  # statements can show a txn just outside the period
            for year in {self.end.year, self.start.year}:
                try:
                    d = date(year, month, day)
                except ValueError:
                    continue
                if self.start - slack <= d <= self.end + slack:
                    return year
        return self.fallback_year


def _parse_header_date(raw: str) -> Optional[date]:
    raw = raw.strip()
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


def build_year_context(full_text: str) -> YearContext:
    """Year context from the whole document: prefer the statement period,
    fall back to any year in a Statement Period/Date header, then today."""
    m = PERIOD_RE.search(full_text)
    if m:
        start, end = _parse_header_date(m.group(1)), _parse_header_date(m.group(2))
        if start and end and start <= end:
            return YearContext(start=start, end=end)

    for line in full_text.splitlines():
        h = DATE_HEADER_RE.search(line)
        if h:
            years = re.findall(r"(20\d{2})", h.group(2))
            if years:
                return YearContext(fallback_year=int(years[-1]))
    return YearContext()


# --- HELPER FUNCTIONS ---


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


def _normalize_date(day_mon: str, ctx: YearContext) -> str:
    clean_dm = day_mon.replace("-", " ").strip()
    try:
        # anchor to a leap year so '29 Feb' parses; the real year comes from ctx
        dt = datetime.strptime(f"{clean_dm} 2000", "%d %b %Y")
    except ValueError:
        return clean_dm  # unparseable token — keep as-is rather than invent a date
    year = ctx.resolve(dt.day, dt.month)
    try:
        return date(year, dt.month, dt.day).strftime("%d-%m-%Y")
    except ValueError:  # e.g. 29 Feb resolved into a non-leap year
        return f"{clean_dm} {year}"


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

def extract_transactions_from_text(pages: List[str]) -> list[dict]:
    """Extract transactions from already-extracted page texts.

    Split out from the PDF wrapper so the parsing logic is unit-testable
    without PDFs. The year context is built once from the whole document —
    pages without their own 'Statement Period' header no longer fall back to
    today's year (which used to split one statement across two years).
    """
    ctx = build_year_context("\n".join(pages))
    results: list[dict] = []

    for text in pages:
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
                        "date": _normalize_date(date_token, ctx),
                        "description": clean_description(desc),
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

                    for m in additional_matches:
                        results.append({
                            "date": _normalize_date(last_date_token, ctx),
                            "description": clean_description(m.group("desc").strip()),
                            "amount": _normalize_number(m.group("amount")),
                            "balance": _normalize_number(m.group("balance")),
                        })

                    # Attempt fallback for transactions without balance shown (Desc + Amt)
                    if not additional_matches:
                        for m in TRANSACTION_WITHOUT_DATE_DESC_AMOUNT_RE.finditer(remaining_text):
                            amount = _normalize_number(m.group("amount"))
                            prev_balance = results[-1]["balance"] if results else 0.0

                            results.append({
                                "date": _normalize_date(last_date_token, ctx),
                                "description": clean_description(m.group("desc").strip()),
                                "amount": amount,
                                "balance": prev_balance + amount,
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
                    # Wrapped description of the previous row — but never glue
                    # page furniture onto it.
                    if results:
                        extra = clean_description(line)
                        if extra:
                            results[-1]["description"] = clean_description(
                                f"{results[-1]['description']} {extra}"
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


def extract_transactions_from_pdf(pdf_path: str) -> list[dict]:
    import pdfplumber  # lazy: everything else works without PDF support installed

    path_obj = Path(pdf_path)
    if not path_obj.exists():
        raise FileNotFoundError(pdf_path)

    with pdfplumber.open(str(path_obj)) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    return extract_transactions_from_text(pages)