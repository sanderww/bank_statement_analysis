"""PDF extraction logic against synthetic statement text (no PDFs needed).

Covers the two bugs observed in real extracts during the 2026-07-07 quality
pass: (1) multi-page statements getting split across YEARS because pages
without a 'Statement Period' header fell back to today's year, and
(2) page furniture / column tokens / leftover amounts polluting descriptions.
"""
from __future__ import annotations

from datetime import date

from bank_statement_analysis.extract import (
    YearContext,
    build_year_context,
    clean_description,
    extract_transactions_from_text,
)


def _page(body: str, period: str | None = None) -> str:
    header = f"Statement Period : {period}\n" if period else ""
    return (
        f"{header}"
        "Transactions in RAND (ZAR)\n"
        "Date Description Amount Balance\n"
        f"{body}"
        "Closing Balance 1,000.00Cr\n"
    )


# --- year inference -----------------------------------------------------------

def test_multi_page_statement_stays_in_one_year():
    # Page 2 has no Statement Period header — it must NOT fall back to today.
    p1 = _page("23 Jun POS Purchase Example Grocer 100.00 900.00Cr\n",
               period="22 June 2025 to 22 July 2025")
    p2 = _page("28 Jun POS Purchase Example Cafe 50.00 850.00Cr\n")
    rows = extract_transactions_from_text([p1, p2])
    assert [r["date"] for r in rows] == ["23-06-2025", "28-06-2025"]


def test_year_boundary_statement_resolves_each_side():
    p = _page(
        "23 Dec POS Purchase Example Grocer 100.00 900.00Cr\n"
        "05 Jan POS Purchase Example Cafe 50.00 850.00Cr\n",
        period="22 December 2025 to 22 January 2026",
    )
    rows = extract_transactions_from_text([p])
    assert rows[0]["date"] == "23-12-2025"
    assert rows[1]["date"] == "05-01-2026"


def test_year_context_falls_back_to_header_year():
    ctx = build_year_context("Statement Date : 22 May 2025\nother text")
    assert ctx.resolve(23, 6) == 2025


def test_year_context_period_parsing():
    ctx = build_year_context("Statement Period : 22 December 2025 to 22 January 2026")
    assert ctx.start == date(2025, 12, 22)
    assert ctx.end == date(2026, 1, 22)
    assert ctx.resolve(28, 12) == 2025
    assert ctx.resolve(3, 1) == 2026


def test_year_context_slack_for_transactions_just_outside_period():
    ctx = YearContext(start=date(2025, 6, 22), end=date(2025, 7, 22))
    assert ctx.resolve(20, 6) == 2025  # 2 days before the period start


# --- description cleaning ------------------------------------------------------

def test_clean_description_strips_page_furniture_and_account_numbers():
    noisy = ("POS Purchase Example Shop 000000*0000 14 Jul Page 2of 3 Delivery Method "
             "F1 R04 Branch Number Account Number Date 0000 0000 60000000000 Charges")
    assert clean_description(noisy) == "POS Purchase Example Shop 000000*0000 14 Jul"


def test_clean_description_strips_leading_amount_leftovers():
    assert clean_description("300.00 38,814.98Cr POS Purchase Example Restaurant") == \
        "POS Purchase Example Restaurant"


def test_clean_description_strips_trailing_column_tokens():
    assert clean_description("POS Purchase Example Grocer 15 Jul Charges") == \
        "POS Purchase Example Grocer 15 Jul"
    assert clean_description("Some Payment Accrued Bank Charges") == "Some Payment"


def test_clean_description_keeps_normal_text():
    assert clean_description("FNB App Payment To Example Person") == \
        "FNB App Payment To Example Person"


def test_page_furniture_continuation_line_is_not_appended():
    p = _page(
        "23 Jun POS Purchase Example Grocer 100.00 900.00Cr\n"
        "Page 2 of 3 Delivery Method F1 R04\n",  # furniture inside the section
        period="22 June 2025 to 22 July 2025",
    )
    rows = extract_transactions_from_text([p])
    assert rows[0]["description"] == "POS Purchase Example Grocer"


def test_wrapped_description_is_still_appended():
    p = _page(
        "23 Jun FNB App Payment To Example 100.00 900.00Cr\n"
        "Person Continued\n",
        period="22 June 2025 to 22 July 2025",
    )
    rows = extract_transactions_from_text([p])
    assert rows[0]["description"] == "FNB App Payment To Example Person Continued"


# --- core parsing ----------------------------------------------------------------

def test_amount_and_balance_parsing():
    p = _page(
        "23 Jun Example Debit 1,234.56 10,000.00Cr\n"
        "24 Jun Example Overdrawn 50.00 100.00Dr\n",
        period="22 June 2025 to 22 July 2025",
    )
    rows = extract_transactions_from_text([p])
    assert rows[0]["amount"] == 1234.56
    assert rows[0]["balance"] == 10000.00
    assert rows[1]["balance"] == -100.00  # Dr = negative balance


def test_second_transaction_hidden_on_same_line():
    p = _page(
        "23 Jun First Txn 100.00 900.00Cr Second Txn On Same Line 200.00 700.00Cr\n",
        period="22 June 2025 to 22 July 2025",
    )
    rows = extract_transactions_from_text([p])
    assert len(rows) == 2
    assert rows[1]["description"] == "Second Txn On Same Line"
    assert rows[1]["date"] == "23-06-2025"
    assert rows[1]["balance"] == 700.00
