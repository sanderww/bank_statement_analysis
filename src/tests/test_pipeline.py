"""Pipeline enrichment: normalisation, dedup, CSV round-trip. Synthetic data only."""
from __future__ import annotations

from bank_statement_analysis.dedup import dedup_key
from bank_statement_analysis.io_utils import read_csv, write_csv
from bank_statement_analysis.services import dedup_rows, enrich_rows, load_csv_data


def _rows():
    return [
        {"date": "01-01-2026", "description": "POS Purchase Grocer", "amount": -200.0, "balance": 700.0},
        {"date": "02-01-2026", "description": "", "amount": 50.0, "balance": 650.0},  # dropped: no desc
        {"date": "03-01-2026", "description": "Zero fee", "amount": 0.0, "balance": 650.0},  # dropped: zero
        {"date": "04-01-2026", "description": "Monthly Salary Credit", "amount": 40000.0, "balance": 40650.0},
    ]


def test_enrich_normalises_and_derives():
    out = enrich_rows(_rows())
    assert len(out) == 2  # noise rows dropped
    first, second = out
    assert first["amount"] == 200.0  # abs magnitude
    assert first["direction"] == "out"
    assert first["signed_amount"] == -200.0
    assert second["direction"] == "in"
    assert second["signed_amount"] == 40000.0


def test_dedup_key_normalises_whitespace_and_case():
    a = dedup_key("01-01-2026", "POS  Purchase   Grocer", 200.0, 700.0)
    b = dedup_key("01-01-2026", "pos purchase grocer", 200.0, 700.0)
    assert a == b


def test_dedup_rows_keeps_first_seen():
    rows = enrich_rows(_rows())
    doubled = rows + [dict(r) for r in rows]
    assert len(dedup_rows(doubled)) == len(rows)


def test_same_transaction_different_balance_is_not_merged():
    rows = [
        {"date": "01-01-2026", "description": "Coffee", "amount": 50.0, "balance": 950.0, "direction": "out", "signed_amount": -50.0},
        {"date": "01-01-2026", "description": "Coffee", "amount": 50.0, "balance": 900.0, "direction": "out", "signed_amount": -50.0},
    ]
    assert len(dedup_rows(rows)) == 2


def test_csv_round_trip_with_new_columns(tmp_path):
    rows = enrich_rows(_rows())
    path = tmp_path / "extracted.csv"
    write_csv(rows, str(path), include_category=False)

    loaded = load_csv_data([path])
    assert len(loaded) == len(rows)
    assert loaded[0]["direction"] == "out"
    assert float(loaded[0]["signed_amount"]) == -200.0


def test_legacy_csv_without_direction_gets_enriched(tmp_path):
    # Simulate an old extracted CSV (only the four original columns).
    path = tmp_path / "legacy.csv"
    path.write_text(
        "date,description,amount,balance\n"
        "01-01-2026,POS Purchase Grocer,200.00,700.00\n"
        "02-01-2026,Monthly Salary Credit,40000.00,40700.00\n",
        encoding="utf-8",
    )
    loaded = load_csv_data([path])
    assert loaded[0]["direction"] == "out"
    assert loaded[1]["direction"] == "in"
    assert loaded[1]["signed_amount"] == 40000.0
