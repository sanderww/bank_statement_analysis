"""Direction inference: balance-delta primary signal, keyword fallback.

All data here is synthetic — never put real statement rows in this repo.
"""
from __future__ import annotations

from bank_statement_analysis.direction import derive


def test_balance_delta_gives_direction():
    rows = [
        {"date": "01-01-2026", "description": "Opening purchase", "amount": 100.0, "balance": 900.0},
        {"date": "02-01-2026", "description": "POS Purchase Grocer", "amount": 200.0, "balance": 700.0},
        {"date": "03-01-2026", "description": "Transfer received", "amount": 500.0, "balance": 1200.0},
    ]
    out = derive(rows)
    # Row 2: balance fell by exactly the amount -> out
    assert out[1]["direction"] == "out"
    assert out[1]["signed_amount"] == -200.0
    # Row 3: balance rose by exactly the amount -> in
    assert out[2]["direction"] == "in"
    assert out[2]["signed_amount"] == 500.0


def test_first_row_falls_back_to_keywords():
    rows = [
        {"date": "01-01-2026", "description": "Acme Corp Salary Payment", "amount": 40000.0, "balance": 41000.0},
        {"date": "01-01-2026", "description": "Coffee Shop", "amount": 50.0, "balance": 40950.0},
    ]
    out = derive(rows)
    assert out[0]["direction"] == "in"  # 'salar' keyword
    assert out[1]["direction"] == "out"  # balance delta


def test_inconsistent_balance_falls_back_to_keywords_then_out():
    rows = [
        {"date": "01-01-2026", "description": "Mystery Shop", "amount": 100.0, "balance": 500.0},
        # Delta (+300) doesn't match amount (100) -> keyword fallback -> default out
        {"date": "02-01-2026", "description": "Another Shop", "amount": 100.0, "balance": 800.0},
        # Delta mismatch + credit keyword -> in
        {"date": "03-01-2026", "description": "Refund from store", "amount": 100.0, "balance": 1000.0},
    ]
    out = derive(rows)
    assert out[1]["direction"] == "out"
    assert out[2]["direction"] == "in"
    assert out[2]["signed_amount"] == 100.0


def test_delta_within_tolerance_counts():
    rows = [
        {"date": "01-01-2026", "description": "start", "amount": 10.0, "balance": 100.0},
        # delta = -10.03, amount 10.0, tolerance 0.05 -> still 'out'
        {"date": "02-01-2026", "description": "shop", "amount": 10.0, "balance": 89.97},
    ]
    out = derive(rows)
    assert out[1]["direction"] == "out"
