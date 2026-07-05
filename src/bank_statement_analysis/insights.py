"""Pure aggregation for the Insights charts.

Takes categorised transaction rows (dicts from the categorised CSVs) and
returns small JSON-ready summaries; the frontend renders them with Chart.js.
Kept pure (no I/O, no web types) so it is unit-testable. Ported in spirit from
the prototype's charts.py.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

from .categories import COST_CODES, INCOME_CODE, label


def _category(row: dict[str, Any]) -> int | None:
    val = row.get("category")
    if val in (None, ""):
        return None
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _signed_amount(row: dict[str, Any]) -> float:
    try:
        return float(row.get("signed_amount") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _month(row: dict[str, Any]) -> str | None:
    """dd-mm-yyyy -> 'yyyy-mm' (extraction's date format); None if unparseable."""
    raw = (row.get("date") or "").strip()
    try:
        return datetime.strptime(raw, "%d-%m-%Y").strftime("%Y-%m")
    except ValueError:
        return None


def aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summary totals + costs by category + income/costs per month."""
    rows = list(rows)

    total_income = 0.0
    total_costs = 0.0
    total_controllable = 0.0
    total_fixed = 0.0
    n_categorised = 0
    n_uncategorised = 0
    by_category: dict[int, dict[str, Any]] = {}
    by_month: dict[str, dict[str, float]] = {}

    for r in rows:
        cat = _category(r)
        amount = abs(_signed_amount(r))
        is_controllable = str(r.get("controllable") or "").strip().lower() == "yes"

        if cat == INCOME_CODE:
            total_income += amount
            n_categorised += 1
        elif cat in COST_CODES:
            total_costs += amount
            if is_controllable:
                total_controllable += amount
            else:
                total_fixed += amount
            n_categorised += 1
            agg = by_category.setdefault(cat, {"category": cat, "label": label(cat),
                                               "amount": 0.0, "count": 0,
                                               "controllable_amount": 0.0})
            agg["amount"] += amount
            agg["count"] += 1
            if is_controllable:
                agg["controllable_amount"] += amount
        else:
            n_uncategorised += 1

        month = _month(r)
        if month and cat is not None:
            m = by_month.setdefault(month, {"income": 0.0, "costs": 0.0,
                                            "controllable": 0.0, "fixed": 0.0})
            if cat == INCOME_CODE:
                m["income"] += amount
            elif cat in COST_CODES:
                m["costs"] += amount
                m["controllable" if is_controllable else "fixed"] += amount

    costs_by_category = sorted(by_category.values(), key=lambda a: a["amount"], reverse=True)
    for a in costs_by_category:
        a["amount"] = round(a["amount"], 2)
        a["controllable_amount"] = round(a["controllable_amount"], 2)
    months = [
        {"month": m, "income": round(v["income"], 2), "costs": round(v["costs"], 2),
         "controllable": round(v["controllable"], 2), "fixed": round(v["fixed"], 2)}
        for m, v in sorted(by_month.items())
    ]

    return {
        "summary": {
            "total_income": round(total_income, 2),
            "total_costs": round(total_costs, 2),
            "total_controllable": round(total_controllable, 2),
            "total_fixed": round(total_fixed, 2),
            "net": round(total_income - total_costs, 2),
            "n_transactions": len(rows),
            "n_categorised": n_categorised,
            "n_uncategorised": n_uncategorised,
        },
        "costs_by_category": costs_by_category,
        "by_month": months,
    }
