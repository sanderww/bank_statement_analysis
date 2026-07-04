"""The fixed category label set (0-10) — the single source of truth.

0 = Unknown, 1-9 = expense categories, 10 = Income. Matches prompts/v2.txt.
Changing this set is a deliberate decision: it invalidates trained models and
existing training labels.
"""
from __future__ import annotations

from enum import IntEnum


class Category(IntEnum):
    UNKNOWN = 0
    HOUSING_UTILITIES = 1
    GROCERIES_HOUSEHOLD = 2
    CHILDCARE_EDUCATION = 3
    TRANSPORT = 4
    HEALTH_INSURANCE = 5
    DINING = 6
    CLOTHING_PERSONAL_CARE = 7
    ENTERTAINMENT = 8
    FINANCIAL_MISC = 9
    INCOME = 10


CATEGORY_LABELS: dict[Category, str] = {
    Category.UNKNOWN: "Unknown",
    Category.HOUSING_UTILITIES: "Housing & Utilities",
    Category.GROCERIES_HOUSEHOLD: "Groceries & Household",
    Category.CHILDCARE_EDUCATION: "Childcare & Education",
    Category.TRANSPORT: "Transport",
    Category.HEALTH_INSURANCE: "Health & Insurance",
    Category.DINING: "Dining",
    Category.CLOTHING_PERSONAL_CARE: "Clothing & Personal Care",
    Category.ENTERTAINMENT: "Entertainment",
    Category.FINANCIAL_MISC: "Financial & Miscellaneous",
    Category.INCOME: "Income",
}

VALID_CODES: frozenset[int] = frozenset(int(c) for c in Category)

# Expense categories (used by insights aggregation).
COST_CODES: tuple[int, ...] = tuple(range(1, 10))
INCOME_CODE: int = int(Category.INCOME)


def label(code: int | None) -> str:
    """Human-readable label for a category code; 'Unknown' for None/invalid."""
    if code is None:
        return CATEGORY_LABELS[Category.UNKNOWN]
    try:
        return CATEGORY_LABELS[Category(int(code))]
    except (ValueError, KeyError):
        return CATEGORY_LABELS[Category.UNKNOWN]
