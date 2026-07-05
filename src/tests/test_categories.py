"""The 0-10 category set is consistent everywhere it is consumed."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from bank_statement_analysis.categories import (
    CATEGORY_LABELS,
    COST_CODES,
    DEFAULT_CONTROLLABLE,
    INCOME_CODE,
    VALID_CODES,
    Category,
    default_controllable,
    label,
)
from bank_statement_analysis.categorize import CategorizedTransaction


def test_category_codes_cover_0_to_10():
    assert VALID_CODES == frozenset(range(11))
    assert Category.UNKNOWN == 0
    assert Category.INCOME == 10
    assert set(CATEGORY_LABELS) == set(Category)


def test_cost_and_income_partition():
    assert COST_CODES == tuple(range(1, 10))
    assert INCOME_CODE == 10
    assert INCOME_CODE not in COST_CODES
    assert 0 not in COST_CODES


def test_controllable_defaults_cover_all_categories():
    assert set(DEFAULT_CONTROLLABLE) == set(Category)
    # consumption categories default to controllable
    assert default_controllable(int(Category.DINING)) is True
    assert default_controllable(int(Category.GROCERIES_HOUSEHOLD)) is True
    assert default_controllable(int(Category.TRANSPORT)) is True
    # committed categories and income don't
    assert default_controllable(int(Category.HEALTH_INSURANCE)) is False
    assert default_controllable(int(Category.INCOME)) is False
    # bad input is safe
    assert default_controllable(None) is False
    assert default_controllable(99) is False


def test_label_helper_handles_bad_input():
    assert label(2) == "Groceries & Household"
    assert label(10) == "Income"
    assert label(None) == "Unknown"
    assert label(99) == "Unknown"


def test_categorized_transaction_accepts_full_range():
    base = dict(date="01-01-2026", description="x", amount=1.0, balance=1.0)
    for code in (0, 10):
        tx = CategorizedTransaction(**base, category=code, category_label=label(code),
                                    controllable=False)
        assert int(tx.category) == code


def test_categorized_transaction_rejects_out_of_range():
    base = dict(date="01-01-2026", description="x", amount=1.0, balance=1.0)
    with pytest.raises(ValidationError):
        CategorizedTransaction(**base, category=11, category_label="nope", controllable=False)


def test_categorized_transaction_requires_controllable():
    base = dict(date="01-01-2026", description="x", amount=1.0, balance=1.0)
    with pytest.raises(ValidationError):
        CategorizedTransaction(**base, category=2, category_label=label(2))
