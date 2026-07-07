"""Derive transaction direction (money in vs out) and a signed amount.

The extractor emits `amount` as a POSITIVE magnitude for both debits and
credits (verified against real FNB extract output in the prototype), so the
sign cannot be trusted for direction. We derive it here instead.

Primary signal: if balance[i] - balance[i-1] ~= +/- amount, the sign of that
delta gives the direction. When the balance is missing/inconsistent (extraction
noise, or the first row of a statement), fall back to description keywords,
then default to 'out' (most transactions are expenses).

Ported from the prototype (bank_categoriser/direction.py).
"""
from __future__ import annotations

from typing import Any

from .config import BALANCE_MATCH_TOLERANCE

# Substrings that indicate money IN (credit). Everything else defaults to 'out'.
_IN_KEYWORDS = (
    "salar", "credit", "refund", "reversal", "deposit", "interest received",
    "cashback", "rtc credit", "inward",
)


def _direction_from_keywords(description: str) -> str:
    d = description.lower()
    # 'credit card' is a payment (money out) despite containing 'credit'
    if "credit card" in d:
        return "out"
    return "in" if any(k in d for k in _IN_KEYWORDS) else "out"


def derive(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return records (in order) each augmented with 'direction' and
    'signed_amount'. Input records need 'description', 'amount', 'balance'.
    Must be called per statement: the balance-delta signal assumes consecutive
    rows belong to the same running balance."""
    out: list[dict[str, Any]] = []
    prev_balance: float | None = None
    for r in records:
        amount = float(r["amount"])
        balance = float(r["balance"])
        direction: str | None = None

        if prev_balance is not None:
            delta = round(balance - prev_balance, 2)
            if abs(abs(delta) - amount) <= BALANCE_MATCH_TOLERANCE and amount > 0:
                direction = "in" if delta > 0 else "out"

        if direction is None:
            direction = _direction_from_keywords(r["description"])

        signed = amount if direction == "in" else -amount
        out.append({**r, "direction": direction, "signed_amount": round(signed, 2)})
        prev_balance = balance
    return out
