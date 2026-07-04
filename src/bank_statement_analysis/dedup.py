"""Deduplication key for transactions.

Overlapping statements can contain the same transaction. The dedup key is
(date, description, amount, balance) — the running `balance` is included so two
genuinely distinct transactions that share date/description/amount are not
wrongly merged. Ported from the prototype (bank_categoriser/dedup.py).
"""
from __future__ import annotations


def dedup_key(date: str, description: str, amount: float, balance: float) -> str:
    desc = " ".join(description.split()).lower()
    return f"{date}|{desc}|{amount:.2f}|{balance:.2f}"
