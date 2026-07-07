"""LLM categorisation with a stubbed OpenAI client (no API calls)."""
from __future__ import annotations

import json
from types import SimpleNamespace

from bank_statement_analysis import config
from bank_statement_analysis.categorize import CategoryResult, CategoryResults, categorize_rows


class StubClient:
    """Captures requests; returns canned CategoryResults per call."""

    def __init__(self, responder):
        self.calls = []
        outer = self

        class _Responses:
            def parse(self, **kwargs):
                outer.calls.append(kwargs)
                return SimpleNamespace(output_parsed=responder(kwargs))

        self.responses = _Responses()


def _seed_prompt(data_root):
    (config.prompts_dir() / "v2.txt").write_text("categorise things", encoding="utf-8")


def _rows():
    return [
        {"date": "05-01-2026", "description": "POS Purchase Grocer", "amount": 200.0,
         "balance": 700.0, "direction": "out", "signed_amount": -200.0},
        {"date": "25-01-2026", "description": "Employer Payroll Credit", "amount": 40000.0,
         "balance": 40700.0, "direction": "in", "signed_amount": 40000.0},
    ]


def test_batched_payload_carries_direction_and_signed_amount(data_root):
    _seed_prompt(data_root)
    rows = _rows()
    client = StubClient(lambda kwargs: CategoryResults(results=[
        CategoryResult(index=0, category=2, controllable=True),
        CategoryResult(index=1, category=10, controllable=True),  # LLM mistake: income 'controllable'
    ]))

    categorize_rows(rows, prompt_version="v2", client=client)

    # one batched call, not one per row
    assert len(client.calls) == 1
    payload = json.loads(client.calls[0]["input"][1]["content"].split("\n", 1)[1])
    assert payload[0]["direction"] == "out"
    assert payload[0]["amount"] == -200.0     # signed, as prompt v2 describes
    assert payload[1]["amount"] == 40000.0
    assert client.calls[0]["input"][0]["content"] == "categorise things"

    assert rows[0]["category"] == 2
    assert rows[0]["category_label"] == "Groceries & Household"
    assert rows[0]["controllable"] == "yes"
    assert rows[0]["source"] == "openai"
    # income can never be controllable, whatever the LLM said
    assert rows[1]["category"] == 10
    assert rows[1]["controllable"] == "no"


def test_chunking_by_batch_size(data_root):
    _seed_prompt(data_root)
    rows = _rows() + _rows() + _rows()  # 6 rows
    client = StubClient(lambda kwargs: CategoryResults(results=[
        CategoryResult(index=json.loads(kwargs["input"][1]["content"].split("\n", 1)[1])[i]["index"],
                       category=2, controllable=True)
        for i in range(len(json.loads(kwargs["input"][1]["content"].split("\n", 1)[1])))
    ]))

    categorize_rows(rows, prompt_version="v2", client=client, batch_size=4)
    assert len(client.calls) == 2  # 4 + 2
    assert all(r["category"] == 2 for r in rows)


def test_failed_chunk_falls_back_to_unknown(data_root):
    _seed_prompt(data_root)
    rows = _rows()

    def boom(kwargs):
        raise RuntimeError("api down")

    categorize_rows(rows, prompt_version="v2", client=StubClient(boom))
    assert all(r["category"] == 0 for r in rows)
    assert all(r["category_label"] == "Unknown" for r in rows)
    assert all(r["source"] == "openai" for r in rows)


def test_missing_result_for_a_row_is_unknown(data_root):
    _seed_prompt(data_root)
    rows = _rows()
    client = StubClient(lambda kwargs: CategoryResults(results=[
        CategoryResult(index=0, category=6, controllable=True),
        # no result for index 1
    ]))

    categorize_rows(rows, prompt_version="v2", client=client)
    assert rows[0]["category"] == 6
    assert rows[1]["category"] == 0
