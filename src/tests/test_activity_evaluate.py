"""Activity log + model evaluation. Synthetic data only."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bank_statement_analysis import activity, config, model_store
from bank_statement_analysis.io_utils import write_csv
from bank_statement_analysis.server import app


@pytest.fixture()
def client(data_root):
    with TestClient(app) as c:
        yield c


# --- activity module ---------------------------------------------------------

def test_log_event_and_recent(data_root):
    activity.log_event("extract", "first")
    activity.log_event("training", "second")
    activity.log_event("extract", "third")

    entries = activity.recent()
    assert [e["message"] for e in entries] == ["third", "second", "first"]  # newest first
    assert entries[0]["category"] == "extract"
    assert entries[0]["level"] == "info"
    assert entries[0]["ts"]

    only_training = activity.recent(category="training")
    assert [e["message"] for e in only_training] == ["second"]

    assert activity.recent(limit=2) == entries[:2]


def test_unknown_category_becomes_other(data_root):
    activity.log_event("bogus", "hm")
    assert activity.recent()[0]["category"] == "other"


def test_activity_endpoint_and_actions_are_logged(client):
    # a settings change should land in the log via the endpoint
    client.put("/api/settings", json={"confidence_threshold": 0.6})
    data = client.get("/api/activity").json()
    assert data["categories"] == list(activity.CATEGORIES)
    assert any(e["category"] == "settings" and "0.6" in e["message"] for e in data["entries"])

    filtered = client.get("/api/activity", params={"category": "extract"}).json()
    assert all(e["category"] == "extract" for e in filtered["entries"])


# --- model evaluation ----------------------------------------------------------

def _train_synthetic():
    groceries = [{"description": f"POS Purchase Grocer Aisle {i}",
                  "signed_amount": -float(100 + i), "category": 2} for i in range(6)]
    income = [{"description": f"Employer Payroll Credit {i}",
               "signed_amount": float(40000 + i), "category": 10} for i in range(6)]
    model_store.train(groceries + income)


def _reviewed_file(name="2026-02-28_categorised_openai.csv"):
    rows = [
        {"date": "05-01-2026", "description": "POS Purchase Grocer Aisle X", "amount": 120.0,
         "balance": 700.0, "direction": "out", "signed_amount": -120.0,
         "source_statement": "s.pdf", "category": 2, "category_label": "Groceries & Household",
         "controllable": "yes", "source": "user", "confidence": ""},
        {"date": "25-01-2026", "description": "Employer Payroll Credit Feb", "amount": 41000.0,
         "balance": 41700.0, "direction": "in", "signed_amount": 41000.0,
         "source_statement": "s.pdf", "category": 10, "category_label": "Income",
         "controllable": "no", "source": "user", "confidence": ""},
        # the model will get this wrong (trained only on groceries/income)
        {"date": "26-01-2026", "description": "Cinema Tickets Online", "amount": 240.0,
         "balance": 41460.0, "direction": "out", "signed_amount": -240.0,
         "source_statement": "s.pdf", "category": 8, "category_label": "Entertainment",
         "controllable": "yes", "source": "user", "confidence": ""},
        # uncategorised -> excluded from evaluation
        {"date": "27-01-2026", "description": "Mystery", "amount": 5.0,
         "balance": 41455.0, "direction": "out", "signed_amount": -5.0,
         "source_statement": "s.pdf", "category": "", "category_label": "",
         "controllable": "", "source": "", "confidence": ""},
    ]
    config.ensure_dirs()
    path = config.categorised_dir() / name
    write_csv(rows, str(path), include_category=True)
    return path


def test_evaluate_against_reviewed_file(client):
    _train_synthetic()
    path = _reviewed_file()

    res = client.post("/api/models/evaluate", json={"file": path.name})
    assert res.status_code == 200
    data = res.json()
    assert data["model_version"] == 1
    assert data["n_rows"] == 3  # uncategorised row excluded
    assert data["correct"] == 2  # groceries + income right, entertainment unknown to model
    assert data["accuracy"] == round(2 / 3, 4)
    by_cat = {c["category"]: c for c in data["per_category"]}
    assert by_cat[2]["accuracy"] == 1.0
    assert by_cat[8]["accuracy"] == 0.0
    # and it was logged
    log = client.get("/api/activity", params={"category": "training"}).json()["entries"]
    assert any("Evaluated model v1" in e["message"] for e in log)


def test_evaluate_without_model_is_400(client):
    path = _reviewed_file()
    assert client.post("/api/models/evaluate", json={"file": path.name}).status_code == 400


def test_evaluate_without_categorised_rows_is_400(client):
    _train_synthetic()
    rows = [{"date": "01-01-2026", "description": "x", "amount": 1.0, "balance": 1.0,
             "direction": "out", "signed_amount": -1.0, "source_statement": "s.pdf",
             "category": "", "category_label": "", "controllable": "", "source": "", "confidence": ""}]
    config.ensure_dirs()
    path = config.categorised_dir() / "empty_cats.csv"
    write_csv(rows, str(path), include_category=True)
    assert client.post("/api/models/evaluate", json={"file": path.name}).status_code == 400
