"""API endpoints against a tmp data root. Synthetic data only."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from bank_statement_analysis import config
from bank_statement_analysis.io_utils import write_csv
from bank_statement_analysis.server import app


@pytest.fixture()
def client(data_root):
    with TestClient(app) as c:
        yield c


def _categorised_rows():
    return [
        {"date": "05-01-2026", "description": "POS Purchase Grocer", "amount": 200.0,
         "balance": 700.0, "direction": "out", "signed_amount": -200.0,
         "category": 2, "category_label": "Groceries & Household", "source": "openai", "confidence": ""},
        {"date": "25-01-2026", "description": "Employer Payroll Credit", "amount": 40000.0,
         "balance": 40700.0, "direction": "in", "signed_amount": 40000.0,
         "category": 10, "category_label": "Income", "source": "openai", "confidence": ""},
        {"date": "26-01-2026", "description": "Petrol Station", "amount": 800.0,
         "balance": 39900.0, "direction": "out", "signed_amount": -800.0,
         "category": 4, "category_label": "Transport", "source": "local", "confidence": 0.55},
        {"date": "03-02-2026", "description": "Mystery Row", "amount": 10.0,
         "balance": 39890.0, "direction": "out", "signed_amount": -10.0,
         "category": "", "category_label": "", "source": "", "confidence": ""},
    ]


def _write_categorised(filename="2026-01-31_categorised_openai.csv"):
    config.ensure_dirs()
    path = config.categorised_dir() / filename
    write_csv(_categorised_rows(), str(path), include_category=True)
    return path


def _seed_prompts():
    (config.prompts_dir() / "v1.txt").write_text("prompt one", encoding="utf-8")
    (config.prompts_dir() / "v2.txt").write_text("prompt two", encoding="utf-8")


# --- categories / settings ---------------------------------------------------

def test_categories_endpoint(client):
    cats = client.get("/api/categories").json()["categories"]
    assert cats[0] == {"code": 0, "label": "Unknown"}
    assert cats[-1] == {"code": 10, "label": "Income"}
    assert len(cats) == 11


def test_settings_roundtrip(client):
    assert client.get("/api/settings").json()["confidence_threshold"] == 0.70
    res = client.put("/api/settings", json={"confidence_threshold": 0.9})
    assert res.status_code == 200
    assert client.get("/api/settings").json()["confidence_threshold"] == 0.9


def test_settings_threshold_validated(client):
    assert client.put("/api/settings", json={"confidence_threshold": 1.5}).status_code == 400


# --- prompts ------------------------------------------------------------------

def test_prompt_endpoints(client):
    _seed_prompts()
    data = client.get("/api/prompts").json()
    assert data["active"] == "v2"
    assert [p["version"] for p in data["prompts"]] == ["v2", "v1"]

    res = client.post("/api/prompts", json={"text": "prompt three"})
    assert res.status_code == 200
    assert res.json()["version"] == "v3"
    assert client.get("/api/prompts").json()["active"] == "v3"

    assert client.post("/api/prompts/activate", json={"version": "v1"}).status_code == 200
    assert client.get("/api/prompts").json()["active"] == "v1"
    assert client.post("/api/prompts/activate", json={"version": "v99"}).status_code == 404


# --- review -------------------------------------------------------------------

def test_review_load_save_roundtrip(client):
    path = _write_categorised()
    name = path.name

    assert name in client.get("/api/categorised-files").json()["files"]

    data = client.get(f"/api/review/{name}").json()
    assert data["confidence_threshold"] == 0.70
    rows = data["rows"]
    assert len(rows) == 4
    assert rows[0]["category"] == 2
    assert rows[2]["confidence"] == 0.55
    assert rows[3]["category"] is None

    # user fixes the mystery row and re-categorises row 0
    rows[3]["category"] = 6
    rows[3]["source"] = "user"
    rows[0]["category"] = 1
    rows[0]["source"] = "user"
    res = client.put(f"/api/review/{name}", json={"rows": rows})
    assert res.status_code == 200

    reloaded = client.get(f"/api/review/{name}").json()["rows"]
    assert reloaded[0]["category"] == 1
    assert reloaded[0]["category_label"] == "Housing & Utilities"  # recomputed
    assert reloaded[0]["source"] == "user"
    assert reloaded[3]["category"] == 6


def test_review_rejects_bad_category_and_traversal(client):
    path = _write_categorised()
    rows = client.get(f"/api/review/{path.name}").json()["rows"]
    rows[0]["category"] = 42
    assert client.put(f"/api/review/{path.name}", json={"rows": rows}).status_code == 400
    assert client.get("/api/review/../../etc/passwd").status_code in (400, 404)
    assert client.get("/api/review/nope.csv").status_code == 404


def test_promote_to_training_and_train(client):
    path = _write_categorised()
    res = client.post(f"/api/review/{path.name}/promote").json()
    assert res["added"] == 3  # uncategorised row skipped
    assert res["skipped_invalid"] == 1

    stats = client.get("/api/training-data").json()
    assert stats["total_rows"] == 3

    # 3 rows across 3 categories is enough for a (weak) model
    train = client.post("/api/models/train")
    assert train.status_code == 200
    assert train.json()["version"] == 1

    models = client.get("/api/models").json()
    assert models["active"] == 1
    assert models["models"][0]["training_size"] == 3
    assert models["models"][0]["artefact_exists"] is True


def test_train_without_data_is_400(client):
    assert client.post("/api/models/train").status_code == 400


def test_activate_missing_model_is_404(client):
    assert client.post("/api/models/activate", json={"version": 9}).status_code == 404


# --- insights -------------------------------------------------------------------

def test_insights_endpoint(client):
    path = _write_categorised()
    res = client.get("/api/insights", params={"files": [path.name]})
    assert res.status_code == 200
    data = res.json()
    assert data["summary"]["total_income"] == 40000.0
    assert data["summary"]["total_costs"] == 1000.0
    assert data["summary"]["net"] == 39000.0
    assert data["summary"]["n_uncategorised"] == 1
    # costs by category sorted by amount desc: transport 800 > groceries 200
    assert [c["category"] for c in data["costs_by_category"]] == [4, 2]
    assert data["by_month"] == [
        {"month": "2026-01", "income": 40000.0, "costs": 1000.0},
    ]


def test_insights_requires_files(client):
    assert client.get("/api/insights").status_code == 400
