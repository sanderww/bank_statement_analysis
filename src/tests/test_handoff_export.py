"""Claude Code hand-off round trip, final export, prompt editing. Synthetic data."""
from __future__ import annotations

import csv

import pytest
from fastapi.testclient import TestClient

from bank_statement_analysis import config, handoff, prompt_store
from bank_statement_analysis.export import FINAL_COLUMNS, export_final
from bank_statement_analysis.io_utils import write_csv
from bank_statement_analysis.server import app


@pytest.fixture()
def client(data_root):
    with TestClient(app) as c:
        yield c


def _seed_prompts():
    (config.prompts_dir() / "v1.txt").write_text("prompt one", encoding="utf-8")
    (config.prompts_dir() / "v2.txt").write_text("prompt two", encoding="utf-8")


def _write_extracted(name="2026-07-04_synthetic_extracted_raw.csv"):
    config.ensure_dirs()
    rows = [
        {"date": "05-01-2026", "description": "POS Purchase Grocer", "amount": 200.0,
         "balance": 700.0, "direction": "out", "signed_amount": -200.0},
        {"date": "25-01-2026", "description": "Employer Payroll Credit", "amount": 40000.0,
         "balance": 40700.0, "direction": "in", "signed_amount": 40000.0},
    ]
    path = config.extracted_raw_dir() / name
    write_csv(rows, str(path), include_category=False)
    return path


# --- hand-off ----------------------------------------------------------------

def test_handoff_round_trip(client):
    _seed_prompts()
    path = _write_extracted()

    res = client.post("/api/handoff/export", json={"file": path.name})
    assert res.status_code == 200
    data = res.json()
    assert data["rows"] == 2
    assert data["prompt_version"] == "v2"

    # request + instructions + meta exist; instructions embed the prompt
    listing = client.get("/api/handoff").json()
    assert len(listing["requests"]) == 1
    assert listing["requests"][0]["source"] == path.name
    instructions = (config.handoff_dir() / data["instructions_path"].split("/")[-1]).read_text()
    assert "prompt two" in instructions
    assert "id,category" in instructions.replace("`", "")

    # request ids are row indices with signed amounts
    with open(data["request_path"], newline="", encoding="utf-8") as f:
        req_rows = list(csv.DictReader(f))
    assert [r["id"] for r in req_rows] == ["0", "1"]
    assert req_rows[1]["amount"] == "40000.00"

    # simulate a Claude session writing the results file (one invalid row)
    results_name = data["expected_results_name"]
    with open(config.handoff_dir() / results_name, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category"])
        w.writerow([0, 2])
        w.writerow([1, 10])
        w.writerow([99, 5])  # unknown id -> skipped

    assert results_name in client.get("/api/handoff").json()["results"]

    imp = client.post("/api/handoff/import", json={"results": results_name})
    assert imp.status_code == 200
    idata = imp.json()
    assert idata["applied"] == 2
    assert idata["skipped"] == 1

    # output is a normal categorised CSV, reviewable
    review = client.get(f"/api/review/{idata['output_name']}").json()
    assert review["rows"][0]["category"] == 2
    assert review["rows"][0]["source"] == "claude"
    assert review["rows"][1]["category"] == 10


def test_handoff_import_validates_name_and_existence(client):
    assert client.post("/api/handoff/import", json={"results": "nope.csv"}).status_code == 400
    assert client.post("/api/handoff/import", json={"results": "results_123.csv"}).status_code == 404


def test_handoff_export_missing_file_404(client):
    assert client.post("/api/handoff/export", json={"file": "missing.csv"}).status_code == 404


# --- final export ---------------------------------------------------------------

def test_export_final_decoupled(client):
    config.ensure_dirs()
    rows = [
        {"date": "05-01-2026", "description": "POS Purchase Grocer", "amount": 200.0,
         "balance": 700.0, "direction": "out", "signed_amount": -200.0,
         "source_statement": "jan.pdf", "category": 2,
         "category_label": "Groceries & Household", "source": "user", "confidence": ""},
        {"date": "06-01-2026", "description": "Unfinished row", "amount": 10.0,
         "balance": 690.0, "direction": "out", "signed_amount": -10.0,
         "source_statement": "jan.pdf", "category": "", "category_label": "",
         "source": "", "confidence": ""},
    ]
    name = "2026-01-31_categorised_openai.csv"
    write_csv(rows, str(config.categorised_dir() / name), include_category=True)

    res = client.post(f"/api/review/{name}/export")
    assert res.status_code == 200
    data = res.json()
    assert data["rows"] == 1
    assert data["skipped_uncategorised"] == 1

    with open(data["path"], newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        body = list(reader)
    assert header == FINAL_COLUMNS
    assert "balance" not in header and "confidence" not in header  # decoupled
    assert body[0][2] == "-200.00"          # signed amount
    assert body[0][-1] == "jan.pdf"         # traceable to source statement


def test_export_final_missing_file():
    with pytest.raises(FileNotFoundError):
        export_final("missing.csv")


# --- prompt editing ---------------------------------------------------------------

def test_prompt_edit_in_place(client):
    _seed_prompts()
    res = client.put("/api/prompts/v1", json={"text": "edited prompt one"})
    assert res.status_code == 200
    assert prompt_store.read("v1") == "edited prompt one"
    # editing doesn't change the active version
    assert prompt_store.active_version() == "v2"


def test_prompt_edit_validates(client):
    _seed_prompts()
    assert client.put("/api/prompts/v99", json={"text": "x"}).status_code == 404
    assert client.put("/api/prompts/v1", json={"text": "  "}).status_code == 400
