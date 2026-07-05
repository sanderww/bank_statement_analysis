"""settings.json, versioned prompts, versioned models, training data. All
against the tmp data root; all training rows are synthetic."""
from __future__ import annotations

import pytest

from bank_statement_analysis import (
    config,
    model_store,
    prompt_store,
    settings,
    training_data,
)


# --- settings ---------------------------------------------------------------

def test_settings_defaults_and_roundtrip(data_root):
    s = settings.load()
    assert s["active_prompt_version"] == "v2"
    assert s["active_model_version"] is None
    assert s["confidence_threshold"] == config.DEFAULT_CONFIDENCE_THRESHOLD

    settings.save({"confidence_threshold": 0.5})
    assert settings.get("confidence_threshold") == 0.5
    # untouched keys survive a partial save
    assert settings.get("active_prompt_version") == "v2"
    assert config.settings_path().exists()


def test_settings_corrupt_file_falls_back_to_defaults(data_root):
    config.settings_path().write_text("{not json", encoding="utf-8")
    assert settings.get("active_prompt_version") == "v2"


# --- prompts ----------------------------------------------------------------

def _seed_prompts():
    pdir = config.prompts_dir()
    (pdir / "v1.txt").write_text("prompt one", encoding="utf-8")
    (pdir / "v2.txt").write_text("prompt two", encoding="utf-8")


def test_prompt_versions_listed_newest_first(data_root):
    _seed_prompts()
    versions = prompt_store.list_versions()
    assert [v["version"] for v in versions] == ["v2", "v1"]
    assert versions[0]["active"] is True  # v2 is default active


def test_prompt_add_version_activates(data_root):
    _seed_prompts()
    new = prompt_store.add_version("prompt three")
    assert new == "v3"
    assert prompt_store.active_version() == "v3"
    assert prompt_store.active_text() == "prompt three"


def test_prompt_set_active_validates(data_root):
    _seed_prompts()
    prompt_store.set_active("v1")
    assert prompt_store.active_text() == "prompt one"
    with pytest.raises(FileNotFoundError):
        prompt_store.set_active("v99")


def test_prompt_add_rejects_empty(data_root):
    with pytest.raises(ValueError):
        prompt_store.add_version("   ")


# --- models -----------------------------------------------------------------

def _labelled_rows():
    groceries = [{"description": f"POS Purchase Grocer Aisle {i}",
                  "signed_amount": -float(100 + i), "category": 2} for i in range(6)]
    income = [{"description": f"Employer Payroll Credit {i}",
               "signed_amount": float(40000 + i), "category": 10} for i in range(6)]
    return groceries + income


def test_model_train_versioned_and_predict_with_confidence(data_root):
    summary = model_store.train(_labelled_rows(), sources=["synthetic"])
    assert summary["version"] == 1
    assert summary["training_size"] == 12
    assert model_store.active_version() == 1

    meta = model_store.list_versions()
    assert meta[0]["version"] == 1
    assert meta[0]["active"] is True
    assert meta[0]["feature_set"] == model_store.FEATURE_SET
    assert meta[0]["sources"] == ["synthetic"]

    rows = [
        {"description": "POS Purchase Grocer Aisle X", "signed_amount": -120.0},
        {"description": "Employer Payroll Credit Nov", "signed_amount": 41000.0},
    ]
    model_store.predict_rows(rows)
    assert rows[0]["category"] == 2
    assert rows[0]["category_label"] == "Groceries & Household"
    assert rows[0]["controllable"] == "yes"  # groceries default
    assert rows[1]["category"] == 10
    assert rows[1]["controllable"] == "no"   # income default
    assert rows[0]["source"] == "local"
    assert 0.0 <= rows[0]["confidence"] <= 1.0


def test_model_second_train_bumps_version_and_activates(data_root):
    model_store.train(_labelled_rows())
    summary = model_store.train(_labelled_rows())
    assert summary["version"] == 2
    assert model_store.active_version() == 2
    # switching back works
    model_store.set_active(1)
    assert model_store.active_version() == 1


def test_model_train_requires_two_classes(data_root):
    rows = [{"description": "x", "signed_amount": -1.0, "category": 2} for _ in range(3)]
    with pytest.raises(ValueError):
        model_store.train(rows)


def test_predict_without_model_raises(data_root):
    with pytest.raises(model_store.ModelUnavailable):
        model_store.predict_rows([{"description": "x", "signed_amount": -1.0}])


def test_missing_artefact_is_recoverable_error(data_root):
    model_store.train(_labelled_rows())
    (config.models_dir() / "v1" / "model.joblib").unlink()
    with pytest.raises(model_store.ModelUnavailable):
        model_store.load_active()


# --- training data ----------------------------------------------------------

def _reviewed_rows():
    return [
        {"date": "01-01-2026", "description": "POS Purchase Grocer", "amount": 200.0,
         "balance": 700.0, "signed_amount": -200.0, "category": 2},
        {"date": "02-01-2026", "description": "Employer Payroll Credit", "amount": 40000.0,
         "balance": 40700.0, "signed_amount": 40000.0, "category": 10},
        {"date": "03-01-2026", "description": "Uncategorised row", "amount": 10.0,
         "balance": 40690.0, "signed_amount": -10.0, "category": None},
    ]


def test_promote_rows_dedups_and_skips_invalid(data_root):
    res = training_data.promote_rows(_reviewed_rows())
    assert res == {"added": 2, "skipped_duplicate": 0, "skipped_invalid": 1}

    # promoting again: everything already there
    res2 = training_data.promote_rows(_reviewed_rows())
    assert res2["added"] == 0
    assert res2["skipped_duplicate"] == 2

    rows = training_data.load_training_rows()
    assert len(rows) == 2
    assert {r["category"] for r in rows} == {2, 10}
    assert rows[0]["signed_amount"] == -200.0

    st = training_data.stats()
    assert st["total_rows"] == 2
    assert st["files"][0]["name"] == training_data.TRAINING_FILE


def test_legacy_training_csv_uses_signed_amount_fallback(data_root, tmp_path):
    legacy = tmp_path / "legacy_training.csv"
    legacy.write_text(
        "description,amount,category\nOld Style Row,-150.00,2\nOld Income,40000.00,10\n",
        encoding="utf-8",
    )
    rows = training_data.read_training_csv(legacy)
    assert rows[0]["signed_amount"] == -150.0
    assert rows[1]["signed_amount"] == 40000.0
