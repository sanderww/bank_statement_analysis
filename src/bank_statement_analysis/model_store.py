"""Versioned local ML models: train, list, activate, predict with confidence.

Each version lives in models/v{N}/ as model.joblib + metadata.json (training
size, feature set, metrics, sources, created_at). The active version is chosen
in settings.json. `predict` returns a probability per row — used for
low-confidence flagging in the review step. Training is fully offline.

Feature set (ported from the prototype): TF-IDF over the description plus the
signed and absolute amount — direction is a strong income-vs-expense signal.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import config, settings
from .categories import default_controllable, label as category_label

FEATURE_SET = "description+signed_amount+abs_amount"
TEXT_COL = "description"
NUMERIC_COLS = ["signed_amount", "abs_amount"]


class ModelUnavailable(RuntimeError):
    """Raised when an active model is configured but its artefact is missing or
    can't be unpickled (e.g. trained by an older build). Recoverable: the user
    should train a new version."""


def rows_to_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            TEXT_COL: [str(r["description"]) for r in rows],
            "signed_amount": [float(r["signed_amount"]) for r in rows],
            "abs_amount": [abs(float(r["signed_amount"])) for r in rows],
        }
    )


def build_pipeline() -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True), TEXT_COL),
            ("num", StandardScaler(with_mean=False), NUMERIC_COLS),
        ]
    )
    return Pipeline([
        ("features", pre),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])


# --- Training ---------------------------------------------------------------

def train(rows: Sequence[dict[str, Any]], sources: list[str] | None = None) -> dict[str, Any]:
    """Train and store the next model version from labelled rows
    ({description, signed_amount, category}), then make it active.
    Raises ValueError if the data can't support training."""
    if len(rows) < 2:
        raise ValueError("Need at least 2 labelled transactions to train.")
    y = [int(r["category"]) for r in rows]
    if len(set(y)) < 2:
        raise ValueError("Training data must cover at least 2 distinct categories.")
    X = rows_to_frame(rows)

    metrics = _fit_and_score(X, y)
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    version = _next_version()
    model_dir = config.models_dir() / f"v{version}"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dir / "model.joblib")

    metadata = {
        "version": version,
        "training_size": len(rows),
        "feature_set": FEATURE_SET,
        "metrics": metrics,
        "sources": sources or [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    settings.save({"active_model_version": version})
    return {"version": version, "training_size": len(rows), "metrics": metrics}


def _fit_and_score(X, y) -> dict[str, Any]:
    """Quick validation estimate. Stratified holdout when there's enough data
    per class; otherwise training accuracy with a caveat."""
    from collections import Counter

    counts = Counter(y)
    can_split = len(y) >= 20 and min(counts.values()) >= 2
    if can_split:
        from sklearn.model_selection import train_test_split

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        p = build_pipeline()
        p.fit(X_tr, y_tr)
        return {"holdout_accuracy": round(float(p.score(X_te, y_te)), 4),
                "n_train": len(y_tr), "n_holdout": len(y_te)}
    p = build_pipeline()
    p.fit(X, y)
    return {"train_accuracy": round(float(p.score(X, y)), 4),
            "note": "Too little data for a holdout split; train accuracy only."}


# --- Registry ---------------------------------------------------------------

def _version_dirs() -> list[Path]:
    mdir = config.models_dir()
    if not mdir.exists():
        return []
    dirs = [d for d in mdir.iterdir() if d.is_dir() and d.name.startswith("v")
            and d.name[1:].isdigit()]
    return sorted(dirs, key=lambda d: int(d.name[1:]))


def _next_version() -> int:
    dirs = _version_dirs()
    return (int(dirs[-1].name[1:]) + 1) if dirs else 1


def list_versions() -> list[dict[str, Any]]:
    """All model versions (newest first) with their metadata + active flag."""
    active = active_version()
    out: list[dict[str, Any]] = []
    for d in reversed(_version_dirs()):
        version = int(d.name[1:])
        meta_path = d / "metadata.json"
        meta: dict[str, Any] = {"version": version}
        if meta_path.exists():
            try:
                meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                pass
        meta["active"] = version == active
        meta["artefact_exists"] = (d / "model.joblib").exists()
        out.append(meta)
    return out


def active_version() -> int | None:
    v = settings.get("active_model_version")
    return int(v) if v is not None else None


def set_active(version: int) -> None:
    if not (config.models_dir() / f"v{version}" / "model.joblib").exists():
        raise FileNotFoundError(f"Model version v{version} has no artefact.")
    settings.save({"active_model_version": int(version)})


def load_active() -> Pipeline | None:
    """The active model pipeline, or None if no version is configured.
    Raises ModelUnavailable when configured but unusable."""
    version = active_version()
    if version is None:
        return None
    path = config.models_dir() / f"v{version}" / "model.joblib"
    if not path.exists():
        raise ModelUnavailable(
            f"Active model v{version} is configured but its file is missing ({path}). "
            "Train a new model or activate another version."
        )
    try:
        return joblib.load(path)
    except Exception as e:  # stale/incompatible artefact
        raise ModelUnavailable(
            f"Active model v{version} can't be loaded ({type(e).__name__}: {e}). "
            "It was likely trained by an older build — train a new model."
        ) from e


# --- Evaluation ---------------------------------------------------------------

def evaluate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Test the active model against already-categorised rows (the reviewed
    truth). Rows need description, signed_amount and a category. Returns overall
    accuracy plus a per-category breakdown — the 'is the model good enough yet?'
    check before promoting data / after training. Does not mutate rows."""
    labelled = [r for r in rows if r.get("category") not in (None, "")]
    if not labelled:
        raise ValueError("No categorised rows to evaluate against.")
    pipeline = load_active()
    if pipeline is None:
        raise ModelUnavailable("No trained model yet. Train one first.")

    X = rows_to_frame(labelled)
    preds = pipeline.predict(X)

    per_cat: dict[int, dict[str, Any]] = {}
    correct = 0
    for r, pred in zip(labelled, preds):
        truth = int(float(r["category"]))
        agg = per_cat.setdefault(truth, {
            "category": truth, "label": category_label(truth), "n": 0, "correct": 0,
        })
        agg["n"] += 1
        if int(pred) == truth:
            agg["correct"] += 1
            correct += 1

    breakdown = sorted(per_cat.values(), key=lambda a: a["n"], reverse=True)
    for a in breakdown:
        a["accuracy"] = round(a["correct"] / a["n"], 4)
    return {
        "model_version": active_version(),
        "n_rows": len(labelled),
        "correct": correct,
        "accuracy": round(correct / len(labelled), 4),
        "per_category": breakdown,
    }


# --- Prediction -------------------------------------------------------------

def predict_rows(rows: Sequence[dict[str, Any]]) -> None:
    """Categorise `rows` in place using the active model: sets category,
    category_label, source='local' and confidence per row. Rows need
    'description' and 'signed_amount'. Raises ModelUnavailable / RuntimeError
    when no usable model exists."""
    pipeline = load_active()
    if pipeline is None:
        raise ModelUnavailable("No trained model yet. Train one first (Setup → Models).")
    if not rows:
        return
    X = rows_to_frame(rows)
    probs = pipeline.predict_proba(X)
    classes = pipeline.classes_
    for r, prob_row in zip(rows, probs):
        best = prob_row.argmax()
        r["category"] = int(classes[best])
        r["category_label"] = category_label(int(classes[best]))
        # Category-level default — the model doesn't predict this per row;
        # refine in review or via the LLM/hand-off paths.
        r["controllable"] = "yes" if default_controllable(int(classes[best])) else "no"
        r["source"] = "local"
        r["confidence"] = round(float(prob_row[best]), 4)
