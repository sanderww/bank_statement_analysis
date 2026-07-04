from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from joblib import dump

from . import config
from .categories import CATEGORY_LABELS, Category


app = typer.Typer(help="Train a local model to categorize transactions from a labeled CSV.")

def _to_2d_array(x):
    arr = getattr(x, "values", x)
    return np.asarray(arr).reshape(-1, 1)


def _default_model_path() -> Path:
    models_dir = config.models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir / "transactions_classifier.joblib"


def _default_training_data_path() -> Path:
    training_data_dir = config.training_data_dir()
    training_data_dir.mkdir(parents=True, exist_ok=True)
    return training_data_dir / "transactions_categorised_training.csv"


def _load_training_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"description", "amount", "category"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV at {csv_path} missing required columns: {', '.join(sorted(missing))}")
    # Ensure correct dtypes
    df["description"] = df["description"].astype(str)
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    df["category"] = pd.to_numeric(df["category"], downcast="integer", errors="coerce")
    df = df.dropna(subset=["category"])
    df["category"] = df["category"].astype(int)
    # Clip absurd amounts to reduce outlier influence
    df["amount"] = df["amount"].clip(lower=-1_000_000, upper=1_000_000)
    return df


@app.command()
def train(
    csv: Optional[Path] = typer.Option(None, help="Path to labeled transactions CSV (defaults to models/training_data/transactions_categorised_training.csv)"),
    output: Optional[Path] = typer.Option(None, help="Where to save the trained model .joblib"),
    max_iter: int = typer.Option(1000, help="Max iterations for LogisticRegression"),
    C: float = typer.Option(2.0, help="Inverse of regularization strength for LogisticRegression"),
    ngram_max: int = typer.Option(2, help="Max n-gram length for TF-IDF over description"),
) -> None:
    """
    Train a scikit-learn pipeline on labeled transactions and save the artifact.
    """
    csv_path = csv if csv is not None else _default_training_data_path()
    if not csv_path.exists():
        raise typer.BadParameter(f"CSV not found: {csv_path}")

    df = _load_training_data(csv_path)
    X = df[["description", "amount"]]
    y = df["category"]

    # Feature pipelines
    text_features = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, ngram_max), min_df=2)),
        ]
    )
    numeric_features = Pipeline(
        steps=[
            ("reshape", FunctionTransformer(_to_2d_array, validate=False)),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("desc", text_features, "description"),
            ("amt", numeric_features, "amount"),
        ]
    )

    clf = LogisticRegression(
        max_iter=max_iter,
        C=C,
        multi_class="auto",
        class_weight="balanced",
    )

    pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("clf", clf),
        ]
    )

    pipe.fit(X, y)

    out_path = _default_model_path() if output is None else Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "category_labels": {int(k): v for k, v in CATEGORY_LABELS.items()},
        "categories": {int(c.value): c.name for c in Category},
        "source_csv": str(csv_path.resolve()),
    }

    dump({"pipeline": pipe, "metadata": metadata}, out_path)
    typer.echo(f"Saved model to {out_path}")


if __name__ == "__main__":
    app()

def main():
    """
    Entry point for console script to run the train() command without a subcommand.
    """
    typer.run(train)


