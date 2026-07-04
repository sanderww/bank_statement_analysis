"""CLI to train a new local model version from the curated training data.

Thin wrapper over model_store.train + training_data.load_training_rows; the
web UI's Setup → Models section calls the same code. Models are versioned
(models/v{N}/) and the new version becomes active.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from . import model_store, training_data

app = typer.Typer(help="Train a local model to categorize transactions from the curated training data.")


@app.command()
def train(
    csv: Optional[Path] = typer.Option(
        None,
        help="Train from one specific labelled CSV instead of every CSV in models/training_data/",
    ),
) -> None:
    """Train the next model version and make it active."""
    paths = [csv] if csv is not None else training_data.list_training_csvs()
    if not paths:
        raise typer.BadParameter(
            "No training CSVs found. Promote reviewed statements to training data first "
            "(Review step in the UI), or pass --csv."
        )
    for p in paths:
        if not Path(p).exists():
            raise typer.BadParameter(f"CSV not found: {p}")

    rows = training_data.load_training_rows(paths)
    try:
        summary = model_store.train(rows, sources=[str(p) for p in paths])
    except ValueError as e:
        raise typer.BadParameter(str(e))

    typer.echo(
        f"Trained model v{summary['version']} on {summary['training_size']} rows. "
        f"Metrics: {summary['metrics']}"
    )


def main():
    """Entry point for the console script."""
    typer.run(train)


if __name__ == "__main__":
    app()
