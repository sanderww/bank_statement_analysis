from datetime import date
from pathlib import Path
from typing import Optional

import typer

from .extract import extract_transactions_from_pdf
from .categorize import Transaction, categorize_transactions, categorize_transactions_local
from .io_utils import write_csv


app = typer.Typer(help="Extract transactions from FNB PDF statements and export to CSV.")


def _gather_pdfs(input_pdfs: Optional[list[Path]], all_: bool) -> list[Path]:
    project_root = Path(__file__).parent.parent.parent
    statements_dir = project_root / "bank_statements"
    pdfs: list[Path] = []
    if all_:
        pdfs.extend(sorted(statements_dir.glob("*.pdf")))
    if input_pdfs:
        pdfs.extend(input_pdfs)
    # Deduplicate while preserving order
    deduped: list[Path] = []
    seen = set()
    for p in pdfs:
        key = p.resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


@app.command()
def run(
    files: Optional[list[Path]] = typer.Argument(None, help="Specific PDF files to process"),
    all: bool = typer.Option(False, "--all/--no-all", help="Process all PDFs in bank_statements/"),
    output: Path = typer.Option(Path("transactions.csv"), help="Output CSV path"),
    append: bool = typer.Option(False, "--append/--no-append", help="Append to CSV if exists"),
    categorize: bool = typer.Option(False, "--categorize/--no-categorize", help="Add category"),
    categorize_mode: str = typer.Option("openai", help="Categorization mode: 'openai' or 'local'"),
    model: str = typer.Option("o4-mini", help="OpenAI model for categorization (when mode=openai)"),
    prompt_version: str = typer.Option("v1", help="Version of the system prompt to use (e.g. 'v1')"),
    local_model_path: Path = typer.Option(Path("models/transactions_classifier.joblib"), help="Path to local model (when mode=local)"),
):
    pdfs = _gather_pdfs(files, all)
    if not pdfs:
        typer.echo("No PDFs to process.")
        raise typer.Exit(code=1)
    
    # Generate output filename in format: {date}_output_transactions_{categorize_mode}
    if output == Path("transactions.csv"):  # Use default format only if using default output
        project_root = Path(__file__).parent.parent.parent
        output_dir = project_root / "output"
        output_dir.mkdir(exist_ok=True)
        mode = categorize_mode if categorize else "uncategorized"
        today = date.today().strftime("%Y-%m-%d")
        output = output_dir / f"{today}_output_transactions_{mode}.csv"

    all_rows: list[dict] = []
    for p in pdfs:
        typer.echo(f"Extracting from {p} …")
        rows = extract_transactions_from_pdf(str(p))
        all_rows.extend(rows)

    if categorize and all_rows:
        typer.echo(f"Categorizing ({categorize_mode}) …")
        tx_models = [
            Transaction(**{
                "date": r["date"],
                "description": r["description"],
                "amount": float(r["amount"]),
                "balance": float(r["balance"]),
            })
            for r in all_rows
        ]
        if categorize_mode.lower() == "local":
            categorized = categorize_transactions_local(tx_models, model_path=local_model_path)
        elif categorize_mode.lower() == "openai":
            categorized = categorize_transactions(tx_models, model=model, prompt_version=prompt_version)
        else:
            raise typer.BadParameter("categorize_mode must be 'openai' or 'local'")
        # Merge category back
        for i, c in enumerate(categorized):
            all_rows[i]["category"] = int(c.category)
            all_rows[i]["category_label"] = c.category_label

    write_csv(all_rows, str(output), include_category=categorize, append=append)
    typer.echo(f"Wrote {len(all_rows)} rows → {output}")


if __name__ == "__main__":
    app()