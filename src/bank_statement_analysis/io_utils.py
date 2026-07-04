import csv
from pathlib import Path
from typing import Iterable


BASE_FIELDS = ["date", "description", "amount", "balance", "direction", "signed_amount", "source_statement"]
CAT_FIELDS = BASE_FIELDS + ["category", "category_label", "source", "confidence"]


def write_csv(rows: Iterable[dict], output_path: str, include_category: bool, append: bool = False) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = CAT_FIELDS if include_category else BASE_FIELDS
    file_exists = path.exists()
    mode = "a" if append else "w"

    with path.open(mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append or not file_exists:
            writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})



def read_csv(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)
