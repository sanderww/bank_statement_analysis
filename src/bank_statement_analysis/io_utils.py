import csv
from pathlib import Path
from typing import Iterable


BASE_FIELDS = ["date", "description", "amount", "balance"]
CAT_FIELDS = BASE_FIELDS + ["category", "category_label"]


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


