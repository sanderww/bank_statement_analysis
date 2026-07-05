"""Categorisation hand-off to Claude Code / cowork (no API).

An alternative to the OpenAI path: categorise in a Claude Code / cowork session
via a file round-trip, so nothing leaves the machine through this app.
Ported from the prototype (bank_categoriser/handoff.py), adapted to V1's
file-based flow:

  1. export_for_categorisation(extracted_csv) writes to output/handoff/:
       - request_{stamp}.csv   (id, date, direction, amount, description)
       - instructions_{stamp}.md  (embeds the active versioned prompt)
       - request_{stamp}.meta.json  (which source CSV the ids refer to)
  2. The user categorises in a Claude Code / cowork session and saves
     results_{stamp}.csv (columns: id, category) in the same folder.
  3. import_results(results_name) merges the categories back onto the source
     rows (source='claude') and writes a normal categorised CSV, ready for the
     Review step.
"""
from __future__ import annotations

import csv
import json
import re
from datetime import datetime
from pathlib import Path

from . import config, prompt_store
from .categories import VALID_CODES, default_controllable, label as category_label
from .io_utils import write_csv
from .services import load_csv_data

REQUEST_COLUMNS = ["id", "date", "direction", "amount", "description"]

_RESULTS_RE = re.compile(r"^results_(?P<stamp>[0-9_-]+)\.csv$")


def export_for_categorisation(extracted_csv_name: str) -> dict:
    """Write the request CSV + instructions for one extracted CSV."""
    config.ensure_dirs()
    source_path = config.extracted_raw_dir() / extracted_csv_name
    if not source_path.exists():
        raise FileNotFoundError(f"Extracted CSV not found: {extracted_csv_name}")

    rows = load_csv_data([source_path])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    request_path = config.handoff_dir() / f"request_{stamp}.csv"
    with open(request_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(REQUEST_COLUMNS)
        for i, r in enumerate(rows):
            writer.writerow([
                i, r["date"], r["direction"],
                f'{float(r["signed_amount"]):.2f}', r["description"],
            ])

    prompt_version = prompt_store.active_version()
    try:
        prompt_text = prompt_store.active_text()
    except FileNotFoundError:
        prompt_text = "(no active prompt found)"

    instructions_path = config.handoff_dir() / f"instructions_{stamp}.md"
    instructions_path.write_text(
        _instructions(request_path.name, stamp, prompt_version, prompt_text),
        encoding="utf-8",
    )

    meta_path = config.handoff_dir() / f"request_{stamp}.meta.json"
    meta_path.write_text(json.dumps({
        "source": extracted_csv_name,
        "request": request_path.name,
        "prompt_version": prompt_version,
        "rows": len(rows),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")

    return {
        "request_path": str(request_path),
        "instructions_path": str(instructions_path),
        "rows": len(rows),
        "prompt_version": prompt_version,
        "expected_results_name": f"results_{stamp}.csv",
    }


def _instructions(request_filename: str, stamp: str, prompt_version: str, prompt_text: str) -> str:
    return f"""# Categorisation request

Categorise the transactions in `{request_filename}` using the prompt below
(categorisation prompt **{prompt_version}**).

Each request row has: `id, date, direction, amount, description`.
`direction` is `in` (money in / credit) or `out` (money out / debit); `amount`
is signed accordingly.

**Output:** write a results CSV with columns — `id,category,controllable` — one
row per transaction, where `id` matches the request, `category` is an integer
**0-10**, and `controllable` is `yes`/`no` (yes = a consumption cost the account
holder can influence short-term, e.g. eating out, petrol, electricity usage,
groceries; no = fixed/committed costs and all income/unknown rows). Save it as
`results_{stamp}.csv` in this same folder, then import it in the app
(step 2 → Claude Code hand-off).

## Prompt ({prompt_version})

{prompt_text}
"""


def list_files() -> dict:
    """Hand-off folder contents: open requests and importable results."""
    hdir = config.handoff_dir()
    requests, results = [], []
    if hdir.exists():
        for f in sorted(hdir.glob("request_*.meta.json"), reverse=True):
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                requests.append(meta)
            except json.JSONDecodeError:
                continue
        results = sorted((f.name for f in hdir.glob("results_*.csv")), reverse=True)
    return {"requests": requests, "results": results}


def import_results(results_name: str) -> dict:
    """Merge a results CSV (id, category) back onto its request's source rows
    and write a categorised CSV. Returns counts + the output path."""
    m = _RESULTS_RE.match(results_name)
    if not m:
        raise ValueError(
            f"Results file must be named results_<stamp>.csv (got: {results_name})"
        )
    stamp = m.group("stamp")
    hdir = config.handoff_dir()
    results_path = hdir / results_name
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_name}")
    meta_path = hdir / f"request_{stamp}.meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No matching request for {results_name} (request_{stamp}.meta.json missing)")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    source_path = config.extracted_raw_dir() / meta["source"]
    if not source_path.exists():
        raise FileNotFoundError(f"Source extracted CSV is gone: {meta['source']}")
    rows = load_csv_data([source_path])

    applied = skipped = 0
    with open(results_path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            try:
                idx = int(raw["id"])
                cat = int(raw["category"])
            except (KeyError, TypeError, ValueError):
                skipped += 1
                continue
            if 0 <= idx < len(rows) and cat in VALID_CODES:
                rows[idx]["category"] = cat
                rows[idx]["category_label"] = category_label(cat)
                ctrl = (raw.get("controllable") or "").strip().lower()
                if ctrl not in ("yes", "no"):  # absent/invalid -> category default
                    ctrl = "yes" if default_controllable(cat) else "no"
                rows[idx]["controllable"] = ctrl
                rows[idx]["source"] = "claude"
                applied += 1
            else:
                skipped += 1

    config.ensure_dirs()
    out_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_categorised_claude.csv"
    out_path = config.categorised_dir() / out_name
    write_csv(rows, str(out_path), include_category=True)

    return {"applied": applied, "skipped": skipped, "rows": len(rows),
            "output_file": str(out_path), "output_name": out_name}
