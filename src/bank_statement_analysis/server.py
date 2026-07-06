from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import date

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel


from . import activity, config, export, handoff, insights, model_store, prompt_store, training_data
from . import settings as app_settings
from .categories import (
    CATEGORY_LABELS,
    Category,
    VALID_CODES,
    default_controllable,
    label as category_label,
)
from .io_utils import read_csv, write_csv
from .services import extract_data, categorize_data, load_csv_data

from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(app: FastAPI):
    config.ensure_dirs()
    yield


app = FastAPI(lifespan=_lifespan)

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

class ExtractRequest(BaseModel):
    files: List[str]

class CategorizeRequest(BaseModel):
    files: List[str]
    mode: str = "openai"  # "openai" or "local"
    model: str = "gpt-5-nano"

@app.get("/")
async def read_index():
    return FileResponse(static_dir / "index.html")

@app.get("/api/files")
async def list_files():
    """List all PDF files in the bank_statements directory."""
    files = []
    if config.bank_statements_dir().exists():
        files = [f.name for f in config.bank_statements_dir().glob("*.pdf")]
    return {"files": sorted(files)}

@app.get("/api/csv-files")
async def list_csv_files():
    """List CSV files in output/extracted_raw directory."""
    files = []
    if config.extracted_raw_dir().exists():
        files = [f.name for f in config.extracted_raw_dir().glob("*.csv")]
    return {"files": sorted(files)}

@app.post("/api/extract")
async def extract_files(req: ExtractRequest):
    """Extract transactions from PDFs to CSV (uncategorized)."""
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided")

    pdf_paths = [config.bank_statements_dir() / f for f in req.files]

    try:
        all_rows = extract_data(pdf_paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting files: {str(e)}")

    if not all_rows:
        return {"message": "No transactions extracted", "output_file": None}

    config.ensure_dirs()
    today = date.today().strftime("%Y-%m-%d_%H-%M")
    output_file = config.extracted_raw_dir() / f"{today}_extracted_raw.csv"
    
    write_csv(all_rows, str(output_file), include_category=False)

    activity.log_event("extract", f"Extracted {len(all_rows)} transactions from "
                                  f"{len(req.files)} PDF(s) → {output_file.name}")
    return {"message": f"Extracted {len(all_rows)} transactions", "output_file": str(output_file)}

@app.post("/api/categorize")
async def categorize_files(req: CategorizeRequest):
    """Extract AND categorize transactions from PDFs OR load from CSVs."""
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided")

    # Determine if we are processing PDFs or CSVs
    # We assume all files in the request are of the same type for simplicity
    # or we check extensions.
    
    first_file = req.files[0]
    is_csv = first_file.lower().endswith(".csv")
    
    if is_csv:
        # Load from extracted_raw
        paths = [config.extracted_raw_dir() / f for f in req.files]
        try:
            from .services import load_csv_data
            all_rows = load_csv_data(paths)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading CSVs: {str(e)}")
    else:
        # Extract from PDFs
        paths = [config.bank_statements_dir() / f for f in req.files]
        try:
            all_rows = extract_data(paths)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error extracting PDFs: {str(e)}")

    if not all_rows:
        return {"message": "No transactions found", "output_file": None}

    try:
        categorize_data(all_rows, mode=req.mode, model=req.model)
    except model_store.ModelUnavailable as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error categorizing: {str(e)}")

    config.ensure_dirs()
    today = date.today().strftime("%Y-%m-%d_%H-%M")
    output_file = config.categorised_dir() / f"{today}_categorised_{req.mode}.csv"

    write_csv(all_rows, str(output_file), include_category=True)

    activity.log_event("categorise", f"Categorised {len(all_rows)} transactions "
                                     f"({req.mode}) → {output_file.name}")
    return {"message": f"Categorized {len(all_rows)} transactions", "output_file": str(output_file)}


# --------------------------------------------------------------------------
# Categories / settings
# --------------------------------------------------------------------------

@app.get("/api/categories")
async def get_categories():
    """The fixed 0-10 category set (+ default controllable flag) for the UI."""
    return {"categories": [
        {"code": int(c), "label": CATEGORY_LABELS[c],
         "default_controllable": default_controllable(int(c))}
        for c in Category
    ]}


class SettingsUpdate(BaseModel):
    confidence_threshold: Optional[float] = None
    openai_model: Optional[str] = None


@app.get("/api/settings")
async def get_settings():
    return app_settings.load()


@app.put("/api/settings")
async def update_settings(req: SettingsUpdate):
    values = {k: v for k, v in req.model_dump().items() if v is not None}
    if "confidence_threshold" in values and not (0.0 <= values["confidence_threshold"] <= 1.0):
        raise HTTPException(status_code=400, detail="confidence_threshold must be between 0 and 1")
    saved = app_settings.save(values)
    activity.log_event("settings", "Settings updated: " + ", ".join(f"{k}={v}" for k, v in values.items()))
    return saved


# --------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------

class PromptCreate(BaseModel):
    text: str


class PromptActivate(BaseModel):
    version: str


@app.get("/api/prompts")
async def list_prompts():
    return {"prompts": prompt_store.list_versions(), "active": prompt_store.active_version()}


@app.post("/api/prompts")
async def create_prompt(req: PromptCreate):
    try:
        version = prompt_store.add_version(req.text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    activity.log_event("prompts", f"New prompt version {version} created and activated")
    return {"message": f"Saved prompt {version} and made it active", "version": version}


@app.post("/api/prompts/activate")
async def activate_prompt(req: PromptActivate):
    try:
        prompt_store.set_active(req.version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    activity.log_event("prompts", f"Prompt {req.version} activated")
    return {"message": f"Prompt {req.version} is now active", "active": req.version}


@app.put("/api/prompts/{version}")
async def update_prompt(version: str, req: PromptCreate):
    """Edit an existing prompt version in place."""
    try:
        prompt_store.update(version, req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    activity.log_event("prompts", f"Prompt {version} edited in place")
    return {"message": f"Prompt {version} updated", "version": version}


# --------------------------------------------------------------------------
# Models & training data
# --------------------------------------------------------------------------

class ModelActivate(BaseModel):
    version: int


@app.get("/api/models")
async def list_models():
    return {"models": model_store.list_versions(), "active": model_store.active_version()}


@app.post("/api/models/activate")
async def activate_model(req: ModelActivate):
    try:
        model_store.set_active(req.version)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    activity.log_event("training", f"Model v{req.version} activated")
    return {"message": f"Model v{req.version} is now active", "active": req.version}


@app.post("/api/models/train")
async def train_model():
    """Train the next model version from the curated training data."""
    rows = training_data.load_training_rows()
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="No training data yet. Review a statement and promote it to training data first.",
        )
    try:
        summary = model_store.train(
            rows, sources=[p.name for p in training_data.list_training_csvs()]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    activity.log_event("training", f"Trained model v{summary['version']} on "
                                   f"{summary['training_size']} rows — {summary['metrics']}")
    return {"message": f"Trained model v{summary['version']} on {summary['training_size']} rows",
            **summary}


class ModelEvaluate(BaseModel):
    file: str  # a reviewed categorised CSV to test against


@app.post("/api/models/evaluate")
async def evaluate_model(req: ModelEvaluate):
    """Test the active model against a reviewed file's categories — the
    'is it good enough yet?' check in the improvement flow."""
    path = _safe_categorised_path(req.file)
    rows = read_csv(str(path))
    for r in rows:
        r["signed_amount"] = float(r.get("signed_amount") or 0.0)
    try:
        res = model_store.evaluate(rows)
    except model_store.ModelUnavailable as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    activity.log_event("training", f"Evaluated model v{res['model_version']} against {req.file}: "
                                   f"{res['accuracy']:.0%} on {res['n_rows']} rows")
    return {
        "message": f"Model v{res['model_version']} scores {res['accuracy']:.0%} "
                   f"on {res['n_rows']} reviewed rows of {req.file}",
        **res,
    }


@app.get("/api/training-data")
async def training_data_stats():
    return training_data.stats()


# --------------------------------------------------------------------------
# Review
# --------------------------------------------------------------------------

def _safe_categorised_path(filename: str) -> Path:
    """Resolve a categorised CSV by name, refusing path traversal."""
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = config.categorised_dir() / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    return path


@app.get("/api/categorised-files")
async def list_categorised_files():
    """List categorised CSVs (newest first) for the review & insights steps."""
    files = []
    if config.categorised_dir().exists():
        files = sorted((f.name for f in config.categorised_dir().glob("*.csv")), reverse=True)
    return {"files": files}


@app.get("/api/review/{filename}")
async def load_review(filename: str):
    """A categorised CSV as JSON rows for the review table."""
    path = _safe_categorised_path(filename)
    rows = read_csv(str(path))
    for i, r in enumerate(rows):
        r["row"] = i
        for k in ("amount", "balance", "signed_amount"):
            if r.get(k) not in (None, ""):
                r[k] = float(r[k])
        r["category"] = int(float(r["category"])) if r.get("category") not in (None, "") else None
        r["confidence"] = float(r["confidence"]) if r.get("confidence") not in (None, "") else None
        r["controllable"] = r.get("controllable") or ""
    return {
        "filename": filename,
        "rows": rows,
        "confidence_threshold": app_settings.get("confidence_threshold"),
    }


class ReviewSave(BaseModel):
    rows: List[Dict[str, Any]]


@app.put("/api/review/{filename}")
async def save_review(filename: str, req: ReviewSave):
    """Persist review edits back to the categorised CSV. The category label is
    recomputed server-side; rows the user changed should carry source='user'."""
    path = _safe_categorised_path(filename)
    cleaned: List[Dict[str, Any]] = []
    for r in req.rows:
        cat = r.get("category")
        if cat in (None, ""):
            cat = None
        else:
            cat = int(cat)
            if cat not in VALID_CODES:
                raise HTTPException(status_code=400, detail=f"Invalid category code: {cat}")
        controllable = (r.get("controllable") or "").strip().lower()
        if controllable not in ("", "yes", "no"):
            raise HTTPException(status_code=400, detail=f"controllable must be yes/no/empty, got: {controllable}")
        if cat is not None and controllable == "":
            # pre-fill the category default when the user hasn't set it
            controllable = "yes" if default_controllable(cat) else "no"
        cleaned.append({
            "date": r.get("date", ""),
            "description": r.get("description", ""),
            "amount": r.get("amount", ""),
            "balance": r.get("balance", ""),
            "direction": r.get("direction", ""),
            "signed_amount": r.get("signed_amount", ""),
            "source_statement": r.get("source_statement", ""),
            "category": cat if cat is not None else "",
            "category_label": category_label(cat) if cat is not None else "",
            "controllable": controllable if cat is not None else "",
            "source": r.get("source", ""),
            "confidence": r.get("confidence", "") if r.get("confidence") is not None else "",
        })
    write_csv(cleaned, str(path), include_category=True)
    n_user = sum(1 for r in cleaned if r["source"] == "user")
    activity.log_event("review", f"Saved review of {filename}: {len(cleaned)} rows "
                                 f"({n_user} user-corrected)")
    return {"message": f"Saved {len(cleaned)} rows to {filename}"}


@app.post("/api/review/{filename}/promote")
async def promote_to_training(filename: str):
    """Add the file's categorised rows to the curated training data (deduped)."""
    path = _safe_categorised_path(filename)
    rows = read_csv(str(path))
    res = training_data.promote_rows(rows)
    activity.log_event("training", f"Promoted {filename} to training data: "
                                   f"{res['added']} added, {res['skipped_duplicate']} duplicates, "
                                   f"{res['skipped_invalid']} uncategorised skipped")
    return {
        "message": (
            f"Added {res['added']} rows to training data "
            f"({res['skipped_duplicate']} duplicates, {res['skipped_invalid']} uncategorised skipped)"
        ),
        **res,
    }


@app.post("/api/review/{filename}/export")
async def export_final_csv(filename: str):
    """Export the file's categorised rows as a decoupled final CSV."""
    _safe_categorised_path(filename)
    res = export.export_final(filename)
    activity.log_event("export", f"Final CSV exported from {filename}: "
                                 f"{res['rows']} rows → {res['name']}")
    return {
        "message": (
            f"Exported {res['rows']} rows to {res['name']} "
            f"({res['skipped_uncategorised']} uncategorised skipped)"
        ),
        **res,
    }


# --------------------------------------------------------------------------
# Claude Code / cowork hand-off (no API)
# --------------------------------------------------------------------------

class HandoffExport(BaseModel):
    file: str  # an extracted_raw CSV name


class HandoffImport(BaseModel):
    results: str  # results_<stamp>.csv in output/handoff/


@app.get("/api/handoff")
async def handoff_files():
    return handoff.list_files()


@app.post("/api/handoff/export")
async def handoff_export(req: HandoffExport):
    if Path(req.file).name != req.file:
        raise HTTPException(status_code=400, detail="Invalid filename")
    try:
        res = handoff.export_for_categorisation(req.file)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    activity.log_event("handoff", f"Exported {res['rows']} transactions of {req.file} "
                                  f"for Claude Code (prompt {res['prompt_version']})")
    return {
        "message": (
            f"Exported {res['rows']} transactions (prompt {res['prompt_version']}). "
            f"Categorise them in a Claude Code session, save as {res['expected_results_name']}, then import."
        ),
        **res,
    }


@app.post("/api/handoff/import")
async def handoff_import(req: HandoffImport):
    if Path(req.results).name != req.results:
        raise HTTPException(status_code=400, detail="Invalid filename")
    try:
        res = handoff.import_results(req.results)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    activity.log_event("handoff", f"Imported {res['applied']} categories from {req.results} "
                                  f"→ {res['output_name']}")
    return {
        "message": (
            f"Imported {res['applied']} categories ({res['skipped']} skipped). "
            f"Review the result: {res['output_name']}"
        ),
        **res,
    }


# --------------------------------------------------------------------------
# Activity log
# --------------------------------------------------------------------------

@app.get("/api/activity")
async def get_activity(category: Optional[str] = None, limit: int = 100):
    """Recent activity entries, newest first, optionally filtered by category."""
    return {
        "entries": activity.recent(limit=min(limit, 500), category=category),
        "categories": list(activity.CATEGORIES),
    }


# --------------------------------------------------------------------------
# Insights
# --------------------------------------------------------------------------

@app.get("/api/insights")
async def get_insights(files: List[str] = Query(default=[])):
    """Aggregates for the Insights charts across the selected categorised CSVs."""
    if not files:
        raise HTTPException(status_code=400, detail="No files selected")
    rows: List[Dict[str, Any]] = []
    for name in files:
        path = _safe_categorised_path(name)
        rows.extend(read_csv(str(path)))
    return insights.aggregate(rows)
