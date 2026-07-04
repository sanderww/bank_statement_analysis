import os
from typing import List, Optional
from pathlib import Path
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel


from . import config
from .io_utils import write_csv
from .services import extract_data, categorize_data

app = FastAPI()

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def _ensure_dirs() -> None:
    config.ensure_dirs()

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error categorizing: {str(e)}")

    config.ensure_dirs()
    today = date.today().strftime("%Y-%m-%d_%H-%M")
    output_file = config.categorised_dir() / f"{today}_categorised_{req.mode}.csv"
    
    write_csv(all_rows, str(output_file), include_category=True)

    return {"message": f"Categorized {len(all_rows)} transactions", "output_file": str(output_file)}
