import os
from typing import List, Optional
from pathlib import Path
from datetime import date

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel


from .io_utils import write_csv
from .services import extract_data, categorize_data

app = FastAPI()

# Mount static files
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BANK_STATEMENTS_DIR = PROJECT_ROOT / "bank_statements"
OUTPUT_DIR = PROJECT_ROOT / "output"
EXTRACTED_RAW_DIR = OUTPUT_DIR / "extracted_raw"
CATEGORISED_DIR = OUTPUT_DIR / "categorised"

# Ensure output directories exist
EXTRACTED_RAW_DIR.mkdir(parents=True, exist_ok=True)
CATEGORISED_DIR.mkdir(parents=True, exist_ok=True)

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
    if BANK_STATEMENTS_DIR.exists():
        files = [f.name for f in BANK_STATEMENTS_DIR.glob("*.pdf")]
    return {"files": sorted(files)}

@app.get("/api/csv-files")
async def list_csv_files():
    """List CSV files in output/extracted_raw directory."""
    files = []
    if EXTRACTED_RAW_DIR.exists():
        files = [f.name for f in EXTRACTED_RAW_DIR.glob("*.csv")]
    return {"files": sorted(files)}

@app.post("/api/extract")
async def extract_files(req: ExtractRequest):
    """Extract transactions from PDFs to CSV (uncategorized)."""
    if not req.files:
        raise HTTPException(status_code=400, detail="No files provided")

    pdf_paths = [BANK_STATEMENTS_DIR / f for f in req.files]
    
    try:
        all_rows = extract_data(pdf_paths)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting files: {str(e)}")

    if not all_rows:
        return {"message": "No transactions extracted", "output_file": None}

    today = date.today().strftime("%Y-%m-%d_%H-%M")
    output_file = EXTRACTED_RAW_DIR / f"{today}_extracted_raw.csv"
    
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
        paths = [EXTRACTED_RAW_DIR / f for f in req.files]
        try:
            from .services import load_csv_data
            all_rows = load_csv_data(paths)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading CSVs: {str(e)}")
    else:
        # Extract from PDFs
        paths = [BANK_STATEMENTS_DIR / f for f in req.files]
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

    today = date.today().strftime("%Y-%m-%d_%H-%M")
    output_file = CATEGORISED_DIR / f"{today}_categorised_{req.mode}.csv"
    
    write_csv(all_rows, str(output_file), include_category=True)

    return {"message": f"Categorized {len(all_rows)} transactions", "output_file": str(output_file)}
