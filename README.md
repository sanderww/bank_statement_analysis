# Bank Statement Analysis

Extract transactions from FNB PDF bank statements, categorise them (OpenAI LLM
or a local ML model), review and fix the results, and analyse where the money
goes. The local model improves over time: reviewed statements are promoted to a
curated training set, and each retrain produces a new versioned model.

## 🚀 Getting Started

### 1. Installation
Ensure you have `uv` installed, then run:
```bash
make install
```

### 2. Web Interface
```bash
make start-server
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The UI walks through the
workflow:

1. **Process** — extract data from PDF bank statements to CSV. Extraction also
   infers the transaction *direction* (money in/out, from the running balance),
   adds a signed amount, and de-duplicates overlapping statements.
2. **Categorize** — classify transactions with OpenAI (uses the active prompt
   version) or the active local model version (adds a confidence score per row).
3. **Review & Improve** — fix categories inline. Low-confidence predictions are
   highlighted; filter by text, category, or low-confidence only. When a
   statement looks right, **add it to the training data**.
4. **Insights** — income vs costs per month and costs by category for any
   selection of categorised files.
5. **Setup** — manage prompt versions, model versions (train/activate), the
   training data, and settings (e.g. the low-confidence threshold).

![Web UI Screenshot](docs/images/ui_screenshot.png)

### The improvement loop

```
extract → categorise (LLM or local) → review & fix → promote to training data
   ↑                                                        ↓
   └────────────── train new model version ←────────────────┘
```

Categorisation is fine-tuned over time: the model only ever trains from the
explicitly curated set in `models/training_data/` — never silently from raw
output — so you always know what it learned from.

## 🛠 Command Line Usage

- `make extract` — batch extract PDFs to `output/`.
- `make categorize-llm` — extract + categorise via OpenAI (needs `OPENAI_API_KEY`).
- `make categorize-local` — extract + categorise with the active local model version.
- `make train` — train the next model version from `models/training_data/`.
- `make test` — run the test suite.

### Manual Execution
```bash
# Basic Extraction
uv run bank-statement-analysis --all

# Categorization (OpenAI, active prompt version)
uv run bank-statement-analysis --all --categorize

# Categorization (local, active model version)
uv run bank-statement-analysis --all --categorize --categorize-mode local

# Train the next model version
uv run train-transactions-model
```

## 📂 Project Structure
- `bank_statements/` — input PDF files (gitignored — personal data).
- `output/extracted_raw/` — extracted CSVs (gitignored).
- `output/categorised/` — categorised CSVs, edited in the Review step (gitignored).
- `models/v{N}/` — versioned model artefacts + metadata (gitignored).
- `models/training_data/` — the curated training set (gitignored).
- `prompts/v{N}.txt` — versioned categorisation prompts.
- `settings.json` — machine-local state: active prompt/model version, threshold (gitignored).
- `src/` — core Python source; `src/bank_statement_analysis/static/` — web UI.

## 🔒 Privacy

This repo is public. Statements, extracted/categorised CSVs, trained model
artefacts (their vocabulary embeds real transaction descriptions) and training
data are all gitignored — never commit them. Tests use synthetic data only.
The server binds to 127.0.0.1.
