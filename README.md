to create t

PYTHONPATH=src python -m bank_statement_analysis.main --all --output "/Users/sanderwiersma/Documents/dev_projects/bank_statement_analysis/transactions.csv"


## 1. Quick Start (Makefile)

The easiest way to run the project is using the provided `Makefile`.

```bash
# 1. Install dependencies
make install

# 2. Extract data from PDFs (Use Case 1)
make extract
# Output: transactions.csv

# 3. Categorise data via LLM (Use Case 2)
# Processes all files by default, or specify FILES="..."
make categorize-llm
# Output: models/training_data/{YYYY-MM-DD}_transactions_categorised.csv

# 4. Train model with extracted data (Use Case 3)
make train
# Output: models/transactions_classifier.joblib

# 5. Categorise new bank statements with local model (Use Case 4)
# Processes all files by default
make categorize-local

# Or specify specific files (use single quotes for paths with spaces)
make categorize-local FILES="'path/to/file 1.pdf' 'path/to/file 2.pdf'"
# Output: output/{YYYY-MM-DD}_transactions_categorised_local.csv
```

## 2. How to Run from Project Root (Manual)

### Without Installing (Works Reliably)

```bash
PYTHONPATH=src python -m bank_statement_analysis.main --all --output "/Users/sanderwiersma/Documents/dev_projects/bank_statement_analysis/transactions.csv"
```

#### Add Categorization

```bash
PYTHONPATH=src python -m bank_statement_analysis.main --all --categorize --model gpt-5-mini --output "/Users/sanderwiersma/Documents/dev_projects/bank_statement_analysis/transactions_categorised.csv"
```

### Using CLI Name (after `uv sync`)

```bash
bank-statement-analysis --all --output "/Users/sanderwiersma/Documents/dev_projects/bank_statement_analysis/transactions.csv"
```

```bash
bank-statement-analysis --all --categorize --model o4-mini --output "/Users/sanderwiersma/Documents/dev_projects/bank_statement_analysis/transactions_categorised.csv"
```

> **Note:**  
> Omit the `run` subcommand; your current CLI uses root-level options.

### Train a Local Categorization Model

Use the labeled CSV (`models/training_data/transactions_categorised_training.csv`) to train a local scikit-learn model.

Without installing:
```bash
PYTHONPATH=src python -m bank_statement_analysis.train_model --output models/transactions_classifier.joblib
```

Using the CLI name (after `uv sync`):
```bash
uv run train-transactions-model --output models/transactions_classifier.joblib
```

To use a different training CSV file:
```bash
PYTHONPATH=src python -m bank_statement_analysis.train_model --csv path/to/your/training.csv --output models/transactions_classifier.joblib
```

Optional flags:
- `--ngram-max 2` (TF-IDF n-grams)
- `--C 2.0` (LogReg regularization inverse)
- `--max-iter 1000`

### Categorize Using the Local Model

Without installing:
```bash
PYTHONPATH=src python -m bank_statement_analysis.main --all --categorize --categorize-mode local --output "/Users/sanderwiersma/Documents/dev_projects/bank_statement_analysis/transactions_categorised.csv"
```

Using the CLI name (after `uv sync`):
```bash
bank-statement-analysis --all --categorize --categorize-mode local --output "/Users/sanderwiersma/Documents/dev_projects/bank_statement_analysis/transactions_categorised.csv"
```

If your model is saved elsewhere, pass its path:
```bash
bank-statement-analysis --all --categorize --categorize-mode local --local-model-path path/to/transactions_classifier.joblib
```
