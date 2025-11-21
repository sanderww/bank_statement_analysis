to create t

PYTHONPATH=src python -m bank_statement_analysis.main --all --output "/Users/sanderwiersma/Documents/dev_projects/bank_statement_analysis/transactions.csv"


## 2. How to Run from Project Root

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
