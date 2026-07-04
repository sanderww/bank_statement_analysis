.PHONY: install start-server test extract categorize-llm train categorize-local help

DATE := $(shell date +%Y-%m-%d)

# Default target
all: help

help:
	@echo "Available commands:"
	@echo "  make install           - Install dependencies using uv"
	@echo "  make start-server      - Start the web UI at http://127.0.0.1:8000"
	@echo "  make test              - Run the test suite"
	@echo "  make extract           - Extract transactions from all PDFs"
	@echo "  make categorize-llm    - Extract and categorize using OpenAI (requires API key)"
	@echo "  make categorize-local  - Extract and categorize using the active local model version"
	@echo "  make train             - Train the next local model version from models/training_data/"
	@echo "                           Usage: make categorize-local FILES=\"'path/to/file 1.pdf' 'path/to/file 2.pdf'\""
start-server:
	uv run uvicorn bank_statement_analysis.server:app --host 127.0.0.1 --reload

test:
	uv run pytest -q

extract:
	uv run bank-statement-analysis $(if $(FILES),$(FILES),--all) --output "output/$(DATE)_transactions.csv"

categorize-llm:
	uv run bank-statement-analysis $(if $(FILES),$(FILES),--all) --categorize --output "output/categorised/$(DATE)_transactions_categorised.csv"

train:
	uv run train-transactions-model

categorize-local:
	uv run bank-statement-analysis $(if $(FILES),$(FILES),--all) --categorize --categorize-mode local --output "output/categorised/$(DATE)_transactions_categorised_local.csv"

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

merge-to-main:
	@if [ "$(BRANCH)" != "dev" ]; then \
		echo "❌ You must be on the 'dev' branch (current: $(BRANCH))"; \
		exit 1; \
	fi

	@if [ -z "$(m)" ]; then \
		echo "❌ Please provide a commit message using m=\"your message\""; \
		exit 1; \
	fi

	git add .
	git commit -m "$(m)"

	git checkout main
	git merge dev
	git checkout dev