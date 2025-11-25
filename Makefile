.PHONY: install extract categorize-llm train categorize-local help

DATE := $(shell date +%Y-%m-%d)

# Default target
all: help

help:
	@echo "Available commands:"
	@echo "  make install           - Install dependencies using uv"
	@echo "  make extract           - Extract transactions from all PDFs to transactions.csv"
	@echo "  make categorize-llm    - Extract and categorize using OpenAI (requires API key)"
	@echo "  make train             - Train local model using models/training_data/transactions_categorised_training.csv"
	@echo "  make categorize-local  - Extract and categorize using the local trained model"
	@echo "                           Usage: make categorize-local FILES=\"'path/to/file 1.pdf' 'path/to/file 2.pdf'\""

install:
	uv sync

extract:
	uv run bank-statement-analysis --all --output "transactions.csv"

categorize-llm:
	uv run bank-statement-analysis $(if $(FILES),$(FILES),--all) --categorize --model gpt-5-nano --output "models/training_data/$(DATE)_transactions_categorised.csv"

train:
	uv run train-transactions-model --output models/transactions_classifier.joblib

categorize-local:
	uv run bank-statement-analysis $(if $(FILES),$(FILES),--all) --categorize --categorize-mode local --output "output/$(DATE)_transactions_categorised_local.csv"

BRANCH := $(shell git rev-parse --abbrev-ref HEAD)

deploy:
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