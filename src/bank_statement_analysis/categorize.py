import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import load

from . import config
from .categories import CATEGORY_LABELS, Category

# Load env from project root .env
load_dotenv()


logger = logging.getLogger(__name__)


class Transaction(BaseModel):
    date: str  # dd-mm-yyyy
    description: str
    amount: float
    balance: float


class CategorizedTransaction(Transaction):
    category: Category = Field(..., description="Category enum 0..10 (0=Unknown, 10=Income)")
    category_label: str


def _build_system_prompt(version: str = "v1") -> str:
    prompt_path = config.prompts_dir() / f"{version}.txt"
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text().strip()


def categorize_transactions(
    transactions: List[Transaction],
    model: str = "gpt-5-mini",
    prompt_version: str = config.DEFAULT_PROMPT_VERSION,
) -> List[CategorizedTransaction]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    logger.info("Initialized OpenAI client; preparing to categorize %d transactions with model '%s' and prompt '%s'", len(transactions), model, prompt_version)

    categorized: List[CategorizedTransaction] = []
    system_prompt = _build_system_prompt(prompt_version)

    for tx in transactions:
        user_content = (
            "Transaction JSON follows. Return a fully populated CategorizedTransaction.\n" +
            tx.model_dump_json()
        )

        try:
            logger.info("Calling OpenAI.responses.parse for transaction date=%s amount=%.2f desc=%.60s", tx.date, tx.amount, tx.description)
            resp = client.responses.parse(
                model=model,
                input=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_content,
                    },
                ],
                text_format=CategorizedTransaction,
            )
            item = resp.output_parsed
        except Exception as e:
            logger.warning("OpenAI parse failed for transaction date=%s amount=%.2f: %s. Falling back to Unknown.", tx.date, tx.amount, str(e))
            item = CategorizedTransaction(
                **tx.model_dump(),
                category=Category.UNKNOWN,
                category_label=CATEGORY_LABELS[Category.UNKNOWN],
            )

        if not item.category_label:
            item.category_label = CATEGORY_LABELS.get(item.category, str(item.category))
        logger.info(
            "Assigned category=%s label=%s for transaction date=%s amount=%.2f desc=%.60s",
            item.category.name if isinstance(item.category, Category) else str(item.category),
            item.category_label,
            item.date,
            item.amount,
            item.description,
        )
        categorized.append(item)

    return categorized


def _default_local_model_path() -> Path:
    return config.models_dir() / "transactions_classifier.joblib"


def categorize_transactions_local(
    transactions: List[Transaction],
    model_path: Optional[Path] = None,
) -> List[CategorizedTransaction]:
    """
    Categorize using a locally trained scikit-learn pipeline saved via joblib.
    """
    artifact_path = _default_local_model_path() if model_path is None else Path(model_path)
    if not artifact_path.exists():
        raise RuntimeError(f"Local model artifact not found at {artifact_path}. Train one with 'train-transactions-model train'.")

    logger.info("Loading local categorization model from %s", artifact_path)
    artifact = load(artifact_path)
    pipe = artifact["pipeline"]
    metadata = artifact.get("metadata", {})
    category_labels = metadata.get("category_labels", {})

    # Build inference dataframe
    df = pd.DataFrame([t.model_dump() for t in transactions])
    if "description" not in df or "amount" not in df:
        raise ValidationError("Transactions must include 'description' and 'amount' fields.")

    preds = pipe.predict(df[["description", "amount"]])
    results: List[CategorizedTransaction] = []
    for tx, cat_id in zip(transactions, preds):
        try:
            cat_enum = Category(int(cat_id))
        except Exception:
            cat_enum = Category.UNKNOWN
        label = CATEGORY_LABELS.get(cat_enum, category_labels.get(int(cat_enum), str(int(cat_enum))))
        results.append(
            CategorizedTransaction(
                **tx.model_dump(),
                category=cat_enum,
                category_label=label,
            )
        )
    return results

