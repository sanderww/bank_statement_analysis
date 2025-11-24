import os
from enum import IntEnum
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from joblib import load

# Load env from project root .env
load_dotenv()


logger = logging.getLogger(__name__)


class Category(IntEnum):
    HOUSING_UTILITIES = 1
    GROCERIES_HOUSEHOLD = 2
    CHILDCARE_EDUCATION = 3
    TRANSPORT = 4
    HEALTH_INSURANCE = 5
    FOOD_DINING = 6
    CLOTHING_PERSONAL_CARE = 7
    LEISURE_ENTERTAINMENT = 8
    FINANCIAL_MISC = 9


CATEGORY_LABELS = {
    Category.HOUSING_UTILITIES: "Housing & Utilities",
    Category.GROCERIES_HOUSEHOLD: "Groceries & Household",
    Category.CHILDCARE_EDUCATION: "Childcare & Education",
    Category.TRANSPORT: "Transport",
    Category.HEALTH_INSURANCE: "Health & Insurance",
    Category.FOOD_DINING: "Food & Dining",
    Category.CLOTHING_PERSONAL_CARE: "Clothing & Personal Care",
    Category.LEISURE_ENTERTAINMENT: "Leisure & Entertainment",
    Category.FINANCIAL_MISC: "Financial & Miscellaneous",
}


class Transaction(BaseModel):
    date: str  # dd-mm-yyyy
    description: str
    amount: float
    balance: float


class CategorizedTransaction(Transaction):
    category: Category = Field(..., description="Category enum 1..9")
    category_label: str


def _build_system_prompt() -> str:
    return (
        "You are a helpful financial categorization assistant. "
        "Given a bank transaction with description and amount, assign one of the following categories strictly as an integer 1..9: "
        "1 Housing & Utilities; 2 Groceries & Household; 3 Childcare & Education; 4 Transport; 5 Health & Insurance; "
        "6 Food & Dining; 7 Clothing & Personal Care; 8 Leisure & Entertainment; 9 Financial & Miscellaneous; 0. Unknown"
        """No that:
- "Fhps Fees" are school fees
- "Miway" is car insurance
- "Jicamasalariwie8508" is salary"""
        "try to pute all transactions in a category but make sure that to mark transaction as unknown if it's its too hard to categorise "
    )


def _json_schema_for_model() -> dict:
    return {
        "name": "CategorizedTransaction",
        "schema": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "date": {"type": "string"},
                "description": {"type": "string"},
                "amount": {"type": "number"},
                "balance": {"type": "number"},
                "category": {"type": "integer", "minimum": 1, "maximum": 9},
                "category_label": {"type": "string"},
            },
            "required": [
                "date",
                "description",
                "amount",
                "balance",
                "category",
                "category_label",
            ],
        },
        "strict": True,
    }


def categorize_transactions(transactions: List[Transaction], model: str = "gpt-5-mini") -> List[CategorizedTransaction]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)
    logger.info("Initialized OpenAI client; preparing to categorize %d transactions with model '%s'", len(transactions), model)

    categorized: List[CategorizedTransaction] = []
    for tx in transactions:
        system_prompt = _build_system_prompt()
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
            logger.warning("OpenAI parse failed for transaction date=%s amount=%.2f: %s. Falling back to default category.", tx.date, tx.amount, str(e))
            item = CategorizedTransaction(
                **tx.model_dump(),
                category=Category.FINANCIAL_MISC,
                category_label=CATEGORY_LABELS[Category.FINANCIAL_MISC],
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
    return Path(__file__).parent.parent.parent / "models" / "transactions_classifier.joblib"


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
            cat_enum = Category.FINANCIAL_MISC
        label = CATEGORY_LABELS.get(cat_enum, category_labels.get(int(cat_enum), str(int(cat_enum))))
        results.append(
            CategorizedTransaction(
                **tx.model_dump(),
                category=cat_enum,
                category_label=label,
            )
        )
    return results

