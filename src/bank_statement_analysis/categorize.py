import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import logging

from . import config
from .categories import CATEGORY_LABELS, Category, default_controllable

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
    controllable: bool = Field(
        ...,
        description="True if this is a controllable/consumption cost the account "
                    "holder can influence short-term (eating out, petrol, "
                    "electricity usage, groceries); False for fixed/committed "
                    "costs (mortgage, school fees, insurance) and for income.",
    )


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
                controllable=default_controllable(Category.UNKNOWN),
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


# Local-model categorisation lives in model_store.predict_rows (versioned
# models with confidence); the OpenAI path above is the only LLM path.

