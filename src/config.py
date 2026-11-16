"""Configuration and cost helpers for Day 7 Agentic RAG."""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

# Gemini pricing estimates (USD per 1M tokens) for transparent cost reporting.
# gemini-embedding-001: ~$0.15 / 1M input tokens (~4 chars/token)
# gemini-2.5-flash: ~$0.15 / 1M input tokens, ~$0.60 / 1M output tokens
EMBED_USD_PER_1M_TOKENS = 0.15
FLASH_INPUT_USD_PER_1M = 0.15
FLASH_OUTPUT_USD_PER_1M = 0.60


@dataclass(frozen=True)
class Settings:
    database_url: str
    gemini_api_key: str
    embedding_model: str
    agent_model: str
    day6_strategy: str


def get_settings() -> Settings:
    db_url = os.getenv("DATABASE_URL", "").strip()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL is missing in .env")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in .env")
    return Settings(
        database_url=db_url,
        gemini_api_key=api_key,
        embedding_model=os.getenv("EMBEDDING_MODEL", "gemini-embedding-001").strip(),
        agent_model=os.getenv("AGENT_MODEL", "gemini-3.5-flash-lite").strip(),
        day6_strategy=os.getenv("DAY6_STRATEGY", "A").strip(),
    )


def estimate_embed_cost_usd(chars: int) -> float:
    tokens = chars / 4.0
    return (tokens / 1_000_000.0) * EMBED_USD_PER_1M_TOKENS


def estimate_llm_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        (input_tokens / 1_000_000.0) * FLASH_INPUT_USD_PER_1M
        + (output_tokens / 1_000_000.0) * FLASH_OUTPUT_USD_PER_1M
    )
