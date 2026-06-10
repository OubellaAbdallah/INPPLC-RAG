"""Query translation utility for multilingual search optimization."""

from __future__ import annotations

import httpx
from src.config.settings import Settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def translate_query_to_french(query: str, settings: Settings) -> str:
    """Translate user search query to French if needed to optimize retrieval.

    Since the document store is in French, matching English or Arabic queries
    directly against the index results in poor recall. Translating the search
    query to French resolves this, while preserving the user's original language
    in the final LLM response generation step.

    Args:
        query: Original query string.
        settings: Application settings containing API credentials.

    Returns:
        French translation of the query, or the original query if translation fails
        or is unnecessary.
    """
    if not query.strip():
        return query

    # Quick check: if the query contains only simple numbers or is very short, skip
    if len(query.strip()) < 3:
        return query

    import time

    max_retries = 3
    delay = 1.0

    for attempt in range(max_retries):
        try:
            with httpx.Client() as client:
                resp = client.post(
                    "https://api.mistral.ai/v1/chat/completions",
                    json={
                        "model": settings.orchestrator_model_name or "mistral-large-latest",
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a translation assistant. Translate the user query into French. "
                                    "If the query is already in French, return it exactly as is, without any changes. "
                                    "Do NOT add any quotes, comments, greetings, or formatting. Output ONLY the translated query."
                                )
                            },
                            {
                                "role": "user",
                                "content": query
                            }
                        ],
                        "temperature": 0.0,
                        "max_tokens": 150,
                    },
                    headers={
                        "Authorization": f"Bearer {settings.mistral_api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=10.0,
                )
                
                # Check for rate limits specifically
                if resp.status_code == 429:
                    if attempt < max_retries - 1:
                        logger.warning("query_translation_rate_limited", attempt=attempt + 1, delay=delay)
                        time.sleep(delay)
                        delay *= 2.0
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                translated = resp.json()["choices"][0]["message"]["content"].strip()
                
                # Clean up potential LLM artifact quotes
                if translated.startswith('"') and translated.endswith('"'):
                    translated = translated[1:-1].strip()
                if translated.startswith("'") and translated.endswith("'"):
                    translated = translated[1:-1].strip()

                if translated:
                    logger.info("query_translated_to_french", original=query, translated=translated)
                    return translated

        except Exception as exc:
            if attempt < max_retries - 1:
                logger.warning("query_translation_retry", attempt=attempt + 1, delay=delay, error=str(exc))
                time.sleep(delay)
                delay *= 2.0
                continue
            logger.warning("query_translation_failed", error=str(exc), query=query)
            return query

    return query
