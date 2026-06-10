"""LLM-based query classifier for intelligent routing.

Routes queries to RAG pipeline, direct generation, or out-of-scope handling.
"""

from __future__ import annotations

from enum import Enum

from langchain_community.llms import Ollama

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class QueryClass(str, Enum):
    """Query classification categories.

    RAG_REQUIRED: Factual, topic-specific INPPLC queries that need retrieval.
    DIRECT_RESPONSE: Greetings, meta-questions, or general queries answerable
        without retrieval.
    OUT_OF_SCOPE: Queries unrelated to INPPLC, transparency, or anti-corruption.
    """

    RAG_REQUIRED = "rag_required"
    DIRECT_RESPONSE = "direct_response"
    OUT_OF_SCOPE = "out_of_scope"


CLASSIFICATION_PROMPT = """You are a query router for an INPPLC information system.

Classify the following user query into exactly one category:

1. RAG_REQUIRED: The query is factual or topic-specific about INPPLC missions, \
anti-corruption, probity, transparency, and requires information from the knowledge base to answer accurately.
   Examples: "What are the missions of INPPLC?", "How to report corruption?", \
"What is the national anti-corruption strategy?"

2. DIRECT_RESPONSE: The query is a greeting, meta-question about the system, \
or a general question that can be answered without retrieval.
   Examples: "Hello", "How do you work?", "Thanks for the help", "What can you help me with?"

3. OUT_OF_SCOPE: The query is unrelated to INPPLC, transparency, or anti-corruption topics.
   Examples: "What's the capital of France?", "Help me write Python code"

Query: "{query}"

Respond with ONLY the category name: RAG_REQUIRED, DIRECT_RESPONSE, or OUT_OF_SCOPE."""


def classify_query(
    query: str,
    llm: Ollama,
    settings: Settings | None = None,
) -> QueryClass:
    """Classify a user query for routing.

    Args:
        query: The user's input query.
        llm: LLM instance for classification.
        settings: Application settings.

    Returns:
        QueryClass enum value indicating the routing decision.
    """
    prompt = CLASSIFICATION_PROMPT.format(query=query)

    try:
        response = llm.invoke(prompt)
        
        # Extract content if response is a Message object or mock with .content
        if hasattr(response, "content"):
            raw_content = response.content
        else:
            raw_content = response
            
        result = raw_content.strip().upper() if isinstance(raw_content, str) else str(raw_content).upper()

        if "RAG_REQUIRED" in result:
            classification = QueryClass.RAG_REQUIRED
        elif "DIRECT" in result:
            classification = QueryClass.DIRECT_RESPONSE
        elif "OUT_OF_SCOPE" in result or "OUT" in result:
            classification = QueryClass.OUT_OF_SCOPE
        else:
            # Default to RAG_REQUIRED for safety (avoids missing retrieval)
            classification = QueryClass.RAG_REQUIRED
            logger.warning(
                "classification_fallback",
                raw_response=result[:50],
                defaulted_to="RAG_REQUIRED",
            )

        logger.info(
            "query_classified",
            query=query[:80],
            classification=classification.value,
        )
        return classification

    except Exception as e:
        logger.error("classification_error", error=str(e))
        return QueryClass.RAG_REQUIRED
