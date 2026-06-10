"""ChromaDB vector store management.

Provides factory functions for creating and loading ChromaDB collections
with persistent storage to data/indices/chroma_db/.

Compatible with chromadb >= 1.3 (PersistentClient API).
"""

from __future__ import annotations

import chromadb
from langchain_chroma import Chroma

from src.config.settings import Settings, get_settings
from src.retrieval.embeddings import EmbeddingModel
from src.utils.logging import get_logger

logger = get_logger(__name__)


def get_vectorstore(settings: Settings | None = None) -> Chroma:
    """Load the ChromaDB vector store using the PersistentClient API.

    Creates a :class:`chromadb.PersistentClient` pointed at the configured
    ``chroma_db_dir``, then wraps it in a LangChain :class:`~langchain_chroma.Chroma`
    instance for compatibility with the rest of the pipeline.

    Args:
        settings: Application settings. Uses ``get_settings()`` if not provided.

    Returns:
        Configured :class:`~langchain_chroma.Chroma` vector store instance.
    """
    settings = settings or get_settings()
    embedding_model = EmbeddingModel(settings)

    # chromadb >= 1.3 uses PersistentClient instead of persist_directory kwarg
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))

    vectorstore = Chroma(
        client=chroma_client,
        collection_name=settings.collection_name,
        embedding_function=embedding_model,
    )

    logger.info(
        "vectorstore_loaded",
        collection=settings.collection_name,
        persist_dir=str(settings.chroma_db_dir),
    )

    return vectorstore
