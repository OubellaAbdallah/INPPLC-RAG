"""TF-IDF sparse retrieval index.

Loads a pre-built TF-IDF index from pickle for fast startup.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TfidfIndex:
    """TF-IDF sparse retrieval index.

    Loads from a serialized pickle file built during ingestion.
    Eliminates the startup delay of initializing from ChromaDB.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix: Any = None
        self._corpus: list[str] = []
        self._loaded = False

    def load(self, index_path: Path | None = None) -> bool:
        """Load the TF-IDF index from pickle.

        Args:
            index_path: Path to the pickle file. Defaults to settings.

        Returns:
            True if loaded successfully, False otherwise.
        """
        path = index_path or self._settings.tfidf_index_path

        if not path.exists():
            logger.warning("tfidf_index_not_found", path=str(path))
            return False

        try:
            with open(path, "rb") as f:
                data = pickle.load(f)

            self._vectorizer = data["vectorizer"]
            self._tfidf_matrix = data["tfidf_matrix"]
            self._corpus = data["corpus"]
            self._loaded = True

            logger.info(
                "tfidf_index_loaded",
                documents=len(self._corpus),
                path=str(path),
            )
            return True

        except Exception as e:
            logger.error("tfidf_load_error", error=str(e))
            return False

    def build_from_documents(self, documents: list[str]) -> None:
        """Build a TF-IDF index from a list of document texts.

        Used as fallback when no pickle file exists.

        Args:
            documents: List of document text strings.
        """
        self._corpus = documents
        self._vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self._vectorizer.fit_transform(documents)
        self._loaded = True

        logger.info("tfidf_index_built", documents=len(documents))

    @property
    def is_loaded(self) -> bool:
        """Whether the TF-IDF index is loaded and ready."""
        return self._loaded

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[int, float]]:
        """Search the TF-IDF index for relevant documents.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of (document_index, score) tuples sorted by score descending.
        """
        if not self._loaded or self._vectorizer is None or self._tfidf_matrix is None:
            logger.warning("tfidf_not_loaded")
            return []

        query_vec = self._vectorizer.transform([query])
        
        # Calculate cosine similarity between query and all documents
        # cosine_similarity returns a matrix of shape (1, n_docs)
        scores = cosine_similarity(query_vec, self._tfidf_matrix).flatten()

        top_k = min(k, len(scores))
        
        # Avoid argpartition to keep it simple, just sort
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        return [(idx, float(scores[idx])) for idx in top_indices if scores[idx] > 0]

    def get_document(self, index: int) -> str:
        """Retrieve a document by its index.

        Args:
            index: Document index in the corpus.

        Returns:
            Document text content.
        """
        if 0 <= index < len(self._corpus):
            return self._corpus[index]
        return ""

    def search_documents(
        self,
        query: str,
        k: int = 10,
    ) -> list[tuple[Document, float]]:
        """Search and return Document objects with scores.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of (Document, score) tuples.
        """
        results = self.search(query, k)
        return [
            (Document(page_content=self.get_document(idx)), score)
            for idx, score in results
        ]
