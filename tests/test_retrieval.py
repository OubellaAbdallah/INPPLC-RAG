"""Tests for the retrieval pipeline modules."""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

class TestTfidfIndex:
    """Tests for TF-IDF sparse index."""

    def test_build_and_search(self) -> None:
        """Should build index and return relevant results."""
        from src.retrieval.tfidf_index import TfidfIndex

        index = TfidfIndex()
        docs = [
            "Anti-corruption reports are important for public transparency",
            "Whistleblower protection helps prevent corruption",
            "Python is a programming language",
        ]
        index.build_from_documents(docs)

        results = index.search("corruption transparency", k=2)
        assert len(results) == 2
        # First result should be the most relevant
        assert results[0][0] == 0  # Index of transparency/corruption doc

    def test_search_returns_scores(self) -> None:
        """Search results should include scores."""
        from src.retrieval.tfidf_index import TfidfIndex

        index = TfidfIndex()
        index.build_from_documents(["test document about corruption prevention"])
        results = index.search("corruption")
        assert len(results) > 0
        assert isinstance(results[0][1], float)

    def test_empty_index(self) -> None:
        """Should handle search on unloaded index."""
        from src.retrieval.tfidf_index import TfidfIndex

        index = TfidfIndex()
        assert index.search("test") == []
        assert not index.is_loaded

    def test_get_document(self) -> None:
        """Should retrieve document by index."""
        from src.retrieval.tfidf_index import TfidfIndex

        index = TfidfIndex()
        index.build_from_documents(["doc one", "doc two"])
        assert index.get_document(0) == "doc one"
        assert index.get_document(1) == "doc two"
        assert index.get_document(99) == ""  # Out of range


class TestDocumentRepacker:
    """Tests for semantic repacking."""

    def test_repack_small_set(self, sample_documents: list[Document]) -> None:
        """Should handle small document sets."""
        from src.retrieval.repacker import DocumentRepacker

        repacker = DocumentRepacker()

        # With 2 or fewer docs, should return as-is
        result = repacker.repack_by_similarity(sample_documents[:2])
        assert len(result) == 2

    def test_token_limit_repacking(self, sample_documents: list[Document]) -> None:
        """Token-limited repacking should respect budget."""
        from src.retrieval.repacker import DocumentRepacker

        repacker = DocumentRepacker()
        result = repacker.repack_by_token_limit(sample_documents, max_tokens=50)

        # Should return fewer docs than input
        assert len(result) <= len(sample_documents)
        assert len(result) >= 1

    def test_token_limit_empty(self) -> None:
        """Should handle empty document list."""
        from src.retrieval.repacker import DocumentRepacker

        repacker = DocumentRepacker()
        result = repacker.repack_by_token_limit([])
        assert result == []
