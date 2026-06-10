"""Shared pytest fixtures for INPPLC tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from src.config.settings import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Create test settings with temporary directories."""
    return Settings(
        ollama_base_url="http://localhost:11434",
        hf_api_token="test-token",
        log_level="DEBUG",
        _env_file=None,
    )


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample documents for testing."""
    return [
        Document(
            page_content="INPPLC is a national institution in Morocco dedicated to probity, prevention, "
            "and combating corruption, serving to promote good governance.",
            metadata={"source_file": "test1.pdf", "element_type": "text"},
        ),
        Document(
            page_content="The services and tasks provided by INPPLC include receiving whistleblowing reports, "
            "assessing integrity in public administrations, and promoting ethics.",
            metadata={"source_file": "test2.pdf", "element_type": "text"},
        ),
        Document(
            page_content="Regular declarations of assets and transparency measures can help detect "
            "conflicts of interest early and improve public trust.",
            metadata={"source_file": "test3.pdf", "element_type": "text"},
        ),
        Document(
            page_content="Anti-corruption education and awareness campaigns are essential components "
            "of the national integrity strategy in Morocco.",
            metadata={"source_file": "test4.pdf", "element_type": "text"},
        ),
        Document(
            page_content="The protection of whistleblowers is a key legislative measure to encourage "
            "reporting of financial irregularities and corruption.",
            metadata={"source_file": "test5.pdf", "element_type": "text"},
        ),
    ]


@pytest.fixture
def mock_llm() -> MagicMock:
    """Create a mock LLM for testing."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(content="Test response")
    return mock
