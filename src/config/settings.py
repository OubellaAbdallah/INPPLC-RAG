"""Configuration settings for the INPPLC RAG system.

Uses Pydantic BaseSettings for environment variable loading and validation.
All system parameters are centralized here.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file.

    Attributes:
        ollama_base_url: Ollama API base URL for the local fine-tuned model (Stage 1
            orchestrator and legacy generation).
        hf_api_token: HuggingFace API token for model downloads.
        mistral_api_key: Mistral Cloud API key (Stage 2 final generator).
        embedding_model_name: Dense embedding model identifier (BAAI/bge-m3).
        llm_model_name: Local Ollama model name served at ollama_base_url.
        orchestrator_model_name: Ollama model used as the Stage 1 Mistral orchestrator.
        stage2_model_name: Mistral Cloud model name for the Stage 2 final generator.
        llm_temperature: Temperature for LLM generation.
        reranker_model_name: Cross-encoder model for reranking (disabled by default).
        chunk_size: Character count per text chunk (350).
        chunk_overlap: Character overlap between chunks (80).
        hybrid_alpha: Dense/sparse fusion weight (legacy hybrid retrieval).
        bm25_k1: BM25 term frequency saturation.
        bm25_b: BM25 document length normalization.
        retrieval_top_k: Number of dense chunks to retrieve per query (k=5).
        reranking_enabled: Whether to apply cross-encoder reranking (disabled).
        repacking_enabled: Whether to apply semantic repacking.
        repacking_method: Repacking strategy ('similarity' or 'token_limit').
        max_tokens_per_group: Max tokens per repacked document group.
        collection_name: ChromaDB collection name.
        log_level: Logging verbosity level.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- API Configuration ---
    ollama_base_url: str = Field(
        default="https://unfinished-apical-keaton.ngrok-free.dev",
        description="Ollama API base URL"
    )
    hf_api_token: str = Field(default=os.getenv("HF_TOKEN"), description="HuggingFace API token")
    mistral_api_key: str = Field(
        default=os.getenv("MISTRAL_API_KEY"),
        description="Mistral API key for OCR and chat endpoints",
    )

    # --- Model Identifiers ---
    embedding_model_name: str = Field(
        default="BAAI/bge-m3",
        description="Dense embedding model for multilingual retrieval",
    )
    llm_model_name: str = Field(
        default="mistral-ft",
        description="Legacy Ollama model name (still used by ResponseGenerator)",
    )
    orchestrator_model_name: str = Field(
        default="mistral-large-latest",
        description=(
            "Stage 1 orchestrator: Mistral Cloud API via mistral_api_key. "
            "Acts as strict domain guardrail and context builder."
        ),
    )
    stage2_model_name: str = Field(
        default="mistral-ft",
        description=(
            "Stage 2 generator: Local fine-tuned model called via ollama_base_url "
            "to produce the final end-user response from the orchestrated context."
        ),
    )
    llm_temperature: float = Field(
        default=0.1,
        description="LLM generation temperature",
    )
    reranker_model_name: str = Field(
        default="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        description="Cross-encoder for reranking (bypassed when reranking_enabled=False)",
    )

    # --- Chunking Parameters ---
    chunk_size: int = Field(
        default=350,
        description="Token count per text chunk",
    )
    chunk_overlap: int = Field(
        default=80,
        description="Token overlap between adjacent chunks",
    )

    # --- Hybrid Retrieval ---
    hybrid_alpha: float = Field(
        default=0.6,
        description="Dense/sparse fusion weight (alpha)",
    )
    wrrf_k: int = Field(
        default=60,
        description="wRRF constant k parameter",
    )
    retrieval_top_k: int = Field(
        default=10,
        description="Number of dense chunks to retrieve per query (pure dense, k=10)",
    )
    retrieval_min_score: float = Field(
        default=0.6,
        description="Minimum cosine similarity score to retain a chunk",
    )
    reranking_enabled: bool = Field(
        default=False,
        description="Enable cross-encoder reranking (disabled for latency optimisation)",
    )
    repacking_enabled: bool = Field(
        default=True,
        description="Enable semantic repacking for diversity",
    )
    repacking_method: str = Field(
        default="similarity",
        description="Repacking strategy: 'similarity' or 'token_limit'",
    )
    max_tokens_per_group: int = Field(
        default=2048,
        description="Max tokens per repacked document group",
    )

    # --- Storage ---
    collection_name: str = Field(
        default="inpplc_documents",
        description="ChromaDB collection name",
    )

    # --- Logging ---
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )

    @property
    def base_dir(self) -> Path:
        """Project root directory."""
        return Path(__file__).resolve().parent.parent.parent

    @property
    def data_dir(self) -> Path:
        """Data directory containing corpus and indices."""
        return self.base_dir / "data"

    @property
    def raw_pdf_dir(self) -> Path:
        """Directory containing raw downloaded PDFs."""
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        """Directory for intermediate processed outputs."""
        return self.data_dir / "processed"

    @property
    def indices_dir(self) -> Path:
        """Directory for vector store and BM25 indices."""
        return self.data_dir / "indices"

    @property
    def chroma_db_dir(self) -> Path:
        """ChromaDB persistence directory."""
        return self.indices_dir / "chroma_db"

    @property
    def tfidf_index_path(self) -> Path:
        """TF-IDF serialized corpus path."""
        return self.indices_dir / "tfidf_corpus.pkl"

    @property
    def manifest_path(self) -> Path:
        """Ingestion manifest for idempotent processing."""
        return self.processed_dir / "manifest.json"

    @property
    def assets_dir(self) -> Path:
        """Static assets directory for the Streamlit app."""
        return self.base_dir / "src" / "app" / "assets"

    def ensure_directories(self) -> None:
        """Create all required data directories if they don't exist."""
        for directory in [
            self.raw_pdf_dir,
            self.processed_dir,
            self.indices_dir,
            self.chroma_db_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached singleton Settings instance.

    Returns:
        The application Settings loaded from environment.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
