"""Hybrid retrieval combining dense and sparse search with score fusion.

Implements the paper's Eq. 1: S_final = alpha * S_dense + (1-alpha) * S_sparse
with min-max normalization and configurable alpha (default 0.7, Table II).

Pipeline: Query -> Dense(top-k) + BM25(top-k) -> Score Fusion -> Return fused results
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config.settings import Settings, get_settings
from src.retrieval.tfidf_index import TfidfIndex
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.translation import translate_query_to_french
from src.utils.logging import get_logger

logger = get_logger(__name__)


def hybrid_retrieve(
    query: str,
    vectorstore: Chroma,
    tfidf_index: TfidfIndex,
    settings: Settings | None = None,
    k: int | None = None,
) -> list[Document]:
    """Perform hybrid dense+sparse retrieval with alpha-weighted score fusion.

    Implements Eq. 1 from the paper:
        S_final = alpha * S_dense_norm + (1 - alpha) * S_sparse_norm

    Both dense and sparse scores are min-max normalized to [0,1] before fusion.

    Args:
        query: Search query string.
        vectorstore: ChromaDB vector store for dense retrieval.
        tfidf_index: TF-IDF index for sparse retrieval.
        settings: Application settings. Uses defaults if not provided.
        k: Number of results to return. Defaults to settings.retrieval_top_k.

    Returns:
        List of Document objects sorted by fused score descending,
        with 'fused_score', 'dense_score', and 'sparse_score' in metadata.
    """
    settings = settings or get_settings()
    k = k or settings.retrieval_top_k
    alpha = settings.hybrid_alpha

    # Retrieve candidates from both indexes (fetch more than k for fusion)
    fetch_k = k * 5

    embedding_model = EmbeddingModel(settings)
    translated_query = translate_query_to_french(query, settings)

    # --- Dense retrieval with scores ---
    query_embedding = embedding_model.embed_query(query)
    dense_results: list[tuple[Document, float]] = (
        vectorstore.similarity_search_by_vector_with_relevance_scores(
            query_embedding, k=fetch_k
        )
    )
    if translated_query != query:
        translated_embedding = embedding_model.embed_query(translated_query)
        translated_dense = vectorstore.similarity_search_by_vector_with_relevance_scores(
            translated_embedding, k=fetch_k
        )
        # Merge dense results, keeping the highest similarity score for duplicates
        dense_map = {doc.page_content[:200]: (doc, score) for doc, score in dense_results}
        for doc, score in translated_dense:
            key = doc.page_content[:200]
            if key not in dense_map or score > dense_map[key][1]:
                dense_map[key] = (doc, score)
        dense_results = list(dense_map.values())

    # --- Sparse retrieval with scores ---
    sparse_results: list[tuple[Document, float]] = []
    if tfidf_index.is_loaded:
        sparse_results = tfidf_index.search_documents(query, k=fetch_k)
        if translated_query != query:
            translated_sparse = tfidf_index.search_documents(translated_query, k=fetch_k)
            # Merge sparse results, keeping the highest score for duplicates
            sparse_map = {doc.page_content[:200]: (doc, score) for doc, score in sparse_results}
            for doc, score in translated_sparse:
                key = doc.page_content[:200]
                if key not in sparse_map or score > sparse_map[key][1]:
                    sparse_map[key] = (doc, score)
            sparse_results = list(sparse_map.values())

    if not dense_results and not sparse_results:
        logger.warning("no_retrieval_results", query=query[:100])
        return []

    # --- Build score maps keyed by document content ---
    dense_scores: dict[str, float] = {}
    dense_docs: dict[str, Document] = {}
    for doc, score in dense_results:
        content_key = doc.page_content[:200]  # Use prefix as key
        dense_scores[content_key] = score
        dense_docs[content_key] = doc

    sparse_scores: dict[str, float] = {}
    sparse_docs: dict[str, Document] = {}
    for doc, score in sparse_results:
        content_key = doc.page_content[:200]
        sparse_scores[content_key] = score
        sparse_docs[content_key] = doc

    # --- Collect all unique document keys ---
    all_keys = set(dense_scores.keys()) | set(sparse_scores.keys())

    # --- Compute wRRF Scores ---
    wrrf_k = settings.wrrf_k

    # Calculate 1-based ranks
    dense_ranks = {doc.page_content[:200]: rank for rank, (doc, _) in enumerate(dense_results, 1)}
    sparse_ranks = {doc.page_content[:200]: rank for rank, (doc, _) in enumerate(sparse_results, 1)}

    fused_results: list[tuple[str, float, float, float]] = []
    for key in all_keys:
        d_rank = dense_ranks.get(key, float('inf'))
        s_rank = sparse_ranks.get(key, float('inf'))

        d_wrrf = alpha * (1.0 / (wrrf_k + d_rank)) if d_rank != float('inf') else 0.0
        s_wrrf = (1 - alpha) * (1.0 / (wrrf_k + s_rank)) if s_rank != float('inf') else 0.0
        fused_score = d_wrrf + s_wrrf

        # Keep original raw scores for metadata
        raw_d_score = dense_scores.get(key, 0.0)
        raw_s_score = sparse_scores.get(key, 0.0)

        fused_results.append((key, fused_score, raw_d_score, raw_s_score))

    # --- Sort by fused score descending ---
    fused_results.sort(key=lambda x: x[1], reverse=True)

    # --- Build output documents with scores in metadata ---
    output: list[Document] = []
    for content_key, fused_score, d_score, s_score in fused_results[:k]:
        doc = dense_docs.get(content_key) or sparse_docs.get(content_key)
        if doc is None:
            continue

        doc.metadata["fused_score"] = round(fused_score, 4)
        doc.metadata["dense_score"] = round(d_score, 4)
        doc.metadata["sparse_score"] = round(s_score, 4)
        output.append(doc)

    logger.info(
        "hybrid_retrieval_complete",
        query=query[:80],
        dense_candidates=len(dense_results),
        sparse_candidates=len(sparse_results),
        fused_results=len(output),
        alpha=alpha,
    )

    return output
