"""Pure dense vector retrieval for the two-stage RAG pipeline.

Retrieves the top-k most semantically similar chunks from the ChromaDB
vector store using the native collection API. No BM25 fusion, no cross-encoder
reranking — optimised for low latency with BAAI/bge-m3 embeddings.

Includes a minimum content-length safety filter to discard any remaining
junk/noise chunks (single periods, whitespace, OCR artefacts).
"""

from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.config.settings import Settings, get_settings
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.translation import translate_query_to_french
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Minimum character length for a chunk to be considered valid.
# Chunks shorter than this (e.g. ".", whitespace) are noise from
# OCR/chunking artefacts and are discarded at retrieval time.
MIN_CHUNK_CHARS = 20


def dense_retrieve(
    query: str,
    vectorstore: Chroma,
    settings: Settings | None = None,
    k: int | None = None,
) -> list[Document]:
    """Retrieve the top-k chunks via pure dense vector similarity.

    Embeds the query with ``EmbeddingModel`` (BAAI/bge-m3), queries the
    underlying ChromaDB collection directly with ``collection.query()``,
    and post-filters any remaining junk chunks that are too short to be useful.

    Args:
        query: Raw user query string.
        vectorstore: Loaded ChromaDB ``Chroma`` instance to search against.
        settings: Application settings. Uses ``get_settings()`` if not provided.
        k: Number of chunks to return. Defaults to ``settings.retrieval_top_k``
           (currently 10).

    Returns:
        List of :class:`~langchain_core.documents.Document` objects ordered by
        ascending distance (most similar first), each enriched with a
        ``dense_score`` metadata key.

    Raises:
        RuntimeError: If the vectorstore query fails unexpectedly.
    """
    settings = settings or get_settings()
    k = k if k is not None else settings.retrieval_top_k
    embedding_model = EmbeddingModel(settings)

    # Translate query to French for cross-lingual coverage
    translated_query = translate_query_to_french(query, settings)

    try:
        query_vectors = [embedding_model.embed_query(query)]
        if translated_query != query:
            query_vectors.append(embedding_model.embed_query(translated_query))
    except Exception as exc:  # noqa: BLE001
        logger.error("dense_retrieve_embed_failed", error=str(exc), query=query[:80])
        raise RuntimeError(f"Failed to embed query: {exc}") from exc

    # Over-fetch slightly to compensate for any remaining short chunks
    # that pass the MIN_CHUNK_CHARS filter.
    fetch_k = k * 3

    try:
        collection = vectorstore._collection
        results = collection.query(
            query_embeddings=query_vectors,
            n_results=fetch_k,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("dense_retrieve_search_failed", error=str(exc), query=query[:80])
        raise RuntimeError(f"ChromaDB similarity search failed: {exc}") from exc

    # Merge and deduplicate results from both queries (original + French translation)
    documents_map = {}
    skipped_junk = 0

    num_queries = len(query_vectors)
    for q_idx in range(num_queries):
        doc_texts = results["documents"][q_idx] if results["documents"] and len(results["documents"]) > q_idx else []
        metadatas = results["metadatas"][q_idx] if results["metadatas"] and len(results["metadatas"]) > q_idx else []
        distances = results["distances"][q_idx] if results["distances"] and len(results["distances"]) > q_idx else []

        for text, meta, dist in zip(doc_texts, metadatas, distances):
            # Skip junk/noise chunks that are too short to be useful
            if not text or len(text.strip()) < MIN_CHUNK_CHARS:
                skipped_junk += 1
                continue

            meta = meta or {}
            dist_val = round(float(dist), 4)

            # Cosine distance in Chroma: lower is more similar.
            # Deduplicate by content prefix and retain the best score (lowest distance).
            content_key = text.strip()[:200]
            if content_key not in documents_map or dist_val < documents_map[content_key].metadata["dense_score"]:
                meta["dense_score"] = dist_val
                documents_map[content_key] = Document(page_content=text, metadata=meta)

    # Sort documents by distance ascending (lowest distance first) and cap at k
    documents = list(documents_map.values())
    documents.sort(key=lambda d: d.metadata.get("dense_score", 1.0))
    documents = documents[:k]

    if skipped_junk > 0:
        logger.warning(
            "junk_chunks_filtered",
            skipped=skipped_junk,
            query=query[:80],
        )

    logger.info(
        "dense_retrieval_complete",
        query=query[:80],
        k=k,
        fetched=fetch_k * num_queries,
        junk_skipped=skipped_junk,
        returned=len(documents),
        top_score=documents[0].metadata["dense_score"] if documents else None,
    )

    try:
        # Safe printing to avoid Windows cp1252 terminal encoding crashes
        print(f"\n{'='*50}\nRETRIEVED CHUNKS FOR QUERY: '{query}' (translated to French: '{translated_query}')\n{'='*50}")
        if skipped_junk > 0:
            print(f"[!] Filtered {skipped_junk} junk chunks (< {MIN_CHUNK_CHARS} chars)")
        for i, doc in enumerate(documents, 1):
            score = doc.metadata.get("dense_score", "N/A")
            source = doc.metadata.get("source_file", "unknown")
            print(f"\n--- Chunk {i} | Score: {score} | Source: {source} ---")
            try:
                print(doc.page_content.strip())
            except UnicodeEncodeError:
                print(doc.page_content.strip().encode('utf-8', errors='replace').decode('cp1252', errors='replace'))
        print(f"\n{'='*50}\n")
    except Exception as print_exc:
        logger.warning("dense_retrieve_print_failed", error=str(print_exc))

    return documents

