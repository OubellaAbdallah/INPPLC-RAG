"""Utility script to rebuild the TF-IDF index from documents stored in ChromaDB.

Usage:
    python scripts/rebuild_tfidf.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config.settings import get_settings
from src.retrieval.vectorstore import get_vectorstore


def main() -> None:
    settings = get_settings()
    print("Loading vectorstore...")
    vectorstore = get_vectorstore(settings)
    
    collection = vectorstore._collection
    count = collection.count()
    print(f"Total documents in ChromaDB collection: {count}")
    
    documents: list[Document] = []
    batch_size = 500
    for offset in range(0, count, batch_size):
        results = collection.get(limit=batch_size, offset=offset)
        if results and "documents" in results and results["documents"]:
            docs_list = results["documents"]
            metas_list = results["metadatas"] or [None] * len(docs_list)
            for text, meta in zip(docs_list, metas_list):
                documents.append(Document(page_content=text, metadata=meta or {}))
            
    print(f"Loaded {len(documents)} documents.")
    
    if not documents:
        print("ERROR: No documents found in ChromaDB vector store! Ingestion must be run first.")
        sys.exit(1)
        
    print("Building TF-IDF index...")
    corpus = [doc.page_content for doc in documents]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    index_data = {
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "corpus": corpus,
    }
    
    index_path = settings.tfidf_index_path
    index_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving TF-IDF index to: {index_path}")
    with open(index_path, "wb") as f:
        pickle.dump(index_data, f)
        
    print("TF-IDF index rebuilt successfully!")


if __name__ == "__main__":
    main()
