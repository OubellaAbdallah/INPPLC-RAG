"""Re-embed all raw .txt documents into ChromaDB using BAAI/bge-m3.

Reads every *.txt file from data/raw/, chunks them with the updated
DocumentChunker (350 chars / 80 overlap, trilingual separators), and
indexes everything into a FRESH ChromaDB persistent collection.

Usage:
    python scripts/rebuild_embeddings.py
    python scripts/rebuild_embeddings.py --batch-size 32
    python scripts/rebuild_embeddings.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import uuid
from pathlib import Path

# ── Project root on sys.path ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb
from tqdm import tqdm

from src.config.settings import get_settings
from src.ingestion.chunker import DocumentChunker
from src.retrieval.embeddings import EmbeddingModel


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild ChromaDB embeddings from raw .txt files using BAAI/bge-m3"
    )
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Chunks per add() call (default: 64)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Chunk without writing to ChromaDB")
    args = parser.parse_args()

    settings = get_settings()
    raw_dir  = settings.raw_pdf_dir       # data/raw/
    db_dir   = settings.chroma_db_dir     # data/indices/chroma_db/
    col_name = settings.collection_name   # INPPLC

    print("\n" + "=" * 62)
    print("  INPPLC - Embedding Rebuild  (BAAI/bge-m3)")
    print("=" * 62)
    print(f"  Source dir      : {raw_dir}")
    print(f"  Embedding model : {settings.embedding_model_name}")
    print(f"  Chunk size      : {settings.chunk_size} chars")
    print(f"  Chunk overlap   : {settings.chunk_overlap} chars")
    print(f"  Collection      : {col_name}")
    print(f"  ChromaDB dir    : {db_dir}")
    if args.dry_run:
        print("  Mode            : DRY RUN - no writes")
    print("=" * 62 + "\n")

    # ── Step 1: Discover .txt files ───────────────────────────────────────
    txt_files = sorted(raw_dir.glob("*.txt"))
    if not txt_files:
        print(f"ERROR: No .txt files found in {raw_dir}")
        sys.exit(1)
    print(f"Found {len(txt_files)} .txt files\n")

    # ── Step 2: Wipe old ChromaDB ─────────────────────────────────────────
    if not args.dry_run and db_dir.exists():
        print("Removing old ChromaDB collection... ", end="", flush=True)
        shutil.rmtree(db_dir)
        db_dir.mkdir(parents=True, exist_ok=True)
        print("done.\n")

    # ── Step 3: Load + chunk all .txt files ───────────────────────────────
    print("Reading & chunking text files...")
    chunker   = DocumentChunker(settings)
    all_texts : list[str]       = []
    all_metas : list[dict]      = []
    all_ids   : list[str]       = []
    skipped   = 0

    with tqdm(txt_files, desc="  Chunking", unit="file",
              colour="cyan", dynamic_ncols=True) as pbar:
        for txt_path in pbar:
            pbar.set_postfix_str(txt_path.name[:38])
            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError as exc:
                tqdm.write(f"  Skipping {txt_path.name}: {exc}")
                skipped += 1
                continue

            if not text:
                skipped += 1
                continue

            metadata = {
                "source_file" : txt_path.stem,
                "file_name"   : txt_path.name,
                "element_type": "text",
            }
            chunks = chunker.chunk_text(text, metadata)
            for chunk in chunks:
                all_texts.append(chunk.page_content)
                all_metas.append({k: str(v) for k, v in chunk.metadata.items()})
                all_ids.append(str(uuid.uuid4()))

    total = len(all_texts)
    print(f"\nChunking complete -> {total:,} chunks  ({skipped} files skipped)\n")

    if args.dry_run:
        print("Dry run complete. Exiting without writing to ChromaDB.")
        return

    if total == 0:
        print("ERROR: No chunks produced. Check that your .txt files contain text.")
        sys.exit(1)

    # ── Step 4: Load embedding model ──────────────────────────────────────
    print(f"Loading embedding model: {settings.embedding_model_name}")
    print("  (First run downloads the model weights - this may take a few minutes)\n")
    t0 = time.perf_counter()
    embedder = EmbeddingModel(settings)
    _ = embedder.embed_query("init")   # trigger lazy load
    print(f"  Model ready in {time.perf_counter() - t0:.1f}s\n")

    # ── Step 5: Create ChromaDB persistent client & collection ────────────
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(
        name=col_name,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Step 6: Embed + index in batches ──────────────────────────────────
    batch_size = args.batch_size
    n_batches  = (total + batch_size - 1) // batch_size

    print(f"Indexing {total:,} chunks in {n_batches} batches of {batch_size}...\n")
    t_index = time.perf_counter()

    with tqdm(range(0, total, batch_size), total=n_batches,
              desc="  Embedding", unit="batch",
              colour="green", dynamic_ncols=True) as pbar:
        for start in pbar:
            end   = min(start + batch_size, total)
            texts = all_texts[start:end]
            metas = all_metas[start:end]
            ids   = all_ids  [start:end]
            pbar.set_postfix_str(f"{end:,}/{total:,} chunks")

            embeddings = embedder.embed_documents(texts)
            collection.add(
                ids        = ids,
                embeddings = embeddings,
                documents  = texts,
                metadatas  = metas,
            )

    elapsed = time.perf_counter() - t_index

    # ── Step 7: Verify count ──────────────────────────────────────────────
    final_count = collection.count()

    print("\n" + "=" * 62)
    print("  DONE!")
    print("=" * 62)
    print(f"  .txt files read      : {len(txt_files) - skipped}")
    print(f"  Total chunks indexed : {final_count:,}")
    print(f"  Time elapsed         : {elapsed:.0f}s  ({elapsed / 60:.1f} min)")
    print(f"  Throughput           : {total / elapsed:.0f} chunks/sec")
    print(f"  ChromaDB location    : {db_dir}")
    print("=" * 62 + "\n")


if __name__ == "__main__":
    main()
