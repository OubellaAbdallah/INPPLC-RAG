#!/usr/bin/env python3
"""Generate a grounded benchmark test set from existing ChromaDB chunks using Mistral.

Each test item contains:
- A question in French, Arabic, and English
- A ground truth reference answer extracted ONLY from the source chunk
- The source chunk text itself (for traceability)

Usage:
    python scripts/generate_synthetic_testset.py --count 30 --output data/requests/grounded_requetes.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mistralai import Mistral
from tqdm import tqdm

from src.config.settings import get_settings
from src.retrieval.vectorstore import get_vectorstore


SYSTEM_PROMPT = """\
You are an expert evaluator for Retrieval-Augmented Generation (RAG) systems.

Your task is to create a high-quality test item from a provided text chunk. You must:

1. Write a FACTUAL QUESTION that is **directly and fully answerable** from the chunk.
   - The question must be specific (not vague like "what does this text say?").
   - The question must sound like a natural user inquiry — do NOT reference "the text" or "the document".
   - Generate the question in 3 languages: French (fr), Arabic (ar), English (en).
   - All 3 versions must ask the SAME question, just translated.

2. Write a GROUND TRUTH ANSWER in French that:
   - Answers the question using ONLY facts from the provided chunk.
   - Is concise (2-4 sentences max).
   - Contains NO disclaimers, NO "according to the text", NO recommendations to consult other sources.
   - Uses direct, factual language.

Return ONLY a JSON object with this exact structure:
{
  "fr": "Question en français ?",
  "ar": "السؤال بالعربية ؟",
  "en": "Question in English?",
  "ground_truth": "Réponse factuelle concise en français basée uniquement sur le texte."
}"""


def call_mistral_with_retry(
    client: Mistral,
    model: str,
    system_prompt: str,
    user_content: str,
    max_retries: int = 6,
    base_backoff: float = 3.0,
) -> str | None:
    """Call Mistral API with exponential backoff for rate limits."""
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower() or "Rate limit" in err_str:
                sleep_time = base_backoff * (2 ** attempt)
                tqdm.write(f"  Rate limited. Sleeping {sleep_time:.0f}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
            else:
                tqdm.write(f"  API error: {e}")
                return None
    tqdm.write("  Max retries reached. Skipping.")
    return None


def sample_quality_chunks(collection, count: int) -> list[dict]:
    """Sample high-quality chunks from ChromaDB, filtering out noise."""
    total_count = collection.count()
    if total_count == 0:
        return []

    # Sample more offsets than needed to account for filtering
    sample_size = min(total_count, count * 15)
    offsets = random.sample(range(total_count), sample_size)

    sampled = []
    seen_prefixes = set()

    for offset in offsets:
        if len(sampled) >= count:
            break

        res = collection.get(limit=1, offset=offset)
        if not res or not res["documents"]:
            continue

        text = res["documents"][0].strip()
        meta = res["metadatas"][0] if res["metadatas"] else {}
        doc_id = res["ids"][0]

        # Quality filters
        if len(text) < 300 or len(text) > 1500:
            continue

        # Skip chunks that are mostly numbers/noise
        alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
        if alpha_ratio < 0.4:
            continue

        # Deduplicate by content prefix
        prefix = text[:150]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        sampled.append({
            "id": doc_id,
            "text": text,
            "metadata": meta,
            "source_file": meta.get("source_file", ""),
        })

    return sampled


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate grounded test queries with ground truth answers from ChromaDB chunks."
    )
    parser.add_argument(
        "--count", type=int, default=30,
        help="Number of query items to generate (default: 30)",
    )
    parser.add_argument(
        "--output", type=str, default="data/requests/grounded_requetes.json",
        help="Path to save the generated JSON",
    )
    args = parser.parse_args()

    settings = get_settings()
    if not settings.mistral_api_key:
        print("ERROR: MISTRAL_API_KEY is not set.")
        sys.exit(1)

    print("Connecting to ChromaDB...")
    vectorstore = get_vectorstore(settings)
    collection = vectorstore._collection
    total_count = collection.count()
    print(f"Collection '{settings.collection_name}' has {total_count:,} chunks.")

    if total_count == 0:
        print("ERROR: ChromaDB is empty. Run ingestion first.")
        sys.exit(1)

    client = Mistral(api_key=settings.mistral_api_key)

    print(f"\nSampling {args.count} high-quality chunks...")
    chunks = sample_quality_chunks(collection, args.count)
    print(f"Selected {len(chunks)} chunks after quality filtering.\n")

    benchmark = []

    with tqdm(chunks, desc="Generating grounded queries", unit="chunk") as pbar:
        for idx, chunk in enumerate(pbar):
            pbar.set_postfix_str(f"{len(benchmark)} generated")

            # Rate limiting
            time.sleep(2.0)

            user_content = (
                f"Text Chunk:\n\"\"\"\n{chunk['text']}\n\"\"\"\n\n"
                f"Generate a factual question (in fr, ar, en) and a ground truth answer from this chunk."
            )

            raw = call_mistral_with_retry(client, "mistral-large-latest", SYSTEM_PROMPT, user_content)
            if raw is None:
                continue

            try:
                parsed = json.loads(raw)

                # Validate all required keys exist and are non-empty strings
                required_keys = ["fr", "ar", "en", "ground_truth"]
                if not all(k in parsed and isinstance(parsed[k], str) and parsed[k].strip() for k in required_keys):
                    tqdm.write(f"  Skipping chunk {chunk['id']}: missing or empty keys in response")
                    continue

                item = {
                    "chunk_id": chunk["id"],
                    "difficulte": "facile",
                    "source_file": chunk["source_file"],
                    "source_chunk_text": chunk["text"],
                    "requetes": {
                        "fr": parsed["fr"].strip(),
                        "ar": parsed["ar"].strip(),
                        "en": parsed["en"].strip(),
                    },
                    "ground_truth": parsed["ground_truth"].strip(),
                }
                benchmark.append(item)

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                tqdm.write(f"  Parse error for chunk {chunk['id']}: {e}")
                continue

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Generated {len(benchmark)} grounded test queries with ground truth answers.")
    print(f"Saved to: {output_path}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
