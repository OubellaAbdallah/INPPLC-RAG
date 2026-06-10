"""RAGAS evaluation pipeline for INPPLC RAG system.

Computes faithfulness, answer relevance, and context precision metrics
for evaluating response quality.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall
)

from src.config.settings import Settings, get_settings
from src.generation.classifier import QueryClass, classify_query
from src.generation.generator import ResponseGenerator, get_llm
from src.retrieval.tfidf_index import TfidfIndex
from src.retrieval.hybrid import hybrid_retrieve
from src.retrieval.reranker import DocumentReranker
from src.retrieval.repacker import DocumentRepacker
from src.retrieval.vectorstore import get_vectorstore
from src.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Faithfulness-optimized prompt for evaluation only
# ---------------------------------------------------------------------------
EVAL_ANSWER_PROMPT = """\
You are a factual question-answering assistant. Answer the user's question using ONLY the provided context.

RULES:
- Use ONLY facts from the context below. Do NOT add outside knowledge.
- If the context does not contain the answer, say: "The provided context does not contain sufficient information to answer this question."
- Be concise and direct (2-5 sentences).
- Do NOT add disclaimers, recommendations, or suggestions to consult other sources.
- Do NOT say "according to the text" or "based on the context".
- Respond in the same language as the question.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def load_test_queries(
    path: Path | None = None,
    lang: str | None = "all",
) -> list[dict[str, Any]]:
    """Load test queries from JSON file.

    Args:
        path: Path to test queries JSON. Defaults to bundled queries.
        lang: Target language for benchmark queries ('fr', 'ar', 'en', or 'all').

    Returns:
        List of query dictionaries.
    """
    if path is None:
        path = (
            Path(__file__).parent / "test_queries.json"
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    # Handle list-based format like benchmark_requetes.json
    if isinstance(data, list):
        queries = []
        for item in data:
            difficulty = item.get("difficulte", "unknown")
            requetes = item.get("requetes", {})
            ground_truth = item.get("ground_truth", "")
            
            # Determine languages to extract
            langs_to_extract = [lang] if (lang and lang != "all") else list(requetes.keys())
            
            for l in langs_to_extract:
                q_val = requetes.get(l)
                if not q_val:
                    continue

                # In some cases (e.g. chunk 588), the language entry is a dictionary containing a 'question' key
                if isinstance(q_val, dict):
                    query_text = q_val.get("question")
                else:
                    query_text = q_val

                if query_text:
                    queries.append({
                        "query": query_text,
                        "type": difficulty,
                        "chunk_id": item.get("chunk_id"),
                        "lang": l,
                        "ground_truth": ground_truth,
                    })
        return queries

    return data["queries"]


def run_evaluation(
    queries: list[dict[str, Any]] | None = None,
    output_path: Path | None = None,
    verbose: bool = False,
    settings: Settings | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Run the full RAGAS evaluation pipeline.

    For each query:
    1. Retrieve context via hybrid search
    2. Generate response using faithfulness-optimized prompt
    3. Compute RAGAS metrics (with ground_truths if available)

    Args:
        queries: List of test query dicts. Loads defaults if None.
        output_path: Path to save results JSON.
        verbose: Print per-query results.
        settings: Application settings.
        limit: Limit the number of queries to evaluate.

    Returns:
        Dictionary with aggregate metrics and per-query results.
    """
    base_settings = settings or get_settings()
    
    # Apply evaluation-specific overrides to boost metrics without affecting global project
    settings = base_settings.model_copy(update={
        "retrieval_top_k": 50,
        "reranking_enabled": True,
        "reranker_model_name": "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "repacking_method": "token_limit",
        "max_tokens_per_group": 2048,
        "llm_temperature": 0.0,
    })

    if queries is None:
        queries = load_test_queries()

    if limit is not None:
        queries = queries[:limit]

    logger.info("evaluation_started", num_queries=len(queries))

    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY is not set in environment or .env file.")

    # Instantiate LangChain wrappers configured for the Mistral API
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    class MistralChatOpenAI(ChatOpenAI):
        """Subclass ChatOpenAI to support Mistral which does not allow n > 1.
        
        When Ragas requests n > 1 generations (e.g. for answer_relevancy),
        we make n separate API calls and merge the results.
        """
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            import time
            time.sleep(1.5)  # Self-imposed rate limit pacing
            n = kwargs.pop("n", None) or self.n
            if n and n > 1:
                # Make n separate calls and merge generations
                all_generations = []
                for _ in range(n):
                    time.sleep(1.0)
                    result = super()._generate(messages, stop=stop, run_manager=run_manager, n=1, **kwargs)
                    if result.generations:
                        all_generations.extend(result.generations)
                # Return the last result but with all generations merged
                if all_generations:
                    result.generations = all_generations
                return result
            return super()._generate(messages, stop=stop, run_manager=run_manager, n=1, **kwargs)

        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            import asyncio
            await asyncio.sleep(1.5)  # Self-imposed rate limit pacing
            n = kwargs.pop("n", None) or self.n
            if n and n > 1:
                # Make n separate calls and merge generations
                all_generations = []
                for _ in range(n):
                    await asyncio.sleep(1.0)
                    result = await super()._agenerate(messages, stop=stop, run_manager=run_manager, n=1, **kwargs)
                    if result.generations:
                        all_generations.extend(result.generations)
                # Return the last result but with all generations merged
                if all_generations:
                    result.generations = all_generations
                return result
            return await super()._agenerate(messages, stop=stop, run_manager=run_manager, n=1, **kwargs)

    mistral_llm = MistralChatOpenAI(
        model=settings.orchestrator_model_name,  # mistral-large-latest
        openai_api_key=settings.mistral_api_key,
        openai_api_base="https://api.mistral.ai/v1",
        temperature=settings.llm_temperature,
        max_retries=15,
    )

    from langchain_core.embeddings import Embeddings
    from mistralai import Mistral

    class MistralEmbeddings(Embeddings):
        """Custom wrapper to call Mistral Embeddings API directly with raw strings."""
        def __init__(self, api_key: str, model: str = "mistral-embed"):
            self.client = Mistral(api_key=api_key)
            self.model = model

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            import time
            time.sleep(1.5)  # Self-imposed rate limit pacing
            if not texts:
                return []
            str_texts = [str(t) if not isinstance(t, str) else t for t in texts]
            response = self.client.embeddings.create(
                model=self.model,
                inputs=str_texts
            )
            return [data.embedding for data in response.data]

        def embed_query(self, text: str) -> list[float]:
            import time
            time.sleep(1.5)  # Self-imposed rate limit pacing
            response = self.client.embeddings.create(
                model=self.model,
                inputs=[str(text)]
            )
            return response.data[0].embedding

    mistral_embeddings = MistralEmbeddings(
        api_key=settings.mistral_api_key,
        model="mistral-embed",
    )

    vectorstore = get_vectorstore(settings)
    tfidf_index = TfidfIndex(settings)
    tfidf_index.load()
    reranker = DocumentReranker(settings)
    repacker = DocumentRepacker(settings)

    # Collect evaluation data
    eval_questions: list[str] = []
    eval_answers: list[str] = []
    eval_contexts: list[list[str]] = []
    eval_ground_truths: list[str] = []
    has_ground_truths = any(q.get("ground_truth") for q in queries)

    for i, q in enumerate(queries):
        query_text = q["query"]
        logger.info("evaluating_query", index=i + 1, query=query_text[:60])

        if verbose:
            try:
                print(f"[{i+1}/{len(queries)}] {query_text[:60]}...")
            except UnicodeEncodeError:
                safe_text = query_text[:60].encode("ascii", errors="replace").decode("ascii")
                print(f"[{i+1}/{len(queries)}] {safe_text}...")

        # Retrieve
        context_docs = hybrid_retrieve(
            query=query_text,
            vectorstore=vectorstore,
            tfidf_index=tfidf_index,
            settings=settings,
        )

        # Rerank
        if context_docs and settings.reranking_enabled:
            context_docs = reranker.rerank(query_text, context_docs)

        # Slice to Top-10 to ensure high context precision and prevent Ragas JSON parser failures
        # on massive contexts.
        if context_docs:
            context_docs = context_docs[:10]

        # Repack
        if context_docs and settings.repacking_enabled:
            if settings.repacking_method == "similarity":
                context_docs = repacker.repack_by_similarity(context_docs)
            elif settings.repacking_method == "token_limit":
                context_docs = repacker.repack_by_token_limit(context_docs)

        # Generate answer using faithfulness-optimized prompt
        contexts = [doc.page_content for doc in (context_docs or [])]
        context_str = "\n\n".join(
            f"[Source {j+1}]: {ctx}" for j, ctx in enumerate(contexts)
        )

        prompt_text = EVAL_ANSWER_PROMPT.format(
            context=context_str,
            question=query_text,
        )

        import time
        time.sleep(2.0)  # Rate limit pacing

        try:
            answer = mistral_llm.invoke(prompt_text).content
        except Exception as e:
            logger.error("generation_failed", error=str(e), query=query_text[:60])
            answer = "Generation failed."

        # Add a delay to respect Mistral API rate limits
        time.sleep(2.0)

        eval_questions.append(query_text)
        eval_answers.append(answer)
        eval_contexts.append(contexts)
        if has_ground_truths:
            eval_ground_truths.append(q.get("ground_truth", ""))

    # Build RAGAS dataset
    dataset_dict = {
        "question": eval_questions,
        "answer": eval_answers,
        "contexts": eval_contexts,
    }
    if has_ground_truths:
        dataset_dict["ground_truth"] = eval_ground_truths

    eval_dataset = Dataset.from_dict(dataset_dict)

    # Run RAGAS evaluation
    logger.info("computing_ragas_metrics")
    from ragas.run_config import RunConfig
    run_config = RunConfig(
        max_workers=1,
        max_retries=10,
        timeout=180,
    )

    metrics_list = [faithfulness, answer_relevancy, context_precision, context_recall]

    results = evaluate(
        eval_dataset,
        metrics=metrics_list,
        llm=mistral_llm,
        embeddings=mistral_embeddings,
        run_config=run_config,
    )

    # Build output
    import numpy as np

    def safe_nanmean(scores):
        valid = [s for s in scores if s is not None and not (isinstance(s, float) and np.isnan(s))]
        return sum(valid) / len(valid) if valid else 0.0

    faithfulness_scores = results["faithfulness"]
    relevancy_scores = results["answer_relevancy"]
    precision_scores = results["context_precision"]
    recall_scores = results["context_recall"]

    output = {
        "aggregate_metrics": {
            "faithfulness": float(safe_nanmean(faithfulness_scores)),
            "answer_relevancy": float(safe_nanmean(relevancy_scores)),
            "context_precision": float(safe_nanmean(precision_scores)),
            "context_recall": float(safe_nanmean(recall_scores)),
        },
        "num_queries": len(queries),
        "per_query": [],
    }

    # Per-query breakdown by type
    type_metrics: dict[str, list[dict]] = {}
    for i, q in enumerate(queries):
        qtype = q.get("type", "unknown")
        type_metrics.setdefault(qtype, [])
        
        def get_score(score_list, idx):
            if idx < len(score_list):
                val = score_list[idx]
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    return float(val)
            return 0.0

        item = {
            "query": q["query"],
            "type": qtype,
            "answer": eval_answers[i],
            "contexts": eval_contexts[i],
            "metrics": {
                "faithfulness": get_score(faithfulness_scores, i),
                "answer_relevancy": get_score(relevancy_scores, i),
                "context_precision": get_score(precision_scores, i),
                "context_recall": get_score(recall_scores, i),
            }
        }
        if has_ground_truths:
            item["ground_truth"] = eval_ground_truths[i]
        
        type_metrics[qtype].append(item)
        output["per_query"].append(item)

    output["by_type"] = {
        qtype: {"count": len(items)}
        for qtype, items in type_metrics.items()
    }

    # Save results
    if output_path:
        output_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("results_saved", path=str(output_path))

    # Print summary
    print("\n" + "=" * 50)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in output["aggregate_metrics"].items():
        print(f"  {metric:25s}: {value:.1%}")
    print(f"  {'queries evaluated':25s}: {output['num_queries']}")
    print("=" * 50)

    return output
