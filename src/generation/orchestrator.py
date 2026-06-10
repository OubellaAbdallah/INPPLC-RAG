"""Single-stage LLM generation pipeline for the INPPLC RAG system.

Temporarily bypasses the local fine-tuned mistral-ft model on ngrok.
Directly uses Mistral Large (via Mistral Cloud API) to generate the final answer
based on the retrieved chunks and user query.
"""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from src.config.settings import Settings, get_settings
from src.generation.prompts import ANSWER_PROMPT_TEMPLATE
from src.utils.helpers import format_chat_history_for_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)


_MISTRAL_API_BASE = "https://api.mistral.ai/v1"

class TwoStagePipeline:
    """Temporarily repurposed as a Single-Stage Pipeline using Mistral Large directly.
    
    Kept the class name 'TwoStagePipeline' so we don't break imports in api.py.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def run(
        self,
        query: str,
        documents: list[Document],
        conversation_history: list[HumanMessage | AIMessage] | None = None,
    ) -> str:
        messages = self._build_messages(query, documents, conversation_history)
        payload = self._build_payload(messages, stream=False)

        async with httpx.AsyncClient() as client:
            try:
                resp = await client.post(
                    f"{_MISTRAL_API_BASE}/chat/completions",
                    json=payload,
                    headers=self._auth_headers(),
                    timeout=60.0,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                logger.error("mistral_cloud_api_error", error=str(exc))
                raise RuntimeError(f"Error calling Mistral Cloud API: {exc!r}") from exc

    async def run_stream(
        self,
        query: str,
        documents: list[Document],
        conversation_history: list[HumanMessage | AIMessage] | None = None,
    ) -> AsyncGenerator[str, None]:
        messages = self._build_messages(query, documents, conversation_history)
        payload = self._build_payload(messages, stream=True)

        async with httpx.AsyncClient() as client:
            try:
                async with client.stream(
                    "POST",
                    f"{_MISTRAL_API_BASE}/chat/completions",
                    json=payload,
                    headers=self._auth_headers(),
                    timeout=90.0,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data_str = line[len("data:"):].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            token = data["choices"][0].get("delta", {}).get("content", "")
                            if token:
                                yield token
                        except (KeyError, ValueError):
                            continue
            except Exception as exc:
                logger.error("mistral_cloud_stream_error", error=str(exc))
                raise RuntimeError(f"Error during Mistral Cloud stream: {exc!r}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._settings.mistral_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_messages(
        self,
        query: str,
        chunks: list[Document],
        conversation_history: list[HumanMessage | AIMessage] | None,
    ) -> list[dict[str, str]]:
        
        history_str = ""
        if conversation_history:
            history_str = format_chat_history_for_prompt(conversation_history, max_turns=6)

        chunk_lines = []
        for i, doc in enumerate(chunks, 1):
            source = doc.metadata.get("source_file", "unknown")
            chunk_lines.append(f"[Chunk {i} — {source}]\n{doc.page_content}")
        chunks_block = "\n\n".join(chunk_lines) if chunk_lines else "No relevant information found."

        # Format the ANSWER_PROMPT_TEMPLATE with the arguments it expects
        prompt_content = ANSWER_PROMPT_TEMPLATE.format(
            context=chunks_block,
            chat_history=history_str,
            question=query
        )
        
        return [{"role": "user", "content": prompt_content}]

    def _build_payload(self, messages: list[dict[str, str]], stream: bool) -> dict[str, Any]:
        return {
            "model": self._settings.orchestrator_model_name, # uses mistral-large-latest
            "messages": messages,
            "temperature": self._settings.llm_temperature,
            "stream": stream,
        }
