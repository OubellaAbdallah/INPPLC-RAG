"""FastAPI backend for the INPPLC RAG Chatbot.

Exposes the full two-stage RAG pipeline (dense retrieval → Mistral orchestrator →
Mistral API generation) as a REST API consumed by the HTML/CSS/JS frontend.
Static frontend served from /app.

Pipeline flow per request:
  1. dense_retrieve(query, k=5)  →  top-5 semantic chunks from ChromaDB
  2. MistralOrchestrator         →  domain guardrail + structured context (Stage 1)
  3. MistralAPIGenerator         →  final user-facing response via Mistral Cloud (Stage 2)
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config.settings import get_settings
from src.generation.orchestrator import TwoStagePipeline
from src.retrieval.dense_retriever import dense_retrieve
from src.retrieval.vectorstore import get_vectorstore

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="INPPLC RAG API",
    description="Two-stage RAG chatbot for INPPLC information assistant",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the static frontend at /app
app.mount("/app", StaticFiles(directory="webapp", html=True), name="webapp")

# ---------------------------------------------------------------------------
# Persistent session store on disk
# {session_id: [{"role": "user"|"assistant", "content": str, "sources": [...]}]}
# ---------------------------------------------------------------------------

SESSIONS_FILE = "data/chat_sessions.json"


def _load_sessions_from_disk() -> dict[str, list[dict[str, Any]]]:
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to load sessions: {e}")
            return {}
    return {}


def _save_sessions_to_disk() -> None:
    try:
        os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(_sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[ERROR] Failed to save sessions: {e}")


_sessions: dict[str, list[dict[str, Any]]] = _load_sessions_from_disk()

# ---------------------------------------------------------------------------
# Lazy-loaded singletons (loaded on first request to avoid slow startup)
# ---------------------------------------------------------------------------

_settings = None
_vectorstore = None
_pipeline: TwoStagePipeline | None = None


def _get_pipeline() -> tuple[Any, TwoStagePipeline]:
    """Initialise and return the vectorstore + two-stage pipeline singletons."""
    global _settings, _vectorstore, _pipeline

    if _pipeline is None:
        _settings = get_settings()
        _vectorstore = get_vectorstore(_settings)
        _pipeline = TwoStagePipeline(settings=_settings)

    return _vectorstore, _pipeline


# ---------------------------------------------------------------------------
# Query Rewriting — History-Aware & Multilingual Optimizer
# ---------------------------------------------------------------------------

_REWRITE_SYSTEM_PROMPT = """\
You are a query rewriter for a RAG system. Your ONLY job is to rewrite the \
user's latest message into a single, standalone search query optimized for semantic search \
WITHOUT any conversation history.

Rules:
1. Output ONLY the rewritten query, nothing else. No explanation, no quotes.
2. Resolve any coreferences or pronouns (e.g. "who is he?", "tell me more") using the conversation history.
3. Keep the same language as the user's latest message (English -> English, French -> French, Arabic -> Arabic). Do NOT translate.
4. Focus exclusively on search keywords, key entities (names, roles, titles), and concepts. Remove all conversational filler, greetings, and question headers (e.g. "Who is...", "What is...", "Can you tell me...", "Please").
"""


async def rewrite_query(
    user_message: str,
    history: list[dict[str, str]],
    settings: Any,
) -> str:
    """Use Mistral Large to rewrite/optimize the query for RAG search.

    Translates/adapts English queries and resolves references using history.
    If the call fails, returns the original message.
    """
    history_block = "None"
    if history:
        # Build a compact history (last 6 messages max)
        recent = history[-6:]
        history_lines = []
        for m in recent:
            role = "User" if m["role"] == "user" else "Assistant"
            history_lines.append(f"{role}: {m['content'][:300]}")
        history_block = "\n".join(history_lines)

    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CONVERSATION HISTORY:\n{history_block}\n\n"
                f"LATEST USER MESSAGE:\n{user_message}\n\n"
                "Rewritten standalone query:"
            ),
        },
    ]

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                json={
                    "model": settings.orchestrator_model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 200,
                },
                headers={
                    "Authorization": f"Bearer {settings.mistral_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=15.0,
            )
            resp.raise_for_status()
            rewritten = resp.json()["choices"][0]["message"]["content"].strip()
            if rewritten:
                try:
                    print(f"\n[QUERY REWRITE] '{user_message}' -> '{rewritten}'\n")
                except UnicodeEncodeError:
                    pass
                return rewritten
    except Exception as exc:
        try:
            print(f"\n[WARNING] Query rewrite failed (using original): {exc}\n")
        except UnicodeEncodeError:
            pass

    return user_message


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str
    message: str


class NewSessionResponse(BaseModel):
    session_id: str


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: list[dict[str, Any]] | None = None


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[ChatMessage]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/session", response_model=NewSessionResponse)
def create_session() -> dict[str, str]:
    """Create a new chat session and return its ID."""
    sid = str(uuid.uuid4())
    _sessions[sid] = []
    _save_sessions_to_disk()
    return {"session_id": sid}


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str) -> dict[str, str]:
    """Clear and delete a chat session."""
    _sessions.pop(session_id, None)
    _save_sessions_to_disk()
    return {"status": "deleted", "session_id": session_id}


@app.get("/api/sessions")
def list_sessions() -> dict[str, Any]:
    """Return all active session IDs with their first message (for sidebar)."""
    result = []
    for sid, msgs in _sessions.items():
        first_user = next((m["content"] for m in msgs if m["role"] == "user"), None)
        result.append({"session_id": sid, "preview": first_user or "New Chat"})
    return {"sessions": result}


@app.get("/api/session/{session_id}/history", response_model=HistoryResponse)
def get_history(session_id: str) -> dict[str, Any]:
    """Return the full message history for a session."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "messages": _sessions[session_id]}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Send a message and stream the assistant response back as SSE.

    Pipeline:
        query_rewrite → dense_retrieve (k=10) → Mistral Large (streaming)
    """
    if req.session_id not in _sessions:
        _sessions[req.session_id] = []

    history = _sessions[req.session_id]

    # Build LangChain message history for context
    from langchain_core.messages import AIMessage, HumanMessage
    lc_history: list[HumanMessage | AIMessage] = []
    for m in history:
        if m["role"] == "user":
            lc_history.append(HumanMessage(content=m["content"]))
        else:
            lc_history.append(AIMessage(content=m["content"]))

    # Persist user message immediately
    history.append({"role": "user", "content": req.message})
    _save_sessions_to_disk()

    async def event_stream():
        vectorstore, pipeline = _get_pipeline()

        # --- Step 0: History-Aware Query Rewriting ---
        search_query = await rewrite_query(req.message, history[:-1], _settings)

        # --- Step 1: Dense retrieval using the rewritten query ---
        try:
            docs = await asyncio.to_thread(
                dense_retrieve,
                search_query,
                vectorstore,
                _settings,
            )
        except RuntimeError as exc:
            logger.error("retrieval_error", error=str(exc))
            error_msg = f"⚠️ [Une erreur est survenue lors de la récupération des documents. Veuillez réessayer. / حدث خطأ أثناء استرداد المستندات.]"
            yield f"data: {json.dumps({'token': error_msg})}\n\n"
            yield "data: [DONE]\n\n"
            return

        # Prepare and yield sources event
        serialized_sources = []
        for d in docs:
            serialized_sources.append({
                "source_file": d.metadata.get("source_file", "unknown"),
                "content": d.page_content,
                "score": d.metadata.get("dense_score", 0.0),
            })
        
        yield f"data: {json.dumps({'sources': serialized_sources})}\n\n"

        full_response = ""
        try:
            # --- Step 2: LLM generation with streaming ---
            async for token in pipeline.run_stream(
                query=req.message,  # original user message for the prompt
                documents=docs,
                conversation_history=lc_history,
            ):
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"
        except RuntimeError as exc:
            logger.error("generation_error", error=str(exc))
            error_msg = f"\n\n⚠️ [Erreur système lors de la génération de la réponse. / خطأ في النظام أثناء إنشاء الاستجابة.]"
            full_response += error_msg
            yield f"data: {json.dumps({'token': error_msg})}\n\n"
        finally:
            history.append({
                "role": "assistant",
                "content": full_response,
                "sources": serialized_sources
            })
            _save_sessions_to_disk()

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
