# Legal Consultation RAG Assistant — INPPLC

> 🇫🇷 *Une version française de ce document est disponible dans [`README.fr.md`](README.fr.md).*

An open-source, pro-bono **Retrieval-Augmented Generation (RAG)** assistant designed for legal and regulatory consultation around Morocco's **National Authority for Probity, Prevention and the Fight against Corruption (INPPLC)**. It helps users navigate anti-corruption legislation, asset declaration requirements, national assessment reports, and public governance frameworks with verifiable source citations.

---

## 🚀 Key Features

### 1. Multimodal Document Ingestion (Mistral OCR + LLM)
- Extracts text from complex official documents (regulatory PDFs and scans) using the Mistral OCR API.
- Applies LLM post-processing to convert dense tables, charts, figures, and diagrams into accurate natural-language descriptions.
- Uses SHA-256 content hashing (idempotency manifest) to skip already-processed documents during re-runs.

### 2. Guardrails & Automatic Query Classification
- Classifies incoming messages into three routing categories before retrieval:
  - **`RAG_REQUIRED`**: Factual legal or institutional query requiring document retrieval.
  - **`DIRECT_RESPONSE`**: Greetings or general conversational queries answered directly without retrieval.
  - **`OUT_OF_SCOPE`**: Requests outside the regulatory/anti-corruption domain, handled with a polite refusal to prevent out-of-domain hallucinations.

### 3. History-Aware Query Rewriting
- Reformulates follow-up questions using prior conversation turns to preserve context across multi-step legal inquiries.

### 4. Hybrid Semantic & Lexical Retrieval (Dense + Sparse)
- Combines dense vector search (`BAAI/bge-large-en-v1.5` embeddings in ChromaDB) with sparse lexical retrieval (TF-IDF & BM25) and re-ranking to maximize recall and precision on legal terminology.

### 5. Real-Time Async Streaming (Server-Sent Events)
- Streams Mistral Large responses token-by-token from FastAPI via SSE to minimize perceived latency.

### 6. Source Traceability & Audit Trail Panel
- Includes a dedicated audit side panel in the web UI showing exact retrieved document excerpts, source filenames, and relevance scores for every answer.

### 7. Session Management & Persistent History
- Persists chat sessions locally (`data/chat_sessions.json`) so users can resume previous consultations.

### 8. Structured Legal Brief Export (PDF)
- Generates downloadable consultation briefs (Facts, Analysis, Recommendations, Legal Sources, and relevance scores) in PDF format via `html2pdf.js`.

---

## 🛠️ Technical Architecture

```text
                           User Message
                                │
                                ▼
              Classification & Guardrail (LLM)
              ├── DIRECT_RESPONSE -> Immediate reply
              ├── OUT_OF_SCOPE    -> Polite refusal
              └── RAG_REQUIRED    -> Continue pipeline
                                │
                                ▼
                History-Aware Query Rewriting
                     (Mistral API)
                                │
                                ▼
                Hybrid Retrieval (Dense + Sparse)
              ├── Dense : ChromaDB (BAAI/bge-large-en-v1.5)
              └── Sparse: TF-IDF / BM25 + Re-ranking
                                │
                                ▼
                     Streamed Generation (SSE)
                       (Mistral Large API)
                                │
                                ▼
               Web UI + Source Audit Trail + PDF Export
```

---

## 📂 Project Structure

```text
INPPLC-RAG/
├── api.py                      # Main FastAPI server (SSE streaming, chat & session endpoints)
├── requirements.txt            # Production Python dependencies
├── pyproject.toml              # Poetry configuration for local development
├── LICENSE                     # MIT License
├── webapp/                     # Frontend Single Page Application
│   ├── index.html              # Application layout
│   ├── style.css               # Styling (green #1B5E35 & gold #C8A84B theme)
│   ├── app.js                  # UI logic, SSE client, PDF export, and session handling
│   └── assets/                 # Static assets
├── src/                        # Core Python package
│   ├── config/                 # Pydantic Settings configuration
│   ├── ingestion/              # OCR extraction, figure/table captioning, and chunking
│   ├── retrieval/              # Dense/sparse indexing, hybrid search, and re-ranking
│   ├── generation/             # Prompt templates, query classification, and LLM calls
│   └── utils/                  # Structured logging and formatting helpers
├── data/                       # Application data directories
│   ├── raw/                    # Source regulatory documents (PDF, TXT)
│   ├── processed/              # Intermediate parsed artifacts
│   └── indices/                # Local vector and lexical indices (ChromaDB, BM25)
└── scripts/                    # CLI administration & evaluation utilities
    ├── run_ingestion.py        # Parse and index raw documents
    ├── cleanup_junk_chunks.py  # Remove invalid or overly short chunks from ChromaDB
    ├── test_rag_terminal.py    # Interactive terminal CLI for testing the pipeline
    └── run_evaluation.py       # Automated Ragas evaluation suite
```

---

## ⚙️ Setup & Installation

### Prerequisites
- **Python 3.11+**
- A valid **Mistral API key**

### 1. Install Dependencies
```bash
# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env` and add your API key:
```bash
cp .env.example .env
```

```env
MISTRAL_API_KEY="your_mistral_api_key"
```

---

## 🚀 Running the Application

### Step 1: Index Documents (Ingestion)
Place regulatory documents in `data/raw/`, then run:
```bash
python -m scripts.run_ingestion
```
To run a faster test pass without generating LLM descriptions for tables and figures:
```bash
python -m scripts.run_ingestion --skip-captioning
```

### Step 2: Start the API Server
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

### Step 3: Open the Web Interface
- **Served directly by FastAPI:** Open `http://localhost:8000/app` in your browser.
- **Standalone static server (optional):**
  ```bash
  cd webapp
  python -m http.server 3000
  ```
  Then visit `http://localhost:3000`.

---

## 🐳 Docker Deployment

Ensure your `.env` file is configured, then build and start the stack:
```bash
docker-compose up --build -d
```
The web interface will be available at `http://localhost:8000/app`.

If local indices already exist in `data/indices/`, the container mounts and uses them immediately. To run or refresh ingestion inside the running container:
```bash
docker exec -it inpplc-chatbot python -m scripts.run_ingestion
```

---

## 🛠️ Developer Commands

- **Run unit and integration tests:**
  ```bash
  python -m pytest
  ```
- **Test the RAG pipeline interactively in the terminal:**
  ```bash
  python -m scripts.test_rag_terminal
  ```
- **Clean up invalid or short chunks in ChromaDB:**
  ```bash
  python -m scripts.cleanup_junk_chunks
  ```
- **Rebuild sparse lexical indices only:**
  ```bash
  python -m scripts.rebuild_tfidf
  ```
- **Rebuild dense semantic embeddings only:**
  ```bash
  python -m scripts.rebuild_embeddings
  ```

---

## 🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome. Feel free to open an issue or submit a pull request—especially around multilingual retrieval improvements (French/Arabic), local/self-hosted model adapters, and test coverage.

---

## 📄 License

This project is licensed under the **MIT License** — see the [`LICENSE`](LICENSE) file for details.
