# Guide de Démarrage et d'Exécution / Running Guide

This guide walks you through setting up and running the INPPLC RAG system from scratch. 

---

## 📋 Table of Contents
1. [Prerequisites](#-prerequisites)
2. [Installation & Setup](#-1-installation--setup)
3. [Ingestion & Indexing](#-2-ingestion--indexing-data-preparation)
4. [Running the Application](#-3-running-the-application)
   - [Option A: Native Local Execution](#option-a-native-local-execution)
   - [Option B: Containerized Execution (Docker)](#option-b-containerized-execution-docker)
5. [Development & Evaluation Utilities](#-4-development--evaluation-utilities)
6. [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🛠️ Prerequisites

Before starting, ensure you have the following installed and configured:

### 1. System Dependencies
* **Python 3.11** or **3.12**
* **Git**
* **Docker & Docker Compose** (optional, for containerized run)
* **Windows-specific requirements** (for processing PDFs locally without Docker):
  * **Poppler-utils**: Required by `pdf2image` / `unstructured`. Download the Windows binaries, extract them, and add the `bin` directory to your System PATH.
  * **Tesseract OCR**: Required for OCR. Download the Windows installer (e.g., from UB Mannheim), install it, and add the installation folder (usually `C:\Program Files\Tesseract-OCR`) to your System PATH.

### 2. Large Language Models (LLM) & API Keys
* **Mistral API Key**: Required for OCR ingestion, the Stage 1 Orchestrator, and the Stage 2 final generator. Get one at [console.mistral.ai](https://console.mistral.ai/).
* **Ollama**: Required to run the local fine-tuned model (`mistral-ft`) for Stage 2.
  * Install Ollama from [ollama.com](https://ollama.com).
  * Ensure the fine-tuned model `mistral-ft` is pulled and available locally:
    ```bash
    ollama list
    ```
* **HuggingFace API Token (Optional/Recommended)**: Required for downloading the local dense embedding model (`BAAI/bge-m3`) and the optional cross-encoder reranker.

---

## 🚀 1. Installation & Setup

Follow these steps to set up your local development environment:

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Inpplc-RAG
```

### Step 2: Create and Activate Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .venv\Scripts\Activate.ps1
  ```
* **macOS/Linux:**
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Python Dependencies
Using `pip` (with `requirements.txt`):
```bash
pip install -r requirements.txt
```
*Alternatively, if you are using Poetry:*
```bash
poetry install
```

### Step 4: Configure Environment Variables
Copy the template environment file:
```bash
cp .env.example .env
```
Open the `.env` file and fill in your keys and configuration:
```env
# Ollama Configuration (Default is localhost, or change to custom host/ngrok)
OLLAMA_BASE_URL=http://localhost:11434

# Mistral AI Authentication
MISTRAL_API_KEY="your-mistral-api-key-here"

# HuggingFace Token (For downloading models)
HF_TOKEN="your-huggingface-token-here"

# Logging configuration
LOG_LEVEL=INFO
```

---

## 🗂️ 2. Ingestion & Indexing (Data Preparation)

Before running the chatbot or querying the system, you must ingest the official regulatory documents.

### Step 1: Place Raw Documents
Place all target PDF and TXT documents into the source directory:
```
data/raw/
```

### Step 2: Run Ingestion Pipeline
This parses documents, handles tables/images using Mistral OCR (captioning), chunks the text, and stores dense embeddings in ChromaDB.

* **Standard Ingestion (includes table/image captioning via Mistral OCR):**
  ```bash
  python -m scripts.run_ingestion
  ```
* **Fast Ingestion (skips table/image captioning - useful for quick testing):**
  ```bash
  python -m scripts.run_ingestion --skip-captioning
  ```
* **Custom Data Directory:**
  ```bash
  python -m scripts.run_ingestion --data-dir path/to/custom/raw/dir
  ```

### Step 3: Rebuild Secondary Indices (If needed)
If you already have text chunks in ChromaDB and only want to rebuild search indices without reprocessing raw files:
* **Rebuild TF-IDF Sparse Index (Lexical Search):**
  ```bash
  python -m scripts.rebuild_tfidf
  ```
* **Rebuild ChromaDB Dense Index (Semantic Search from raw `.txt` files directly):**
  ```bash
  python scripts/rebuild_embeddings.py
  ```
* **Clean up low-quality/junk chunks:**
  ```bash
  python -m scripts.cleanup_junk_chunks
  ```

---

## 🌐 3. Running the Application

### Option A: Native Local Execution

#### 1. Start the FastAPI Server
Run the backend API using Uvicorn. This handles session persistence, hybrid search retrieval, and two-stage generation:
```bash
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```
The FastAPI documentation is available at `http://localhost:8000/docs`.

#### 2. Access the Frontend Web App
* **Directly via the API Server:**
  Open your web browser and go to:
  ```
  http://localhost:8000/app
  ```
* **Standalone UI Server (Optional / Development):**
  If you want to run the frontend separately (e.g. to test CORS or run it on a separate port):
  ```bash
  cd webapp
  python -m http.server 3000
  ```
  Access it at `http://localhost:3000`. It will automatically communicate with the backend running on port `8000`.

---

### Option B: Containerized Execution (Docker)

Docker Compose builds and bundles all system dependencies (including `poppler-utils` and `tesseract-ocr`) and mounts the data folders automatically.

#### 1. Start the Containers
```bash
docker-compose up --build -d
```
The application will start in the background. Access the web interface at:
```
http://localhost:8000/app
```

#### 2. Run Ingestion Inside the Container
If you place new files in `data/raw/` on your host machine, they will synchronize with the container. Run the ingestion command inside the running container:
```bash
docker exec -it inpplc-chatbot python -m scripts.run_ingestion
```

#### 3. View Logs / Stop Services
* **View running logs:**
  ```bash
  docker logs -f inpplc-chatbot
  ```
* **Stop the containers:**
  ```bash
  docker-compose down
  ```

---

## 🧪 4. Development & Evaluation Utilities

### 1. Interactive Terminal Client
To query the RAG pipeline directly from the command line without launching the web browser:
```bash
python -m scripts.test_rag_terminal
```

### 2. Run Unit & Integration Tests
Ensure everything is functioning as expected by running `pytest`:
```bash
python -m pytest
```

### 3. Run RAGAS Performance Evaluation
Evaluate your RAG pipeline's faithfulness, answer relevance, context precision, and context recall:
```bash
python -m scripts.run_evaluation
```
* **Evaluation Options:**
  * Define a specific testset: `--queries path/to/testset.json`
  * Limit evaluation size: `--limit 10`
  * Target a specific language: `--lang fr` (or `ar`, `en`)
  * Save results to a custom file: `--output evaluation_results.json`

### 4. Export Evaluation Metrics to CSV
Export the results of the evaluation for analysis or reporting:
```bash
python -m scripts.export_metrics --input evaluation_results.json --output metrics.csv
```

---

## ❓ Troubleshooting & FAQ

### Q: I get a `ConnectionError` when communicating with Ollama.
* **A:** Make sure the Ollama application is active and running in the background. If running inside Docker, ensure `OLLAMA_BASE_URL` in `.env` is set to `http://host.docker.internal:11434` so the container can resolve the host machine's localhost.

### Q: ChromaDB or HuggingFace fails to download embedding models.
* **A:** Check your internet connection and verify that your HF token in `.env` (`HF_TOKEN`) is correct and has read permissions. If you are behind a corporate proxy, you might need to configure system proxy environment variables.

### Q: Mistral API requests are timing out.
* **A:** Ensure your `MISTRAL_API_KEY` is active and has sufficient credits. You can inspect the API latency in the backend console logs.
