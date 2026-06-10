"""PDF document parser using the Mistral OCR API.

Sends PDFs to Mistral's OCR endpoint and then post-processes the raw
markdown output with a structured prompt that:
  • extracts clean, organized text
  • describes graphs, charts, and schemas in natural language
  • describes tables with all numerical data preserved

Implements SHA-256 hash-based idempotency to skip already-processed documents.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import httpx
from mistralai.client import Mistral

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Post-processing prompt sent to Mistral chat after OCR extraction
# ---------------------------------------------------------------------------

POST_PROCESS_PROMPT = """\
You are a text extraction tool. You receive raw OCR output from a PDF. \
Your ONLY job is to return the cleaned-up text from the document.

STRICT RULES — VIOLATION OF ANY RULE IS UNACCEPTABLE:
1. Output ONLY the text that exists in the document. Nothing else.
2. NEVER add introductions like "Voici le texte", "Here is", "Note:", etc.
3. NEVER add your own commentary, analysis, explanations, or observations.
4. NEVER add notes about OCR quality or missing content.
5. NEVER fabricate or invent content that is not in the original.
6. If the OCR output contains very little readable text, return ONLY that text.
7. If the OCR output is empty or unreadable, return exactly: [EMPTY DOCUMENT]

CLEANING RULES:
- Fix obvious OCR errors (broken words, stray characters) without changing meaning.
- Remove repeated decorative characters (e.g. "AAAAAAA") that are clearly OCR noise.
- Preserve the original language (French, Arabic, English) — do NOT translate.
- Preserve section headings, numbered lists, and bullet points.
- Preserve tables in clean markdown table format.
- For graphs/charts/schemas: write a brief factual description of what the figure shows, \
prefixed with [FIGURE:] on its own line. Keep it to 2-3 sentences max.

START YOUR OUTPUT DIRECTLY WITH THE DOCUMENT CONTENT. NO PREAMBLE.

---
RAW OCR CONTENT:
{ocr_content}
"""

# ---------------------------------------------------------------------------
# Mistral OCR + Chat parser
# ---------------------------------------------------------------------------

OCR_MODEL = "mistral-ocr-latest"
CHAT_MODEL = "mistral-large-latest"


class PDFParser:
    """Mistral OCR-based PDF element extractor.

    Sends PDFs to Mistral OCR, then post-processes the markdown output
    through a Mistral chat model to produce clean, structured text with
    rich descriptions of tables, graphs, and schemas.

    Args:
        settings: Application settings. Uses defaults if not provided.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._output_dir = self._settings.processed_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

        api_key = self._settings.mistral_api_key
        if not api_key:
            raise ValueError(
                "MISTRAL_API_KEY is not set. Add it to your .env file."
            )
        self._client = Mistral(
            api_key=api_key,
            timeout_ms=300_000,  # 5 minutes — large PDFs need time
        )

    # ------------------------------------------------------------------
    # Hashing / manifest helpers (unchanged contract)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_file_hash(filepath: Path) -> str:
        """Compute SHA-256 hash of a file for idempotency checking.

        Args:
            filepath: Path to the file to hash.

        Returns:
            Hex string of the SHA-256 hash.
        """
        sha256 = hashlib.sha256()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _load_manifest(self) -> dict[str, str]:
        """Load the processing manifest tracking already-processed PDFs.

        Returns:
            Dictionary mapping file hashes to filenames.
        """
        manifest_path = self._settings.manifest_path
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        return {}

    def _save_manifest(self, manifest: dict[str, str]) -> None:
        """Save the processing manifest to disk.

        Args:
            manifest: Dictionary mapping file hashes to filenames.
        """
        self._settings.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Mistral OCR
    # ------------------------------------------------------------------

    def _ocr_pdf(self, pdf_path: Path) -> str:
        """Send a PDF to Mistral OCR and return combined markdown.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Raw markdown text from all pages.
        """
        with open(pdf_path, "rb") as f:
            pdf_base64 = base64.b64encode(f.read()).decode("utf-8")

        logger.info("calling_mistral_ocr", filename=pdf_path.name)

        ocr_response = self._client.ocr.process(
            model=OCR_MODEL,
            document={
                "type": "document_url",
                "document_url": f"data:application/pdf;base64,{pdf_base64}",
            },
        )

        pages_md: list[str] = []
        for page in ocr_response.pages:
            pages_md.append(page.markdown)

        raw_markdown = "\n\n".join(pages_md)
        logger.info(
            "ocr_complete",
            filename=pdf_path.name,
            pages=len(ocr_response.pages),
            chars=len(raw_markdown),
        )
        return raw_markdown

    # ------------------------------------------------------------------
    # Post-processing via Mistral Chat
    # ------------------------------------------------------------------

    def _post_process(self, raw_markdown: str, filename: str) -> str:
        """Post-process raw OCR markdown through Mistral chat.

        Uses a structured prompt to extract clean text and generate
        descriptions for tables, graphs, and schemas.

        Args:
            raw_markdown: Raw OCR output.
            filename: Source filename for logging.

        Returns:
            Clean, structured text ready for the RAG pipeline.
        """
        prompt = POST_PROCESS_PROMPT.format(ocr_content=raw_markdown)

        logger.info("post_processing", filename=filename)

        response = self._client.chat.complete(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
        )

        result = response.choices[0].message.content
        logger.info(
            "post_processing_complete",
            filename=filename,
            output_chars=len(result),
        )
        return result

    # ------------------------------------------------------------------
    # Public API (same contract as before)
    # ------------------------------------------------------------------

    def extract_document_elements(
        self,
        file_path: Path,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Extract text, table, and image elements from a PDF via Mistral OCR.

        The Mistral OCR + chat post-processing pipeline returns a single
        unified text block that already contains inline table and figure
        descriptions. We wrap the entire output as a single text element
        so downstream chunking works unchanged.

        For .txt files we fall back to plain text reading.

        Args:
            file_path: Path to the document file.

        Returns:
            Tuple of (text_elements, table_elements, image_elements).
            table_elements and image_elements are always empty because
            their descriptions are inlined into the text by the prompt.
        """
        logger.info("parsing_document", filename=file_path.name)

        if file_path.suffix.lower() == ".pdf":
            # Step 1: OCR
            raw_markdown = self._ocr_pdf(file_path)

            # Step 2: Post-process with structured prompt
            clean_text = self._post_process(raw_markdown, file_path.name)

            text_elements = [
                {
                    "content": clean_text,
                    "type": "text",
                    "metadata": {
                        "source_file": file_path.name,
                        "parser": "mistral_ocr",
                    },
                }
            ]

        elif file_path.suffix.lower() == ".txt":
            content = file_path.read_text(encoding="utf-8", errors="replace")
            text_elements = [
                {
                    "content": content,
                    "type": "text",
                    "metadata": {
                        "source_file": file_path.name,
                        "parser": "plaintext",
                    },
                }
            ] if content.strip() else []

        else:
            logger.warning("unsupported_file_type", filename=file_path.name)
            return [], [], []

        # Table and image descriptions are inlined by the prompt
        table_elements: list[dict[str, Any]] = []
        image_elements: list[dict[str, Any]] = []

        logger.info(
            "document_parsed",
            filename=file_path.name,
            texts=len(text_elements),
            tables=len(table_elements),
            images=len(image_elements),
        )

        return text_elements, table_elements, image_elements

    def _save_elements(
        self,
        pdf_path: Path,
        text_elements: list[dict[str, Any]],
        table_elements: list[dict[str, Any]],
        image_elements: list[dict[str, Any]],
    ) -> Path:
        """Save extracted elements to a JSON file for caching.

        Args:
            pdf_path: Original PDF path (used for naming).
            text_elements: Extracted text elements.
            table_elements: Extracted table elements.
            image_elements: Extracted image elements.

        Returns:
            Path to the saved JSON file.
        """
        output_path = self._output_dir / f"{pdf_path.stem}_elements.json"
        data = {
            "source_file": pdf_path.name,
            "text_elements": text_elements,
            "table_elements": table_elements,
            "image_elements": image_elements,
        }
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def load_cached_elements(
        self, pdf_path: Path
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
        """Load previously extracted elements from cache.

        Args:
            pdf_path: Original PDF path to look up cached results.

        Returns:
            Tuple of (text, table, image) elements if cached, None otherwise.
        """
        cache_path = self._output_dir / f"{pdf_path.stem}_elements.json"
        if not cache_path.exists():
            return None

        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return (
                data["text_elements"],
                data["table_elements"],
                data["image_elements"],
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("cache_load_failed", path=str(cache_path), error=str(e))
            return None

    def process_documents(
        self,
        data_dir: Path | None = None,
    ) -> list[dict[str, Any]]:
        """Process all supported documents in a directory recursively with idempotency.

        Skips files whose SHA-256 hash already exists in the manifest.
        Saves extracted elements to data/processed/ for caching.

        Args:
            data_dir: Directory containing documents. Defaults to settings.raw_pdf_dir.

        Returns:
            List of all processed element sets (text, table, image per document).
        """
        data_dir = data_dir or self._settings.raw_pdf_dir
        manifest = self._load_manifest()
        all_results: list[dict[str, Any]] = []

        # Find both PDF and TXT files recursively
        doc_files = []
        doc_files.extend(data_dir.rglob("*.pdf"))
        doc_files.extend(data_dir.rglob("*.txt"))
        doc_files.extend(data_dir.rglob("*.PDF"))
        doc_files.extend(data_dir.rglob("*.TXT"))

        # Sort for deterministic processing
        doc_files = sorted(list(set(doc_files)))

        logger.info("processing_documents", total=len(doc_files))

        for doc_path in doc_files:
            file_hash = self.compute_file_hash(doc_path)

            if file_hash in manifest:
                logger.info("skipping_processed", filename=doc_path.name)
                cached = self.load_cached_elements(doc_path)
                if cached:
                    text_els, table_els, image_els = cached
                    all_results.append({
                        "source_file": doc_path.name,
                        "text_elements": text_els,
                        "table_elements": table_els,
                        "image_elements": image_els,
                    })
                continue

            try:
                text_els, table_els, image_els = self.extract_document_elements(doc_path)
                self._save_elements(doc_path, text_els, table_els, image_els)

                manifest[file_hash] = doc_path.name
                self._save_manifest(manifest)

                all_results.append({
                    "source_file": doc_path.name,
                    "text_elements": text_els,
                    "table_elements": table_els,
                    "image_elements": image_els,
                })

                # Small delay between API calls to respect rate limits
                time.sleep(0.5)

            except Exception as e:
                logger.error(
                    "document_processing_error",
                    filename=doc_path.name,
                    error=str(e),
                )

        logger.info("processing_complete", total_processed=len(all_results))
        return all_results
