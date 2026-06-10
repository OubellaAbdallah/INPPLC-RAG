import os
import time
import re
import warnings
import base64
import traceback
from pathlib import Path
from mistralai.client import Mistral
from dotenv import load_dotenv
import httpx

# ── 1. Suppress Package Warnings ──────────────────────────────────────────────
warnings.filterwarnings("ignore", category=UserWarning, module="requests")

# Load environment configuration securely
env_path = Path(r"C:\Users\aoube\3D Objects\Inpplc-RAG\.env")
load_dotenv(dotenv_path=env_path)

# Extract API Keys
LLAMA_API_KEY   = os.getenv("LLAMA_CLOUD_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# ── 2. Directory Configurations ───────────────────────────────────────────────
PDF_DIR    = Path(r"C:\Users\aoube\3D Objects\Inpplc-RAG\data\PDFS")
OUTPUT_DIR = Path(r"C:\Users\aoube\3D Objects\Inpplc-RAG\data\TEXTS-LLAMA")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# LlamaParse API Endpoints & Polling Limits
LLAMA_PARSE_URL   = "https://api.cloud.llamaindex.ai/api/parsing"
POLL_INTERVAL     = 3      # Seconds between status checks
POLL_MAX_ATTEMPTS = 120    # Maximum wait time per PDF (6 minutes)

# Feature Flag: Keep disabled to minimize premium token draw and prevent hallucinations
USE_IMAGE_DESCRIPTIONS = False

# ── 3. Initialize Clients ─────────────────────────────────────────────────────
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

# ── 4. Robust Text Post-Processing ────────────────────────────────────────────
def clean_extracted_text(text: str) -> str:
    """Cleans up raw structural layouts and filters out formatting noise."""
    # Remove excessive Markdown markup characters
    text = re.sub(r"[#*_`|]+", " ", text)

    # Strip away horizontal page line separators
    text = re.sub(r"\n[-=]{2,}\n", "\n", text)

    # Wipe out LlamaParse system tags and empty strings
    bad_patterns = [
        r"(?i)document begins",
        r"(?i)document ends",
        r"(?i)here is the text",
        r"(?i)the image shows",
        r"(?i)summary",
        r"(?i)NO_CONTENT_HERE",
        r"\[IMAGE:.*?\]",
        r"\[/?FIGURE.*?\]",
        r"!\[.*?\]\(.*?\)",
        r"(?m)^-{2,}$",
        r"(?m)^={2,}$",
        r"(?m)^\.{2,}$",
    ]
    for p in bad_patterns:
        text = re.sub(p, " ", text)

    # Process and eliminate duplicate headers/footers
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line or len(line) <= 2:
            continue
        # Skip isolated running page numbers
        if re.fullmatch(r"\d+", line):
            continue
        # Skip lingering decoration lines
        if re.fullmatch(r"[-_=\.]{3,}", line):
            continue
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # Clean out directional markers breaking Arabic text indices
    text = re.sub(r"\u200f", "", text)
    text = re.sub(r"\u200e", "", text)

    # Normalize structural spacing
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

# ── 5. LlamaParse Client Interface ───────────────────────────────────────────
def llamaparse_upload(pdf_path: Path) -> str:
    """Uploads document with explicit timeouts optimized for large file pools."""
    with open(pdf_path, "rb") as f:
        files = {"file": (pdf_path.name, f, "application/pdf")}
        data  = {
            "parsing_instruction": """
            Extract ONLY the exact text present in the document.

            Rules:
            - Preserve Arabic exactly as written
            - Preserve French exactly as written
            - Do NOT translate
            - Do NOT summarize
            - Do NOT rewrite
            - Do NOT explain
            - Do NOT add introductions
            - Ignore logos
            - Ignore decorative elements
            - Ignore repeated headers and footers
            - Keep reading order
            - Keep tables as readable text
            - Extract only visible text from charts and figures
            - Do NOT generate image placeholders
            - Do NOT generate figure placeholders
            - Do NOT output NO_CONTENT_HERE
            """,
            "language": "ar",  # Locks directional flow to RTL-stable arrays
            "premium_mode": "false",
        }
        headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
        
        # FIXED: Explicit timeout allocation mapping to prevent network socket drops on 9MB+ files
        timeout_config = httpx.Timeout(timeout=300.0, connect=60.0)

        resp = httpx.post(
            f"{LLAMA_PARSE_URL}/upload",
            headers=headers,
            files=files,
            data=data,
            timeout=timeout_config,
        )
    resp.raise_for_status()
    job_id = resp.json()["id"]
    print(f"    Uploaded → job {job_id}", flush=True)
    return job_id


def llamaparse_wait(job_id: str) -> str:
    """Blocks execution loop natively until LlamaCloud completes extraction."""
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
    for attempt in range(POLL_MAX_ATTEMPTS):
        resp = httpx.get(
            f"{LLAMA_PARSE_URL}/job/{job_id}",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        status = resp.json().get("status", "")
        print(f"    [{attempt+1}] status: {status}", flush=True)
        if status == "SUCCESS":
            return status
        if status in ("ERROR", "CANCELLED"):
            raise RuntimeError(f"LlamaParse job failed: {status}")
        time.sleep(POLL_INTERVAL)
    raise TimeoutError(f"Job {job_id} timed out after maximum polling duration threshold.")


def llamaparse_get_markdown(job_id: str) -> str:
    """Retrieves full multiline page strings across complete document arrays."""
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
    resp = httpx.get(
        f"{LLAMA_PARSE_URL}/job/{job_id}/result/markdown",
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("markdown", "")


def llamaparse_get_images(job_id: str) -> list[dict]:
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
    resp = httpx.get(
        f"{LLAMA_PARSE_URL}/job/{job_id}/result/images",
        headers=headers,
        timeout=60,
    )
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json().get("images", [])


def llamaparse_download_image(job_id: str, image_name: str) -> bytes:
    headers = {"Authorization": f"Bearer {LLAMA_API_KEY}"}
    resp = httpx.get(
        f"{LLAMA_PARSE_URL}/job/{job_id}/result/image/{image_name}",
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.content


def describe_image_with_mistral(image_bytes: bytes, image_name: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    ext = Path(image_name).suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/png")

    response = mistral_client.chat.complete(
        model="pixtral-12b-2409",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {
                        "type": "text",
                        "text": "Extract visible graph information into literal Markdown table layouts.",
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content.strip()


def enrich_markdown_with_image_descriptions(markdown: str, job_id: str, images: list[dict]) -> str:
    if not images:
        return markdown

    for img_meta in images:
        img_name = img_meta.get("name") or img_meta.get("file_name", "")
        if not img_name:
            continue

        try:
            img_bytes   = llamaparse_download_image(job_id, img_name)
            description = describe_image_with_mistral(img_bytes, img_name)
            placeholder = f"[IMAGE: {img_name}]"
            replacement = f"\n{description}\n"
            
            if placeholder in markdown:
                markdown = markdown.replace(placeholder, replacement, 1)
            else:
                markdown += f"\n{replacement}"
        except Exception:
            pass

    return markdown


# ── 6. Worker Lifecycle Pipeline ──────────────────────────────────────────────
def process_pdf(pdf_path: Path) -> str:
    print(f"\n▶ {pdf_path.name} ({pdf_path.stat().st_size/1024:.0f} KB)", flush=True)

    # 1. Thread Stream Upload
    job_id = llamaparse_upload(pdf_path)

    # 2. Async Lifecycle Wait Block
    llamaparse_wait(job_id)

    # 3. Payload Fetching
    print("    Fetching markdown text arrays...", flush=True)
    markdown = llamaparse_get_markdown(job_id)

    # 4. Optional Multimodal Processing Switch
    if USE_IMAGE_DESCRIPTIONS:
        print("    Fetching extracted diagram listings...", flush=True)
        images = llamaparse_get_images(job_id)
        if images:
            print(f"    Processing {len(images)} structural assets with Pixtral...", flush=True)
            markdown = enrich_markdown_with_image_descriptions(markdown, job_id, images)

    return markdown


# ── 7. Core Production Batch Execution Loop ──────────────────────────────────
if __name__ == "__main__":
    pdf_list = sorted(PDF_DIR.glob("**/*.pdf")) + sorted(PDF_DIR.glob("**/*.PDF"))
    pdf_list = list(dict.fromkeys(pdf_list))  # Deduplicate path entries
    print(f"Found {len(pdf_list)} target PDF documents inside source path storage.", flush=True)

    failed = []

    for pdf_path in pdf_list:
        rel      = pdf_path.relative_to(PDF_DIR)
        out_path = OUTPUT_DIR / rel.with_suffix(".txt")

        # Skip files that have already been cleanly parsed on disk
        if out_path.exists():
            print(f"skip (done): {pdf_path.name}", flush=True)
            continue

        # FIXED: Enclosing the individual loop execution inside robust error containment
        try:
            raw_text = process_pdf(pdf_path)

            if raw_text.strip():
                # Clean up extraction strings
                cleaned_text = clean_extracted_text(raw_text)
                
                if cleaned_text.strip():
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(cleaned_text, encoding="utf-8")
                    print(f"  ✅ saved {len(cleaned_text)} chars → {out_path.name}", flush=True)
                else:
                    print(f"  ⚠ Scrubbed text evaluated to an empty sequence layout.", flush=True)
                    failed.append(str(pdf_path))
            else:
                print(f"  ⚠ Target layout returned blank raw sequence vectors.", flush=True)
                failed.append(str(pdf_path))

        except Exception as e:
            # FIXED: Captures runtime connection resets/timeouts, prints notice, and forces processing continuation
            print(f"  ❌ ERROR ENCOUNTERED during execution pass on [{pdf_path.name}]: {e}", flush=True)
            failed.append(str(pdf_path))
            time.sleep(2)
            continue

        time.sleep(1)

    print(f"\n── Batch Processing Run Summary ──────────────────────────────")
    print(f"  Total Processed Load: {len(pdf_list)}")
    print(f"  Clean Completions:    {len(pdf_list) - len(failed)}")
    print(f"  Failure Log Size:    {len(failed)}")
    if failed:
        print("\nReview individual target processing paths failing execution constraints:")
        for item in failed:
            print(f"    - {item}")