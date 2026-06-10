"""CLI script to extract text from PDFs in balanced blocks with resume support.

Splits PDFs into balanced blocks, processes them one at a time, and saves
progress after every single PDF so you can stop (Ctrl+C) and resume later.
Each block gets its own failure log showing which PDFs failed.

Usage:
    python -m scripts.extract_text [--block-size 5] [--max-retries 3]
    python -m scripts.extract_text --data-dir data/PDFS --out-dir data/TEXTS
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

from src.config.settings import get_settings
from src.ingestion.parser import PDFParser
from src.utils.logging import get_logger

STATE_FILENAME = "blocks_state.json"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def discover_pdfs(data_dir: Path) -> list[Path]:
    """Find all PDFs sorted for deterministic ordering."""
    return sorted(set(data_dir.rglob("*.pdf")) | set(data_dir.rglob("*.PDF")))


def build_blocks(pdf_files: list[Path], block_size: int) -> list[list[Path]]:
    """Split pdf_files into balanced blocks of approximately block_size."""
    if not pdf_files:
        return []
    n_blocks = math.ceil(len(pdf_files) / block_size)
    blocks: list[list[Path]] = []
    base, remainder = divmod(len(pdf_files), n_blocks)
    idx = 0
    for i in range(n_blocks):
        size = base + (1 if i < remainder else 0)
        blocks.append(pdf_files[idx : idx + size])
        idx += size
    return blocks


def load_state(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract text from PDFs in balanced blocks (resumable)"
    )
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--block-size", type=int, default=5,
                    help="PDFs per block (default: 5)")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="Retries per failed PDF (default: 3)")
    args = ap.parse_args()

    logger = get_logger("extract_text")
    settings = get_settings()

    data_dir = args.data_dir or settings.data_dir / "PDFS"
    out_dir = args.out_dir or settings.data_dir / "TEXTS"
    state_path = out_dir / STATE_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover PDFs ─────────────────────────────────────────────
    pdf_files = discover_pdfs(data_dir)
    print(f"\n📁 Found {len(pdf_files)} PDF(s) in {data_dir}")
    if not pdf_files:
        print("Nothing to process.")
        return

    # ── Load or create state ──────────────────────────────────────
    state = load_state(state_path)

    if not state or state.get("block_size") != args.block_size:
        blocks = build_blocks(pdf_files, args.block_size)
        state = {
            "block_size": args.block_size,
            "total_pdfs": len(pdf_files),
            "total_blocks": len(blocks),
            "blocks": {},
        }
        for i, blk in enumerate(blocks, 1):
            name = f"block_{i:02d}"
            state["blocks"][name] = {
                "status": "pending",
                "pdfs": [str(p.relative_to(data_dir)) for p in blk],
                "extracted": [],
                "failed": [],
            }
        save_state(state_path, state)
        print(f"📦 Split into {len(blocks)} block(s) of ~{args.block_size} PDFs each")
    else:
        print(f"📦 Resuming: {state['total_blocks']} block(s) found in state")

    # ── Overview ──────────────────────────────────────────────────
    blocks_info = state["blocks"]
    done = sum(1 for b in blocks_info.values() if b["status"] == "done")
    print(f"   ✅ {done}/{len(blocks_info)} block(s) already completed\n")

    # ── Init parser ───────────────────────────────────────────────
    pdf_parser = PDFParser(settings)

    # ── Process blocks one by one ─────────────────────────────────
    for block_name, bdata in blocks_info.items():
        if bdata["status"] == "done":
            continue

        pdfs = bdata["pdfs"]
        print(f"\n{'='*60}")
        print(f"  📦 {block_name}  —  {len(pdfs)} PDF(s)")
        print(f"{'='*60}")

        block_out = out_dir / block_name
        block_out.mkdir(parents=True, exist_ok=True)

        extracted = list(bdata.get("extracted", []))
        failed: list[dict[str, str]] = []

        for rel in pdfs:
            pdf_path = data_dir / rel
            txt_path = block_out / Path(rel).with_suffix(".txt").name

            # Already done
            if rel in extracted or txt_path.exists():
                if rel not in extracted:
                    extracted.append(rel)
                continue

            print(f"  📄 {Path(rel).name} ...", end="", flush=True)
            ok = False
            err = ""

            for attempt in range(1, args.max_retries + 1):
                try:
                    elems, _, _ = pdf_parser.extract_document_elements(pdf_path)
                    if elems:
                        txt_path.write_text(elems[0]["content"], encoding="utf-8")
                        chars = len(elems[0]["content"])
                        print(f" ✅ ({chars:,} chars)")
                        extracted.append(rel)
                        ok = True
                        break
                    else:
                        err = "No text extracted"
                except Exception as e:
                    err = str(e)

                if attempt < args.max_retries:
                    wait = 15 * attempt
                    print(f" ⚠ retry {attempt}/{args.max_retries}...", end="", flush=True)
                    time.sleep(wait)

            if not ok:
                print(f" ❌ {err[:100]}")
                failed.append({"file": rel, "error": err})

            # Save progress after EVERY pdf
            bdata["extracted"] = extracted
            bdata["failed"] = failed
            save_state(state_path, state)

        # ── Mark block done ───────────────────────────────────────
        bdata["status"] = "done"
        bdata["extracted"] = extracted
        bdata["failed"] = failed
        save_state(state_path, state)

        # Per-block failure log
        if failed:
            log = block_out / "failed_pdfs.json"
            log.write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n  ── {block_name} summary ──")
        print(f"     ✅ Extracted: {len(extracted)}/{len(pdfs)}")
        if failed:
            print(f"     ❌ Failed:    {len(failed)}")
            for f in failed:
                print(f"        • {Path(f['file']).name}: {f['error'][:80]}")
            print(f"     📝 Failure log: {block_out / 'failed_pdfs.json'}")
        else:
            print(f"     🎉 All PDFs extracted!")

        print(f"\n  ⏸  Safe to stop here (Ctrl+C). Re-run to continue.\n")

    # ── Final report ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  📊 FINAL REPORT")
    print("=" * 60)
    tot_ok = tot_fail = 0
    for bname, bd in blocks_info.items():
        n_ok = len(bd.get("extracted", []))
        n_fail = len(bd.get("failed", []))
        tot_ok += n_ok
        tot_fail += n_fail
        icon = "✅" if n_fail == 0 else "⚠ "
        print(f"  {icon} {bname}: {n_ok} extracted, {n_fail} failed")

    print(f"\n  Total: {tot_ok}/{state['total_pdfs']} extracted, {tot_fail} failed")
    print(f"  Output: {out_dir}")

    if tot_fail:
        print(f"\n  💡 To retry failed PDFs: edit block status to 'pending' in")
        print(f"     {state_path}")
    else:
        print(f"\n  🎉 All PDFs extracted successfully!")


if __name__ == "__main__":
    main()
