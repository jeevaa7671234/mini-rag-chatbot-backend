"""ACL-aware PDF extraction and page-preserving, overlapping chunking."""
import argparse
import asyncio
import re
import uuid
from dataclasses import dataclass
from pathlib import Path


# Explicit deny-by-default classification; do not accidentally index a new restricted PDF at level 1.
FILE_ACCESS_LEVELS: dict[str, int] = {
    "hr/leave_policy.pdf": 1,
    "hr/code_of_conduct.pdf": 1,
    "hr/performance_review.pdf": 2,
    "finance/expense_policy.pdf": 1,
    "finance/travel_reimbursement.pdf": 1,
    "exec/compensation_committee.pdf": 3,
    "exec/strategic_plan.pdf": 3,
}
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    department: str
    access_level: int
    source_file: str
    page: int

    def payload(self) -> dict:
        return {
            "text": self.text,
            "department": self.department,
            "access_level": self.access_level,
            "source_file": self.source_file,
            "page": self.page,
        }


def normalize(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Page-bounded character chunks; prefer paragraph/sentence/word boundaries."""
    if size <= overlap or overlap < 0:
        raise ValueError("Chunk size must be larger than a non-negative overlap")
    text = normalize(text)
    result: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            minimum = start + int(size * 0.65)
            # Prefer a coherent boundary late in the window, not an arbitrary split.
            for boundary in ("\n\n", ". ", "? ", "! ", "\n", " "):
                index = text.rfind(boundary, minimum, end)
                if index > start:
                    end = index + len(boundary)
                    break
        part = text[start:end].strip()
        if part:
            result.append(part)
        if end >= len(text):
            break
        next_start = max(start + 1, end - overlap)
        # Don't start in the middle of a word. Bounded advance, preserves overlap approximately.
        if next_start > 0 and text[next_start - 1].isalnum():
            while next_start < end and text[next_start].isalnum():
                next_start += 1
        start = next_start
    return result


def pages_from_pdf(path: Path) -> list[tuple[int, str]]:
    """Native text first; OCR sparse/image pages with 2x DPI using installed Tesseract."""
    import fitz
    from PIL import Image

    result: list[tuple[int, str]] = []
    with fitz.open(path) as document:
        for number, page in enumerate(document, start=1):
            text = normalize(page.get_text("text", sort=True))
            if len(text) < 40:
                try:
                    import pytesseract

                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text = normalize(pytesseract.image_to_string(image, config="--psm 3"))
                    if len(ocr_text) > len(text):
                        text = ocr_text
                except (OSError, RuntimeError, ImportError) as exc:
                    if not text:
                        raise RuntimeError(
                            f"OCR required for {path.name}, page {number}. Install Tesseract OCR, "
                            "ensure `tesseract` is on PATH, and install pytesseract."
                        ) from exc
                    # Short but valid native text is still usable without OCR.
                if not text and page.get_images(full=True):
                    raise ValueError(f"OCR produced no readable text for {path.name}, page {number}")
            if text:
                result.append((number, text))
    return result


def collect_chunks(docs_dir: Path) -> list[Chunk]:
    """Refuse unexpected PDFs and missing mandatory inputs; no inferred permissions."""
    root = docs_dir.resolve()
    actual = sorted(path for path in docs_dir.rglob("*.pdf") if path.is_file())
    found = {path.relative_to(docs_dir).as_posix() for path in actual}
    unexpected = found - FILE_ACCESS_LEVELS.keys()
    missing = FILE_ACCESS_LEVELS.keys() - found
    if unexpected:
        raise ValueError(f"Unknown PDF(s) with no defined ACL: {', '.join(sorted(unexpected))}")
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} required PDFs in {docs_dir}: {', '.join(sorted(missing))}. "
            "Use original company PDFs, or run `python scripts/create_demo_docs.py` for synthetic fixtures."
        )
    chunks: list[Chunk] = []
    for path in actual:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"PDF escapes docs directory: {path}")
        relative = path.relative_to(docs_dir).as_posix()
        department = relative.split("/")[0]
        level = FILE_ACCESS_LEVELS[relative]
        pages = pages_from_pdf(path)
        if not pages:
            raise ValueError(f"No extractable text found in {relative}; refusing partial ingestion")
        for page, text in pages:
            for idx, part in enumerate(chunk_text(text)):
                identifier = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{relative}:{page}:{idx}:{part}"))
                chunks.append(Chunk(identifier, part, department, level, path.name, page))
    return chunks


async def ingest(settings) -> int:
    from app.embeddings import make_embedder
    from app.store import make_store

    chunks = await asyncio.to_thread(collect_chunks, settings.docs_dir)
    embedder = make_embedder(settings)
    # Batch embeddings to avoid memory spikes on larger documents.
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), 64):
        vectors.extend(await embedder.embed([c.text for c in chunks[start : start + 64]]))
    store = make_store(settings)
    await store.replace_all(chunks, vectors, embedder.dimensions)
    return len(chunks)


def main() -> None:
    from app.config import get_settings

    parser = argparse.ArgumentParser(description="Ingest all 7 configured policy PDFs")
    parser.parse_args()
    try:
        count = asyncio.run(ingest(get_settings()))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Ingestion failed: {exc}\n")
    print(f"Ingested {count} chunks with page and access metadata.")


if __name__ == "__main__":
    main()
