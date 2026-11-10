"""Reading the source PDFs.

Two kinds of content are pulled out of the same files:

* prose, split into chunks for the text index; and
* figures, saved as PNGs with their printed captions, for the image index.
"""

import io
import json
import os
import re

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException

from mainfile.config.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    FIGURES_DIR,
    FIGURES_META,
    MIN_IMAGE_HEIGHT,
    MIN_IMAGE_WIDTH,
)

logger = get_logger(__name__)


# --- text -----------------------------------------------------------------

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException(f"The specified data path {DATA_PATH} does not exist.")
        logger.info(f"Loading PDF files from directory: {DATA_PATH}")

        loader = DirectoryLoader(DATA_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()

        if not documents:
            raise CustomException("No PDF files found in the directory.")
        else:
            logger.info(f"Loaded {len(documents)} documents from PDF files.")
        return documents
    except Exception as e:
        logger.error(f"Error loading PDF files: {e}")
        return []


def split_documents(documents):
    try:
        if not documents:
            raise CustomException("No documents provided for splitting.")
        logger.info("Splitting documents into smaller chunks.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP)
        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Splited documents into {len(text_chunks)} chunks.")
        return text_chunks
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        return []


# --- figures --------------------------------------------------------------

# Gale figure captions nearly always close with a reproduction credit.
_CREDIT = re.compile(r"\((?:[^()]*(?:Photograph|Illustration|Custom Medical|"
                     r"Photo Researchers|Reproduced by permission)[^()]*)\)",
                     re.IGNORECASE)


def _caption_for(page, rect):
    """Text block sitting directly beneath a figure, in the same column."""
    best = None
    for block in page.get_text("blocks"):
        x0, y0, x1, y1, text = block[0], block[1], block[2], block[3], block[4]
        if not text.strip():
            continue
        # Must sit under the image and share most of its horizontal span,
        # which is what separates a caption from the neighbouring column.
        overlap = min(x1, rect.x1) - max(x0, rect.x0)
        if overlap < 0.5 * rect.width:
            continue
        if y0 < rect.y1 - 4:
            continue
        gap = y0 - rect.y1
        if gap > 140:
            continue
        if best is None or gap < best[0]:
            best = (gap, " ".join(text.split()))
    return best[1] if best else ""


def _section_for(page, rect):
    """Nearest heading above the figure, found by font size rather than casing."""
    spans = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if text:
                    spans.append((span["size"], span["bbox"][3], text))
    if not spans:
        return ""

    body_size = sorted(s[0] for s in spans)[len(spans) // 2]  # median = body text
    headings = [(size, bottom, text) for size, bottom, text in spans
                if size > body_size + 0.6 and bottom <= rect.y0 and len(text) <= 60]
    if not headings:
        return ""
    # closest heading above the figure
    return max(headings, key=lambda h: h[1])[2]


def _clean_caption(caption):
    """Drop the trailing rights credit; it carries no medical meaning."""
    cleaned = _CREDIT.sub("", caption).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    # PDF line wrapping splits words as "albi- cans"; rejoin them.
    cleaned = re.sub(r"(\w)-\s+(\w)", r"\1\2", cleaned)
    return cleaned.strip(" .;,") or caption.strip()


def extract_figures(pdf_dir=DATA_PATH, out_dir=FIGURES_DIR):
    """Extract every usable figure from each PDF. Returns the manifest list."""
    try:
        import pymupdf
        from PIL import Image

        os.makedirs(out_dir, exist_ok=True)
        pdfs = [os.path.join(pdf_dir, f) for f in sorted(os.listdir(pdf_dir))
                if f.lower().endswith(".pdf")]
        if not pdfs:
            raise CustomException(f"No PDFs found in {pdf_dir}")

        manifest = []
        for pdf_path in pdfs:
            logger.info(f"Extracting figures from {pdf_path}")
            doc = pymupdf.open(pdf_path)
            seen_xrefs = set()

            for page_no, page in enumerate(doc, start=1):
                for img in page.get_images(full=True):
                    xref = img[0]
                    if xref in seen_xrefs:
                        continue

                    rects = page.get_image_rects(xref)
                    if not rects:
                        continue

                    try:
                        raw = doc.extract_image(xref)
                    except Exception:
                        continue

                    if (raw["width"] < MIN_IMAGE_WIDTH
                            or raw["height"] < MIN_IMAGE_HEIGHT):
                        continue  # logo, rule or decorative strip

                    seen_xrefs.add(xref)
                    rect = rects[0]
                    caption = _clean_caption(_caption_for(page, rect))
                    section = _section_for(page, rect)

                    figure_id = f"fig_{page_no:04d}_{xref}"
                    filename = f"{figure_id}.png"
                    try:
                        image = Image.open(io.BytesIO(raw["image"])).convert("RGB")
                        image.save(os.path.join(out_dir, filename), "PNG")
                    except Exception as e:
                        logger.warning(f"Could not save image {figure_id}: {e}")
                        continue

                    manifest.append({
                        "id": figure_id,
                        "file": filename,
                        "page": page_no,
                        "source": os.path.basename(pdf_path),
                        "caption": caption,
                        "section": section,
                        "width": raw["width"],
                        "height": raw["height"],
                    })
            doc.close()

        with open(FIGURES_META, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        with_caption = sum(1 for m in manifest if m["caption"])
        logger.info(f"Extracted {len(manifest)} figures "
                    f"({with_caption} with captions) to {out_dir}")
        return manifest

    except Exception as e:
        logger.error(str(CustomException("Failed to extract figures", e)))
        return []


def load_manifest():
    """Read the figure manifest written by extract_figures()."""
    if not os.path.exists(FIGURES_META):
        return []
    with open(FIGURES_META, encoding="utf-8") as f:
        return json.load(f)
