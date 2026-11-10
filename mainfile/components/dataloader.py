"""Build every index the assistant needs, from the PDFs in data/.

    python -m mainfile.components.dataloader              # text + figures
    python -m mainfile.components.dataloader --text-only
    python -m mainfile.components.dataloader --figures-only
    python -m mainfile.components.dataloader --skip-extract   # reuse figures
"""

from mainfile.components.document_loader import (
    extract_figures,
    load_pdf_files,
    split_documents,
)
from mainfile.components.vectorstore import save_image_vector_store, save_vector_store
from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException

logger = get_logger(__name__)


def processandstore_pdf():
    """Text half: load the PDFs, chunk them, build the text index."""
    try:
        logger.info("Starting process to load, split, and store PDF documents.")

        documents = load_pdf_files()

        text_chunks = split_documents(documents)

        save_vector_store(text_chunks)

        logger.info("Process completed successfully.")
        return True

    except Exception as e:
        error_message = CustomException(f"Error in processing and storing PDF documents: {e}")
        logger.error(str(error_message))
        return False


def processandstore_figures(skip_extract=False):
    """Image half: harvest figures, then build the CLIP and caption indexes."""
    try:
        if skip_extract:
            logger.info("Skipping extraction; reusing the existing manifest.")
        else:
            figures = extract_figures()
            if not figures:
                raise CustomException("No figures were extracted from the PDFs.")
            logger.info(f"Extracted {len(figures)} figures.")

        image_db, caption_db = save_image_vector_store()
        if image_db is None or caption_db is None:
            raise CustomException("Figure indexes could not be built.")

        logger.info("Figure pipeline completed successfully.")
        return True

    except Exception as e:
        logger.error(str(CustomException("Figure pipeline failed", e)))
        return False


def build_all(text=True, figures=True, skip_extract=False):
    ok = True
    if text:
        ok = processandstore_pdf() and ok
    if figures:
        ok = processandstore_figures(skip_extract=skip_extract) and ok
    return ok


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    build_all(
        text="--figures-only" not in args,
        figures="--text-only" not in args,
        skip_extract="--skip-extract" in args,
    )
