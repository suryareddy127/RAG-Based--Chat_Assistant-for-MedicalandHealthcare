"""FAISS indexes for both halves of the corpus.

Three indexes are maintained:

* the text index over encyclopedia prose (MiniLM);
* a CLIP index over figure pixels, so a question can match what a figure
  actually depicts; and
* a MiniLM index over figure captions, which is far better at fine medical
  wording than CLIP ("cataract" vs "corneal abrasion").

Figure results merge the last two with reciprocal rank fusion, which needs
only the ranks and so avoids comparing two incompatible score scales.
"""

import os

from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

from mainfile.components.embeddings import ClipEmbeddings, get_huggingface_embeddings
from mainfile.components.document_loader import load_manifest

from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException
from mainfile.config.config import (
    CAPTION_DB_PATH,
    DB_FAISS_PATH,
    IMAGE_CANDIDATES,
    IMAGE_DB_PATH,
    IMAGE_MIN_SCORE,
    IMAGE_TOP_K,
)

logger = get_logger(__name__)

# The indexes are several MB on disk; load each once and reuse across requests.
_vectorstore = None
_image_db = None
_caption_db = None

RRF_K = 60  # standard reciprocal-rank-fusion damping constant


# --- text index -----------------------------------------------------------

def loader_faiss_vectorstore():
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    try:
        logger.info("Loading FAISS vector store from disk.")

        if not os.path.exists(DB_FAISS_PATH):
            logger.warning(
                f"FAISS database not found at {DB_FAISS_PATH}. "
                "Run 'python -m mainfile.components.dataloader' to build it. Returning None."
            )
            return None

        embedding_model = get_huggingface_embeddings()
        if embedding_model is None:
            raise CustomException("Embedding model could not be initialized")

        logger.info(f"FAISS database found at {DB_FAISS_PATH}. Loading...")
        _vectorstore = FAISS.load_local(
            DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
        )
        return _vectorstore

    except Exception as e:
        error_message = CustomException("Error loading FAISS vector store", e)
        logger.error(str(error_message))
        return None


#new vector store save function
def save_vector_store(text_chunks):
    global _vectorstore

    try:
        if not text_chunks:
            raise CustomException("No chunks were found..")

        logger.info("Generating your new vectorstore")

        embedding_model = get_huggingface_embeddings()
        if embedding_model is None:
            raise CustomException("Embedding model could not be initialized")

        db = FAISS.from_documents(text_chunks, embedding_model)

        logger.info("Saving vectorstoree")

        db.save_local(DB_FAISS_PATH)

        logger.info("Vectostore saved sucesfulyy...")

        _vectorstore = db
        return db

    except Exception as e:
        error_message = CustomException("Failed to craete new vectorstore", e)
        logger.error(str(error_message))
        return None


# --- figure indexes -------------------------------------------------------

def _metadata_for(entry):
    return {
        "id": entry["id"],
        "file": entry["file"],
        "page": entry["page"],
        "caption": entry["caption"],
        "section": entry.get("section", ""),
        "source": entry.get("source", ""),
    }


def save_image_vector_store():
    """Build both figure indexes from the extracted manifest."""
    global _image_db, _caption_db

    try:
        manifest = load_manifest()
        if not manifest:
            raise CustomException(
                "No figures found. Run mainfile.components.dataloader first."
            )

        metadatas = [_metadata_for(entry) for entry in manifest]

        logger.info(f"Embedding {len(manifest)} figures with CLIP...")
        image_db = FAISS.from_texts(
            texts=[entry["file"] for entry in manifest],   # filenames -> pixels
            embedding=ClipEmbeddings(),
            metadatas=metadatas,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,  # cosine
        )
        image_db.save_local(IMAGE_DB_PATH)
        logger.info(f"Saved CLIP figure index to {IMAGE_DB_PATH}")

        caption_texts = [
            " ".join(filter(None, [entry.get("section", ""), entry["caption"]]))
            or entry["file"]
            for entry in manifest
        ]
        logger.info("Embedding figure captions...")
        caption_db = FAISS.from_texts(
            texts=caption_texts,
            embedding=get_huggingface_embeddings(),
            metadatas=metadatas,
        )
        caption_db.save_local(CAPTION_DB_PATH)
        logger.info(f"Saved caption index to {CAPTION_DB_PATH}")

        _image_db, _caption_db = image_db, caption_db
        return image_db, caption_db

    except Exception as e:
        logger.error(str(CustomException("Failed to build figure indexes", e)))
        return None, None


def loader_image_vectorstore():
    """Load both figure indexes, caching them for later requests."""
    global _image_db, _caption_db
    if _image_db is not None and _caption_db is not None:
        return _image_db, _caption_db

    try:
        if not (os.path.exists(IMAGE_DB_PATH) and os.path.exists(CAPTION_DB_PATH)):
            logger.warning(
                "Figure indexes not found. Run "
                "'python -m mainfile.components.dataloader' to build them."
            )
            return None, None

        _image_db = FAISS.load_local(
            IMAGE_DB_PATH, ClipEmbeddings(),
            allow_dangerous_deserialization=True,
            distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        )
        _caption_db = FAISS.load_local(
            CAPTION_DB_PATH, get_huggingface_embeddings(),
            allow_dangerous_deserialization=True,
        )
        logger.info("Figure indexes loaded.")
        return _image_db, _caption_db

    except Exception as e:
        logger.error(str(CustomException("Failed to load figure indexes", e)))
        return None, None


def search_figures(query, k=IMAGE_TOP_K, min_score=IMAGE_MIN_SCORE):
    """Return figures relevant to `query`, or [] when none are relevant.

    The CLIP cosine acts as the relevance gate so that questions with no
    matching figure (and non-medical questions) return nothing at all.
    """
    image_db, caption_db = loader_image_vectorstore()
    if image_db is None or caption_db is None:
        return []

    try:
        clip_hits = image_db.similarity_search_with_score(query, k=IMAGE_CANDIDATES)
        caption_hits = caption_db.similarity_search_with_score(query, k=IMAGE_CANDIDATES)

        fused = {}
        clip_score = {}

        for rank, (doc, score) in enumerate(clip_hits):
            fid = doc.metadata["id"]
            clip_score[fid] = float(score)
            fused.setdefault(fid, {"meta": doc.metadata, "score": 0.0})
            fused[fid]["score"] += 1.0 / (RRF_K + rank)

        caption_ids = set()
        for rank, (doc, _score) in enumerate(caption_hits):
            fid = doc.metadata["id"]
            caption_ids.add(fid)
            fused.setdefault(fid, {"meta": doc.metadata, "score": 0.0})
            fused[fid]["score"] += 1.0 / (RRF_K + rank)

        ordered = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

        results = []
        for item in ordered:
            fid = item["meta"]["id"]
            # Only keep figures CLIP itself considers a real visual match.
            if clip_score.get(fid, 0.0) < min_score:
                continue
            # CLIP scores sit in a narrow band, so an unrelated medical photo
            # can clear the gate. Past the best hit, require the caption index
            # to agree before showing a figure.
            if results and fid not in caption_ids:
                continue
            meta = dict(item["meta"])
            meta["score"] = round(clip_score.get(fid, 0.0), 4)
            results.append(meta)
            if len(results) >= k:
                break

        return results

    except Exception as e:
        logger.error(str(CustomException("Figure search failed", e)))
        return []
