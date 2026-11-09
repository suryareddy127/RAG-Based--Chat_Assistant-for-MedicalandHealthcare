from langchain_huggingface import HuggingFaceEmbeddings

from mainfile.common.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Loading the SentenceTransformer takes several seconds, so keep one instance.
_embeddings = None


def get_huggingface_embeddings():
    global _embeddings

    if _embeddings is not None:
        return _embeddings

    try:
        logger.info("Initializing HuggingFace Embeddings.")

        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

        logger.info("HuggingFace Embeddings initialized successfully.")
        return _embeddings
    except Exception as e:
        logger.error(f"Error initializing HuggingFace Embeddings: {e}")
        return None
