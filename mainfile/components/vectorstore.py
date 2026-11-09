import os
from langchain_community.vectorstores import FAISS

from mainfile.components.embeddings import get_huggingface_embeddings

from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException
from mainfile.config.config import DB_FAISS_PATH

logger = get_logger(__name__)

# The FAISS index is ~11 MB on disk; load it once and reuse it across requests.
_vectorstore = None


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
