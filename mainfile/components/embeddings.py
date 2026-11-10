"""Embedding models used by the assistant.

Two different spaces are needed:

* MiniLM for prose - the encyclopedia text chunks and the figure captions.
* CLIP for figures - it places images and text in one shared space, so a
  typed question can be scored directly against figure pixels.
"""

import os

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from mainfile.config.config import CLIP_MODEL_NAME, FIGURES_DIR
from mainfile.common.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Loading these models takes several seconds, so keep one instance of each.
_embeddings = None
_clip_model = None
_clip_processor = None


# --- text embeddings ------------------------------------------------------

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


# --- CLIP image/text embeddings -------------------------------------------

def get_clip_model():
    """Load CLIP once; it is a few hundred MB.

    Uses transformers directly rather than the sentence-transformers wrapper,
    whose clip-ViT-B-32 package does not expose weights in the layout recent
    transformers versions expect.
    """
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor

    from transformers import CLIPModel, CLIPProcessor

    logger.info(f"Loading CLIP model '{CLIP_MODEL_NAME}'...")
    _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    _clip_model.eval()
    _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    logger.info("CLIP model loaded.")
    return _clip_model, _clip_processor


def _normalise(tensor):
    """Unit-length rows, so a dot product is a cosine similarity."""
    return (tensor / tensor.norm(p=2, dim=-1, keepdim=True)).cpu().numpy()


class ClipEmbeddings(Embeddings):
    """CLIP exposed as a LangChain Embeddings implementation.

    embed_documents() receives figure *filenames* rather than prose, because
    the documents being indexed here are images.
    """

    def __init__(self, figures_dir=FIGURES_DIR):
        self.figures_dir = figures_dir

    def _encode_images(self, images):
        import torch

        model, processor = get_clip_model()
        out = []
        for start in range(0, len(images), 16):
            batch = images[start:start + 16]
            inputs = processor(images=batch, return_tensors="pt")
            with torch.no_grad():
                feats = model.get_image_features(**inputs)
            out.extend(_normalise(feats).tolist())
        return out

    def _encode_text(self, texts):
        import torch

        model, processor = get_clip_model()
        inputs = processor(text=texts, return_tensors="pt",
                           padding=True, truncation=True, max_length=77)
        with torch.no_grad():
            feats = model.get_text_features(**inputs)
        return _normalise(feats).tolist()

    def embed_documents(self, texts):
        """`texts` are figure filenames; encode the images they point at."""
        from PIL import Image

        images = []
        for name in texts:
            path = name if os.path.isabs(name) else os.path.join(self.figures_dir, name)
            images.append(Image.open(path).convert("RGB"))
        try:
            return self._encode_images(images)
        finally:
            for image in images:
                image.close()

    def embed_query(self, text):
        """A user question, encoded into the same space as the figures."""
        return self._encode_text([text])[0]
