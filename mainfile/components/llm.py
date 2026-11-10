"""Groq models used by the assistant.

Two are needed, because the text model cannot see:

* a chat model for written answers, via LangChain; and
* a vision-capable model that is sent a figure inline and asked about it.

Prompt wording lives in retreive.py; this module only owns the model clients.
"""

import base64
import io

from langchain_groq import ChatGroq

from mainfile.config.config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    VISION_MAX_TOKENS,
    VISION_MODEL_NAME,
)
from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException

logger = get_logger(__name__)

MAX_IMAGE_EDGE = 768  # keep the upload small; figures need no more detail

_vision_client = None


def load_llm(model_name: str = None, groq_api_key: str = None):
    # Resolved at call time, not at def time, so a late load_dotenv() still works.
    model_name = model_name or GROQ_MODEL_NAME
    groq_api_key = groq_api_key or GROQ_API_KEY

    try:
        if not groq_api_key:
            raise CustomException(
                "GROQ_API_KEY is not set. Add it to your .env file at the project root."
            )

        logger.info(f"Loading LLM from Groq using model '{model_name}'...")

        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

        logger.info("LLM loaded successfully from Groq.")
        return llm

    except Exception as e:
        error_message = CustomException("Failed to load an LLM from Groq", e)
        logger.error(str(error_message))
        return None


# --- vision ---------------------------------------------------------------

def load_vision_client():
    """Raw Groq client; LangChain's ChatGroq has no image message helper."""
    global _vision_client

    if _vision_client is None:
        from groq import Groq

        if not GROQ_API_KEY:
            raise CustomException("GROQ_API_KEY is not set")
        logger.info(f"Loading vision model '{VISION_MODEL_NAME}' from Groq...")
        _vision_client = Groq(api_key=GROQ_API_KEY)

    return _vision_client


def image_to_data_url(path):
    """Downscale and base64-encode a figure for inline upload."""
    from PIL import Image

    with Image.open(path) as image:
        image = image.convert("RGB")
        if max(image.size) > MAX_IMAGE_EDGE:
            ratio = MAX_IMAGE_EDGE / max(image.size)
            image = image.resize(
                (max(1, int(image.width * ratio)), max(1, int(image.height * ratio))),
                Image.LANCZOS,
            )
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)

    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/jpeg;base64,{encoded}"


def ask_about_image(prompt, image_path):
    """Send one figure plus `prompt` to the vision model; '' on failure."""
    try:
        response = load_vision_client().chat.completions.create(
            model=VISION_MODEL_NAME,
            max_tokens=VISION_MAX_TOKENS,
            temperature=0.2,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url",
                     "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }],
        )
        return (response.choices[0].message.content or "").strip()

    except Exception as e:
        logger.error(str(CustomException("Vision model call failed", e)))
        return ""
