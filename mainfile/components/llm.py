from langchain_groq import ChatGroq

from mainfile.config.config import (
    GROQ_API_KEY,
    GROQ_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)
from mainfile.common.logger import get_logger
from mainfile.common.custom_exception import CustomException

logger = get_logger(__name__)


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
