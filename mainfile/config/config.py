import os
from dotenv import load_dotenv

# Load .env here so config values are populated no matter which module is
# imported first. app.py imports the chain before calling load_dotenv(), which
# previously left these constants as None.
load_dotenv()

HF_TOKEN = os.environ.get("HF_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
DB_FAISS_PATH = "vectorstore/db_faiss"
DATA_PATH = "data/"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Chat model served by Groq. The old llama-3.1-8b-instant was decommissioned
# and now returns a 404 model_not_found.
GROQ_MODEL_NAME = os.environ.get("GROQ_MODEL_NAME", "openai/gpt-oss-20b")
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1024"))
RETRIEVER_K = int(os.environ.get("RETRIEVER_K", "3"))

# --- Sessions and chat history -------------------------------------------
# Keep the signing key stable across restarts; a random key per boot would
# invalidate every existing session cookie on every restart.
SECRET_KEY = os.environ.get("SECRET_KEY")

SESSION_STORE_DIR = os.environ.get("SESSION_STORE_DIR", ".flask_session")
CHAT_STORE_DIR = os.environ.get("CHAT_STORE_DIR", ".chat_store")

# Max distinct chats/sessions kept on disk before the cache prunes old ones.
CHAT_STORE_THRESHOLD = int(os.environ.get("CHAT_STORE_THRESHOLD", "500"))
# Idle lifetime of a stored transcript, in seconds (default 7 days).
CHAT_TTL_SECONDS = int(os.environ.get("CHAT_TTL_SECONDS", str(7 * 24 * 3600)))
# Most recent messages retained per chat (user + assistant each count as one).
MAX_HISTORY_MESSAGES = int(os.environ.get("MAX_HISTORY_MESSAGES", "50"))

# --- Multimodal (image) retrieval ----------------------------------------
FIGURES_DIR = os.environ.get("FIGURES_DIR", "data/figures")
FIGURES_META = os.path.join(FIGURES_DIR, "figures.json")
IMAGE_DB_PATH = os.environ.get("IMAGE_DB_PATH", "vectorstore/db_images")
CAPTION_DB_PATH = os.environ.get("CAPTION_DB_PATH", "vectorstore/db_captions")

# CLIP puts images and text in one space, so a text question can be matched
# directly against figure pixels.
CLIP_MODEL_NAME = os.environ.get("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")

# Skip logos/rules/decorative strips when harvesting figures.
MIN_IMAGE_WIDTH = int(os.environ.get("MIN_IMAGE_WIDTH", "150"))
MIN_IMAGE_HEIGHT = int(os.environ.get("MIN_IMAGE_HEIGHT", "150"))

# How many figures to show, and how relevant one must be to appear at all.
IMAGE_TOP_K = int(os.environ.get("IMAGE_TOP_K", "2"))
# Calibrated on this corpus: relevant figures score 0.30+, medical questions
# with no matching figure ~0.26, non-medical ~0.21. 0.28 separates them.
IMAGE_MIN_SCORE = float(os.environ.get("IMAGE_MIN_SCORE", "0.28"))
IMAGE_CANDIDATES = int(os.environ.get("IMAGE_CANDIDATES", "8"))

# Groq model that can actually see an image. Verified vision-capable.
VISION_MODEL_NAME = os.environ.get("VISION_MODEL_NAME", "qwen/qwen3.8-27b")
VISION_MAX_TOKENS = int(os.environ.get("VISION_MAX_TOKENS", "320"))

# Conversational memory: prior turns fed to the question condenser.
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "6"))

