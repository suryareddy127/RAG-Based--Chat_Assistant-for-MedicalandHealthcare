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
