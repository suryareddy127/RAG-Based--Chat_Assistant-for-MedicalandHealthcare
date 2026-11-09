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
