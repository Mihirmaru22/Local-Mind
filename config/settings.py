# config/settings.py
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
PDF_DIR = BASE_DIR / "pdfs"
DB_DIR = BASE_DIR / "chroma_db_local"
LOGS_DIR = BASE_DIR / "logs"
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"

# Models
MAIN_MODEL = os.getenv("MAIN_MODEL", "qwen2.5:7b")
UTILITY_MODEL = "qwen3:1.7b"
EMBEDDING_MODEL = "nomic-embed-text"

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Mem0
USER_ID = "mihirmaru"
