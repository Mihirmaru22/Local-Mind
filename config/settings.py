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
MAIN_MODEL = os.getenv("MAIN_MODEL", "qwen3.5:9b")
UTILITY_MODEL = "phi4-mini:latest"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

# Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Mem0
USER_ID = "mihirmaru"
