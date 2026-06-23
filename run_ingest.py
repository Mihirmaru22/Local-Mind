# localmind/run_ingest.py
import os
import sys

# Ensure the root directory is in the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ingestion.ingest import sync_ingestion

if __name__ == "__main__":
    print(" Starting LocalMind Ingestion Pipeline...")
    print(" Scanning the /pdfs directory...")
    total_chunks = sync_ingestion()
    print(f" Ingestion Complete! Indexed {total_chunks} chunks.")
    print(" You can now start the API with: uvicorn interfaces.api:app --reload")
