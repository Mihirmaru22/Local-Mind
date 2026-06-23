from __future__ import annotations

import json
from pathlib import Path

from config.settings import DB_DIR, PDF_DIR
from observability.file_tracker import (
    CHUNK_TRACKING_FILE,
    build_chunk_tracking_entry,
    initialize_chunk_tracking_report,
    save_chunk_tracking_report,
)
from retrieval.vectorstore import get_vectorstore

from .chunker import get_text_splitter
from .parser import hybrid_pdf_parser


def _load_tracked_entries() -> list[dict]:
    if not CHUNK_TRACKING_FILE.is_file():
        return []

    try:
        data = json.loads(CHUNK_TRACKING_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []

    files = data.get("files", []) if isinstance(data, dict) else []
    return files if isinstance(files, list) else []


def _tracked_file_names(entries: list[dict]) -> set[str]:
    names = set()
    for entry in entries:
        if isinstance(entry, dict):
            file_name = entry.get("file_name")
            if file_name:
                names.add(file_name)
    return names


def _ingest_files(pdf_files: list[Path], existing_entries: list[dict]) -> int:
    vectorstore = get_vectorstore()
    text_splitter = get_text_splitter()
    total_chunks = 0
    tracked_files = list(existing_entries)

    if not tracked_files:
        initialize_chunk_tracking_report()
        print("Tracking report initialized at logs/chunked_files.json")

    for idx, file_path in enumerate(pdf_files, start=1):
        print(f"Processing [{idx}/{len(pdf_files)}]: {file_path.name}")
        try:
            docs = hybrid_pdf_parser(file_path)
            processed_docs = []
            for doc in docs:
                if doc.metadata["type"] == "text":
                    processed_docs.extend(text_splitter.split_documents([doc]))
                else:
                    processed_docs.append(doc)

            vectorstore.add_documents(processed_docs)
            total_chunks += len(processed_docs)

            tracked_files.append(build_chunk_tracking_entry(file_path, docs, processed_docs))
            save_chunk_tracking_report(tracked_files, status="in_progress")

            print(
                f"Indexed {file_path.name} | chars={tracked_files[-1]['characters_extracted']} | chunks={tracked_files[-1]['chunks_created']}"
            )
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    save_chunk_tracking_report(tracked_files, status="completed")
    print("Chunk tracking saved to logs/chunked_files.json")
    print(f"Ingestion Complete! Total chunks: {total_chunks}")
    return total_chunks


def sync_ingestion() -> int:
    print("Starting LocalMind ingestion sync...")
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise ValueError("No PDFs found.")

    db_missing = (not DB_DIR.exists()) or (not any(DB_DIR.iterdir()) if DB_DIR.exists() else True)
    existing_entries = _load_tracked_entries()

    # If the vector DB is missing, rebuild the full index so Chroma is actually created.
    if db_missing:
        print("Chroma DB is missing or empty. Rebuilding the full index...")
        return _ingest_files(pdf_files, [])

    tracked_names = _tracked_file_names(existing_entries)
    missing_files = [file_path for file_path in pdf_files if file_path.name not in tracked_names]

    if not missing_files:
        if existing_entries:
            save_chunk_tracking_report(existing_entries, status="completed")
            print("All PDFs are already tracked. chunked_files.json refreshed.")
        else:
            initialize_chunk_tracking_report()
            print("No existing chunk tracking found. Initialized logs/chunked_files.json")
        return 0

    print(f"Found {len(missing_files)} untracked PDF(s). Indexing them now...")
    return _ingest_files(missing_files, existing_entries)


# Backward-compatible name used by run_ingest.py and any older imports.
run_ingestion = sync_ingestion
