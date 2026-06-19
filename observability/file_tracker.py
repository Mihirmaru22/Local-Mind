from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from config.settings import LOGS_DIR

CHUNK_TRACKING_FILE = LOGS_DIR / "chunked_files.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_chunk_tracking_entry(file_path: Path, source_docs: List[Any], processed_docs: List[Any]) -> Dict[str, Any]:
    extracted_characters = sum(len(doc.page_content) for doc in source_docs)
    return {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "pages_or_sections_extracted": len(source_docs),
        "characters_extracted": extracted_characters,
        "chunks_created": len(processed_docs),
        "tracked_at": _now_iso(),
    }


def save_chunk_tracking_report(entries: List[Dict[str, Any]], status: str = "completed") -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": _now_iso(),
        "status": status,
        "file_count": len(entries),
        "total_characters_extracted": sum(entry.get("characters_extracted", 0) for entry in entries),
        "total_chunks_created": sum(entry.get("chunks_created", 0) for entry in entries),
        "files": entries,
    }
    with open(CHUNK_TRACKING_FILE, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=True)


def initialize_chunk_tracking_report() -> None:
    save_chunk_tracking_report([], status="started")
