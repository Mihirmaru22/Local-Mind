# Uncommitted Changes v1

This file summarizes the current workspace changes that have not been committed yet.

## Current uncommitted files

- `.gitignore`
- `app.py`
- `config/settings.py`
- `ingestion/ingest.py`
- `rag_core.py`
- `observability/file_tracker.py`
- `chat_history/`

## Summary of changes

### 1. Chat history support
- Added persistent chat history storage under `chat_history/`.
- New chat threads are created with their own chat id.
- Switching chats reloads the correct conversation.
- New Chat now opens directly without the callback popup.
- Assistant message rendering hides internal `<think>...</think>` reasoning while preserving the answer format requested by the user.

### 2. Ingestion tracking and logs
- Added `observability/file_tracker.py` to build and write chunk tracking data.
- Added `logs/chunked_files.json` generation for ingestion runs.
- The tracking file is created immediately and updated while PDFs are processed.
- Each tracked file records extracted character count and chunks created.
- Console output now shows per-file ingestion progress.

### 3. Shared path configuration
- Updated `config/settings.py` with shared `LOGS_DIR` and `CHAT_HISTORY_DIR` paths.

### 4. Legacy and Streamlit alignment
- Updated both `ingestion/ingest.py` and `rag_core.py` so log generation works from modular ingestion and legacy/Streamlit startup paths.
- Updated `app.py` so the UI uses the new chat history location and keeps current working behavior intact.

## Notes
- These changes are still local and uncommitted.
- `logs/chunked_files.json` and `chat_history/chat_sessions.json` are runtime data files and may change as the app runs.
