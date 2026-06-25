from __future__ import annotations

import json
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from config.settings import BASE_DIR, CHAT_HISTORY_DIR

LEGACY_CHAT_HISTORY_FILE = BASE_DIR / "chat_sessions.json"
CHAT_HISTORY_FILE = CHAT_HISTORY_DIR / "chat_sessions.json"
_LOCK = threading.Lock()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_message(message: Dict[str, Any]) -> Dict[str, Any]:
    safe = deepcopy(message) if isinstance(message, dict) else {}
    safe["role"] = safe.get("role", "assistant")
    safe["content"] = safe.get("content", "")
    return safe


def _safe_list(values: Any) -> List[Dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [deepcopy(value) for value in values if isinstance(value, dict)]


def _is_hidden_session(session: Dict[str, Any]) -> bool:
    return session.get("id") == "__file_tracking__" or session.get("kind") == "system"


def _short_title(text: str, limit: int = 42) -> str:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return "New Chat"
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "..."


def _normalize_session(session: Dict[str, Any]) -> Dict[str, Any]:
    normalized = deepcopy(session) if isinstance(session, dict) else {}
    messages = [_safe_message(msg) for msg in normalized.get("messages", [])]
    turns = _safe_list(normalized.get("turns", []))

    created_at = normalized.get("created_at") or _now_iso()
    updated_at = normalized.get("updated_at") or created_at

    normalized["id"] = normalized.get("id") or uuid.uuid4().hex
    normalized["title"] = normalized.get("title") or "New Chat"
    normalized["created_at"] = created_at
    normalized["updated_at"] = updated_at
    normalized["messages"] = messages
    normalized["turns"] = turns
    normalized.pop("embedding_metrics", None)
    return normalized


def _sort_sessions(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(sessions, key=lambda s: s.get("updated_at", ""), reverse=True)


def _load_payload() -> List[Dict[str, Any]]:
    source_file = CHAT_HISTORY_FILE if CHAT_HISTORY_FILE.is_file() else LEGACY_CHAT_HISTORY_FILE
    if not source_file.is_file():
        return []

    try:
        with open(source_file, "r", encoding="utf-8") as file:
            payload = json.load(file)
    except Exception:
        return []

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        sessions = payload.get("sessions")
        if isinstance(sessions, list):
            return sessions
        sessions = payload.get("chat_sessions")
        if isinstance(sessions, list):
            return sessions
    return []


def _load_all_sessions() -> List[Dict[str, Any]]:
    return _sort_sessions([_normalize_session(session) for session in _load_payload() if isinstance(session, dict)])


def _write_sessions(sessions: List[Dict[str, Any]]) -> None:
    CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    tmp_file = CHAT_HISTORY_FILE.with_suffix(".tmp")
    with open(tmp_file, "w", encoding="utf-8") as file:
        json.dump(_sort_sessions(sessions), file, indent=2, ensure_ascii=True)
    tmp_file.replace(CHAT_HISTORY_FILE)


def load_chat_sessions() -> List[Dict[str, Any]]:
    with _LOCK:
        return [session for session in _load_all_sessions() if not _is_hidden_session(session)]


def load_all_chat_sessions() -> List[Dict[str, Any]]:
    with _LOCK:
        return _load_all_sessions()


def save_chat_sessions(sessions: List[Dict[str, Any]]) -> None:
    with _LOCK:
        normalized = [_normalize_session(session) for session in sessions if not _is_hidden_session(session)]
        _write_sessions(normalized)


def create_chat_session(title: str = "New Chat") -> Dict[str, Any]:
    sessions = load_chat_sessions()
    timestamp = _now_iso()
    session = {
        "id": uuid.uuid4().hex,
        "title": title,
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
        "turns": [],
    }
    sessions.insert(0, session)
    save_chat_sessions(sessions)
    return session


def rename_chat_session(session_id: str, title: str) -> Optional[Dict[str, Any]]:
    cleaned_title = " ".join(title.strip().split())
    if not cleaned_title:
        return None

    sessions = load_all_chat_sessions()
    target = None
    for session in sessions:
        if session.get("id") == session_id:
            target = session
            break

    if target is None or _is_hidden_session(target):
        return None

    target["title"] = cleaned_title
    target["updated_at"] = _now_iso()
    save_chat_sessions(sessions)
    return target


def get_chat_session(session_id: str) -> Optional[Dict[str, Any]]:
    for session in load_chat_sessions():
        if session.get("id") == session_id:
            return session
    return None


def upsert_chat_message(
    session_id: str,
    role: str,
    content: str,
    sources: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    sessions = load_all_chat_sessions()
    target = None

    for session in sessions:
        if session.get("id") == session_id:
            target = session
            break

    if target is None:
        timestamp = _now_iso()
        target = {
            "id": session_id,
            "title": "New Chat",
            "created_at": timestamp,
            "updated_at": timestamp,
            "messages": [],
            "turns": [],
        }
        sessions.insert(0, target)

    message = {"role": role, "content": content}
    if sources is not None:
        message["sources"] = deepcopy(sources)
    if metadata:
        message.update(deepcopy(metadata))
    target["messages"].append(message)
    target["updated_at"] = _now_iso()

    if role == "user" and target.get("title", "New Chat") in {"New Chat", "Untitled Chat"}:
        target["title"] = _short_title(content)

    query_id = message.get("query_id")
    if query_id:
        turns = target.setdefault("turns", [])
        turn = None
        for existing_turn in turns:
            if existing_turn.get("query_id") == query_id:
                turn = existing_turn
                break

        if turn is None:
            turn = {
                "query_id": query_id,
                "session_id": target["id"],
            }
            turns.append(turn)

        turn["session_id"] = target["id"]
        if role == "user":
            turn["user_query"] = content
            if "query_timestamp" in message:
                turn["query_timestamp"] = message["query_timestamp"]
        elif role == "assistant":
            turn["response"] = content
            if "user_query" in message and "user_query" not in turn:
                turn["user_query"] = message["user_query"]
            if "response_metrics" in message:
                turn["response_metrics"] = deepcopy(message["response_metrics"])

    save_chat_sessions(sessions)
    return target


def delete_chat_session(session_id: str) -> List[Dict[str, Any]]:
    sessions = [session for session in load_chat_sessions() if session.get("id") != session_id]
    save_chat_sessions(sessions)
    return sessions
