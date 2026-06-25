import sys
import os
import time
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import ollama
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from chat_history.store import (
    create_chat_session,
    delete_chat_session,
    get_chat_session,
    load_chat_sessions,
    rename_chat_session,
    upsert_chat_message,
)
from config.settings import MAIN_MODEL
from ingestion.ingest import sync_ingestion
from llm.generator import get_rag_chain
from llm.query_rewriter import get_rewrite_chain
from retrieval.retriever import get_retriever


def format_docs(docs):
    return "\n\n".join(
        [
            f"<doc src='{d.metadata.get('source')}' p='{d.metadata.get('page')}'>{d.page_content}</doc>"
            for d in docs
        ]
    )


def _ensure_chat_state():
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = load_chat_sessions()

    if not st.session_state.chat_sessions:
        st.session_state.chat_sessions = [create_chat_session()]

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = st.session_state.chat_sessions[0]["id"]

    active_ids = {chat["id"] for chat in st.session_state.chat_sessions}
    if st.session_state.active_chat_id not in active_ids:
        st.session_state.active_chat_id = st.session_state.chat_sessions[0]["id"]

    active_chat = get_chat_session(st.session_state.active_chat_id)
    if "messages" not in st.session_state or active_chat is None:
        st.session_state.messages = deepcopy(active_chat["messages"]) if active_chat else []
    elif active_chat and st.session_state.messages != active_chat["messages"]:
        st.session_state.messages = deepcopy(active_chat["messages"])


def _active_chat_index():
    ids = [chat["id"] for chat in st.session_state.chat_sessions]
    try:
        return ids.index(st.session_state.active_chat_id)
    except ValueError:
        return 0


def _chat_label(chat):
    title = chat.get("title", "New Chat")
    messages = chat.get("messages", [])
    user_turns = sum(1 for msg in messages if msg.get("role") == "user")
    updated = chat.get("updated_at", "")
    if updated:
        updated = updated.replace("T", " ")[:16]
        return f"{title} · {user_turns} turns · {updated}"
    return f"{title} · {user_turns} turns"


def _set_active_chat(chat_id):
    chat = get_chat_session(chat_id)
    if chat is None:
        return
    st.session_state.active_chat_id = chat_id
    st.session_state.messages = deepcopy(chat.get("messages", []))


def _create_new_chat():
    chat = create_chat_session()
    st.session_state.chat_sessions = load_chat_sessions()
    st.session_state.active_chat_id = chat["id"]
    st.session_state.messages = []
    st.rerun()


def _clear_current_chat():
    current_id = st.session_state.get("active_chat_id")
    if not current_id:
        return
    sessions = delete_chat_session(current_id)
    if not sessions:
        sessions = [create_chat_session()]
    st.session_state.chat_sessions = sessions
    st.session_state.active_chat_id = sessions[0]["id"]
    st.session_state.messages = []
    st.rerun()


def _set_active_chat(chat_id):
    chat = get_chat_session(chat_id)
    if chat is None:
        return
    st.session_state.active_chat_id = chat_id
    st.session_state.messages = deepcopy(chat.get("messages", []))
    st.session_state.chat_action_chat_id = None
    st.session_state.chat_rename_chat_id = None


def _toggle_chat_menu(chat_id):
    current = st.session_state.get("chat_action_chat_id")
    st.session_state.chat_action_chat_id = None if current == chat_id else chat_id
    if st.session_state.chat_action_chat_id != chat_id:
        st.session_state.chat_rename_chat_id = None


def _begin_rename_chat(chat_id):
    chat = get_chat_session(chat_id)
    st.session_state.chat_action_chat_id = chat_id
    st.session_state.chat_rename_chat_id = chat_id
    st.session_state[f"rename_title_{chat_id}"] = (chat or {}).get("title", "New Chat")


def _save_renamed_chat(chat_id):
    new_title = st.session_state.get(f"rename_title_{chat_id}", "").strip()
    if not new_title:
        return
    rename_chat_session(chat_id, new_title)
    st.session_state.chat_sessions = load_chat_sessions()
    st.session_state.chat_rename_chat_id = None
    st.session_state.chat_action_chat_id = None
    st.rerun()


def _delete_chat_and_select(chat_id):
    sessions = delete_chat_session(chat_id)
    if not sessions:
        sessions = [create_chat_session()]
    st.session_state.chat_sessions = sessions
    st.session_state.active_chat_id = sessions[0]["id"]
    active_chat = get_chat_session(st.session_state.active_chat_id)
    st.session_state.messages = deepcopy(active_chat.get("messages", [])) if active_chat else []
    st.session_state.chat_action_chat_id = None
    st.session_state.chat_rename_chat_id = None
    st.rerun()


def _chat_menu_label(chat):
    title = chat.get("title", "New Chat")
    messages = chat.get("messages", [])
    user_turns = sum(1 for msg in messages if msg.get("role") == "user")
    updated = chat.get("updated_at", "")
    if updated:
        updated = updated.replace("T", " ")[:16]
        return f"{title} · {user_turns} turns · {updated}"
    return f"{title} · {user_turns} turns"


def _ensure_chat_history_ui_state():
    if "chat_history_expanded" not in st.session_state:
        st.session_state.chat_history_expanded = False


def _visible_chat_sessions():
    sessions = st.session_state.chat_sessions
    if st.session_state.get("chat_history_expanded") or len(sessions) <= 8:
        return sessions

    visible = sessions[:8]
    active_id = st.session_state.get("active_chat_id")
    if active_id and not any(chat.get("id") == active_id for chat in visible):
        active_chat = next((chat for chat in sessions if chat.get("id") == active_id), None)
        if active_chat is not None:
            visible = visible[:-1] + [active_chat]
    return visible


def _inject_sidebar_styles():
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(10, 14, 24, 0.98) 0%, rgba(13, 18, 30, 0.99) 100%);
        }

        section[data-testid="stSidebar"] div.stButton > button {
            width: 100%;
            text-align: left;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.06);
            background: rgba(255, 255, 255, 0.03);
            color: rgba(245, 247, 250, 0.96);
            padding: 0.62rem 0.78rem;
            min-height: 2.65rem;
            box-shadow: none;
            font-weight: 500;
        }

        section[data-testid="stSidebar"] div.stButton > button:hover {
            background: rgba(255, 255, 255, 0.07);
            border-color: rgba(255, 255, 255, 0.14);
        }

        section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
            background: linear-gradient(180deg, rgba(66, 84, 128, 0.92) 0%, rgba(42, 56, 92, 0.92) 100%);
            border-color: rgba(150, 173, 255, 0.36);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.08), 0 0 0 1px rgba(91, 113, 168, 0.16);
            color: #f8fbff;
        }

        section[data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
            background: linear-gradient(180deg, rgba(76, 95, 144, 0.98) 0%, rgba(49, 65, 106, 0.98) 100%);
            border-color: rgba(170, 188, 255, 0.42);
        }

        section[data-testid="stSidebar"] div.stButton > button:focus-visible {
            outline: 2px solid rgba(160, 182, 255, 0.55);
            box-shadow: none;
        }

        section[data-testid="stSidebar"] .chat-history-row {
            margin-bottom: 0.2rem;
        }

        section[data-testid="stSidebar"] .chat-history-meta {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_chat_history_sidebar():
    if st.button("+ New Chat", use_container_width=True):
        _create_new_chat()

    visible_sessions = _visible_chat_sessions()

    for chat in visible_sessions:
        chat_id = chat["id"]
        is_active = chat_id == st.session_state.active_chat_id
        is_menu_open = st.session_state.get("chat_action_chat_id") == chat_id
        is_renaming = st.session_state.get("chat_rename_chat_id") == chat_id
        title = chat.get("title", "New Chat")

        row_cols = st.columns([0.86, 0.14], gap="small")
        with row_cols[0]:
            if st.button(
                f"• {title}" if is_active else title,
                key=f"open_chat_{chat_id}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                _set_active_chat(chat_id)
                st.rerun()
        with row_cols[1]:
            if st.button("⋮", key=f"menu_chat_{chat_id}", use_container_width=True):
                _toggle_chat_menu(chat_id)
                st.rerun()

        if is_menu_open:
            action_cols = st.columns(2)
            with action_cols[0]:
                if st.button("Rename", key=f"rename_action_{chat_id}", use_container_width=True):
                    _begin_rename_chat(chat_id)
                    st.rerun()
            with action_cols[1]:
                if st.button("Delete", key=f"delete_action_{chat_id}", use_container_width=True):
                    _delete_chat_and_select(chat_id)

        if is_renaming:
            st.text_input(
                "Rename chat",
                key=f"rename_title_{chat_id}",
                label_visibility="collapsed",
            )
            save_cols = st.columns(2)
            with save_cols[0]:
                if st.button("Save", key=f"save_rename_{chat_id}", use_container_width=True):
                    _save_renamed_chat(chat_id)
            with save_cols[1]:
                if st.button("Cancel", key=f"cancel_rename_{chat_id}", use_container_width=True):
                    st.session_state.chat_rename_chat_id = None
                    st.session_state.chat_action_chat_id = None
                    st.rerun()

    if len(st.session_state.chat_sessions) > 8:
        if st.session_state.get("chat_history_expanded"):
            if st.button("Show less history", use_container_width=True, key="collapse_chat_history"):
                st.session_state.chat_history_expanded = False
                st.rerun()
        else:
            remaining = len(st.session_state.chat_sessions) - len(visible_sessions)
            if st.button(f"Show more history (+{remaining})", use_container_width=True, key="expand_chat_history"):
                st.session_state.chat_history_expanded = True
                st.rerun()


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _ensure_index():
    if st.session_state.get("index_checked"):
        return
    with st.spinner("Checking PDF index..."):
        try:
            added_chunks = sync_ingestion()
            st.session_state.index_added_chunks = added_chunks
            st.session_state.index_status = "ready"
        except Exception as exc:
            st.session_state.index_status = "error"
            st.session_state.index_error = str(exc)
        finally:
            st.session_state.index_checked = True


def _strip_thinking_content(text):
    visible_parts = []
    cursor = 0
    in_think_block = False

    while cursor < len(text):
        if not in_think_block:
            think_start = text.find("<think>", cursor)
            if think_start == -1:
                tail = text[cursor:]
                partial_start = tail.find("<think")
                if partial_start != -1:
                    visible_parts.append(tail[:partial_start])
                else:
                    visible_parts.append(tail)
                break

            visible_parts.append(text[cursor:think_start])
            cursor = think_start + len("<think>")
            in_think_block = True
        else:
            think_end = text.find("</think>", cursor)
            if think_end == -1:
                break
            cursor = think_end + len("</think>")
            in_think_block = False

    return "".join(visible_parts)


def _stream_visible_response(response_stream):
    raw_response = ""
    visible_response = ""

    for chunk in response_stream:
        raw_response += chunk
        next_visible = _strip_thinking_content(raw_response)
        if len(next_visible) > len(visible_response):
            delta = next_visible[len(visible_response):]
            visible_response = next_visible
            if delta:
                yield delta


st.set_page_config(page_title="LocalMind | Enterprise AI", layout="wide", page_icon="🏢")
_ensure_chat_state()
_ensure_index()
_inject_sidebar_styles()

with st.sidebar:
    st.title("🏢 LOCAL-MIND")
    st.caption("Enterprise Intelligence Engine")
    st.divider()

    st.subheader("MAIN GENERATION MODEL")
    st.caption("This model writes the final answer.")

    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        client = ollama.Client(host=ollama_host)
        models_response = client.list()
        available_models = [m["model"] for m in models_response.get("models", [])]
    except Exception as exc:
        st.warning(f"Could not load local models: {exc}")
        available_models = [MAIN_MODEL]

    if not available_models:
        available_models = [MAIN_MODEL]

    default_index = available_models.index(MAIN_MODEL) if MAIN_MODEL in available_models else 0
    selected_model = st.selectbox(
        "Select Main Model",
        available_models,
        index=default_index,
        key="main_gen_model",
    )

    st.divider()

    st.subheader("💬 Chat History")
    _ensure_chat_history_ui_state()
    _render_chat_history_sidebar()

    col_clear = st.columns(1)[0]
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            _clear_current_chat()

    st.divider()

    st.subheader("📊 System Telemetry")
    st.success("UI Ready")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="CPU", value="--")
    with col2:
        st.metric(label="RAM", value="--")

    st.divider()

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        _clear_current_chat()

st.title("💬 Enterprise Compliance Analyst")
st.caption("Powered by LocalMind RAG Engine")
st.caption(f"Current thread: `{st.session_state.active_chat_id[:8]}`")
st.caption(f"Main model: `{selected_model}`")

for message in st.session_state.messages:
    display_name = "Mihir" if message["role"] == "user" else "Local Mind"
    avatar = "👤" if message["role"] == "user" else "🏢"

    with st.chat_message(display_name, avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    query_id = uuid.uuid4().hex
    query_timestamp = _utc_now_iso()
    st.session_state.messages.append({"role": "user", "content": prompt})
    upsert_chat_message(
        st.session_state.active_chat_id,
        "user",
        prompt,
        metadata={
            "session_id": st.session_state.active_chat_id,
            "query_id": query_id,
            "query_timestamp": query_timestamp,
        },
    )
    with st.chat_message("Mihir", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("Local Mind", avatar="🏢"):
        placeholder = st.empty()
        full_response = ""
        start_time = time.time()
        ttft = None

        try:
            with st.spinner("Thinking & Retrieving..."):
                history_messages = st.session_state.messages[:-1]
                chat_history_str = "\n".join([
                    f"{msg['role'].capitalize()}: {msg['content']}"
                    for msg in history_messages[-6:]
                ]) if history_messages else "No previous conversation."

                standalone_question = prompt
                if history_messages:
                    rewrite_chain = get_rewrite_chain()
                    standalone_question = rewrite_chain.invoke({
                        "chat_history": chat_history_str,
                        "question": prompt,
                    }).strip() or prompt

                retriever = get_retriever()
                docs = retriever.invoke(standalone_question)
                context = format_docs(docs)
                memory_context = "No long-term memories active. All answers are grounded solely in retrieved documents."

                dynamic_rag_chain = get_rag_chain(selected_model)
                response_start_time = _utc_now_iso()
                response_perf_start = time.perf_counter()
                response_stream = dynamic_rag_chain.stream({
                    "question": prompt,
                    "context": context,
                    "memory": memory_context,
                    "chat_history": chat_history_str,
                })

                full_response = st.write_stream(_stream_visible_response(response_stream))
                full_response = _strip_thinking_content(full_response)
                response_end_time = _utc_now_iso()
                response_latency_ms = int((time.perf_counter() - response_perf_start) * 1000)
                latency = time.time() - start_time
                st.caption(f"⏱️ Latency: {latency:.2f}s | 🔍 Search: '{standalone_question}' | 🧠 Model: `{selected_model}`")

        except Exception as e:
            st.error(f"Pipeline Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    upsert_chat_message(
        st.session_state.active_chat_id,
        "assistant",
        full_response,
        metadata={
            "session_id": st.session_state.active_chat_id,
            "query_id": query_id,
            "user_query": prompt,
            "response_metrics": {
                "response_start_time": response_start_time,
                "response_end_time": response_end_time,
                "response_latency_ms": response_latency_ms,
                "model_name": selected_model,
            },
        },
    )
    st.session_state.chat_sessions = load_chat_sessions()
