import sys
import time
from copy import deepcopy
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

with st.sidebar:
    st.title("🏢 LOCAL-MIND")
    st.caption("Enterprise Intelligence Engine")
    st.divider()

    st.subheader("MAIN GENERATION MODEL")
    st.caption("This model writes the final answer.")

    ollama_host = st.text_input("Ollama Host", value="http://localhost:11434")
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
    st.caption("Pick a saved chat or start a new thread.")
    chat_ids = [chat["id"] for chat in st.session_state.chat_sessions]
    selected_chat_id = st.selectbox(
        "Saved chats",
        chat_ids,
        index=_active_chat_index(),
        format_func=lambda chat_id: _chat_label(
            get_chat_session(chat_id) or {"title": "New Chat", "messages": []}
        ),
        key="chat_history_selector",
    )

    if selected_chat_id != st.session_state.active_chat_id:
        _set_active_chat(selected_chat_id)
        st.rerun()

    col_new, col_clear = st.columns(2)
    with col_new:
        if st.button("+ New Chat", use_container_width=True):
            _create_new_chat()
    with col_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            _clear_current_chat()

    st.caption(f"Active chat ID: `{st.session_state.active_chat_id[:8]}`")
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    upsert_chat_message(st.session_state.active_chat_id, "user", prompt)
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
                response_stream = dynamic_rag_chain.stream({
                    "question": prompt,
                    "context": context,
                    "memory": memory_context,
                    "chat_history": chat_history_str,
                })

                full_response = st.write_stream(_stream_visible_response(response_stream))
                full_response = _strip_thinking_content(full_response)
                latency = time.time() - start_time
                st.caption(f"⏱️ Latency: {latency:.2f}s | 🔍 Search: '{standalone_question}' | 🧠 Model: `{selected_model}`")

        except Exception as e:
            st.error(f"Pipeline Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
    upsert_chat_message(st.session_state.active_chat_id, "assistant", full_response)
    st.session_state.chat_sessions = load_chat_sessions()
