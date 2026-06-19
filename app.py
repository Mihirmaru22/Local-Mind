import streamlit as st
import time
import threading
import os
from copy import deepcopy
import ollama

# Import the new conversational components from rag_core
from rag_core import (
    retriever, rewrite_chain, m, USER_ID, MAIN_MODEL,
    get_current_performance_metrics, format_docs, get_rag_chain
)
from chat_history import (
    create_chat_session,
    delete_chat_session,
    get_chat_session,
    load_chat_sessions,
    upsert_chat_message,
)


# --- BACKGROUND MEMORY SAVER ---
def save_memory_in_background(text, user_id):
    """Saves memory silently without blocking the Streamlit UI."""
    try:
        m.add(text, user_id=user_id)
    except Exception:
        pass


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


# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Enterprise RAG HUD", layout="wide", page_icon="🏢")

_ensure_chat_state()

# --- SIDEBAR: THE "HUD" ---
with st.sidebar:
    st.title(" LOCAL-MIND Control Center")
    st.divider()

    st.subheader(" MAIN GENERATION MODEL")
    st.caption("This model writes the final answer. Pick your heaviest model.")

    # Fetch local models dynamically from Ollama
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    try:
        client = ollama.Client(host=ollama_host)
        models_response = client.list()
        available_models = [m['model'] for m in models_response.get('models', [])]
    except Exception as e:
        st.error(f"Ollama connection failed: {e}")
        available_models = [MAIN_MODEL]

    if not available_models:
        st.warning("No local models found! Run `ollama pull <model>` in your terminal.")
        available_models = [MAIN_MODEL]

    # Default the dropdown to the MAIN_MODEL
    default_index = available_models.index(MAIN_MODEL) if MAIN_MODEL in available_models else 0

    selected_model = st.selectbox(
        "Select Main Model",
        available_models,
        index=default_index,
        key="main_gen_model"
    )

    st.divider()

    st.subheader(" System Telemetry")
    perf = get_current_performance_metrics()

    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="CPU Usage", value=f"{perf['cpu_percent']:.1f}%")
    with col2:
        st.metric(label="RAM Usage", value=f"{perf['memory_percent']:.1f}%")

    st.progress(perf['memory_percent'] / 100)
    st.caption(f"Used: {perf['memory_used_gb']:.2f} GB | Swap: {perf['swap_percent']:.1f}%")

    st.divider()

    st.subheader(" Chat History")
    st.caption("Pick a saved chat or start a new thread.")
    chat_ids = [chat["id"] for chat in st.session_state.chat_sessions]
    selected_chat_id = st.selectbox(
        "Saved chats",
        chat_ids,
        index=_active_chat_index(),
        format_func=lambda chat_id: _chat_label(get_chat_session(chat_id) or {"title": "New Chat", "messages": []}),
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

    show_debug = st.checkbox(" Show Retrieved Chunks", value=True, key="debug_chunks_checkbox")

# --- MAIN CHAT INTERFACE ---
st.title(" Enterprise Compliance Analyst")
st.caption("Powered by Streaming RAG, Local Rewriting & Mem0 Memory")
st.caption(f"Current thread: `{st.session_state.active_chat_id[:8]}`")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(_strip_thinking_content(message["content"]))
        if "sources" in message and message["sources"] and show_debug:
            with st.expander(" View Retrieved Sources"):
                for i, src in enumerate(message["sources"]):
                    st.markdown(f"**[Rank {i+1}]** {src['source']} (Page {src['page']})")
                    st.code(src['preview'], language="markdown")

if prompt := st.chat_input("Ask a compliance or financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    upsert_chat_message(st.session_state.active_chat_id, "user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(" Thinking & Retrieving..."):
            start_time = time.time()

            try:
                # A. FORMAT HISTORY
                history_messages = st.session_state.messages[:-1]
                chat_history_str = "\n".join([
                    f"{msg['role'].capitalize()}: {msg['content']}"
                    for msg in history_messages[-6:]
                ]) if history_messages else "No previous conversation."

                # B. REWRITE VAGUE QUESTIONS (Runs locally & instantly)
                if history_messages:
                    standalone_question = rewrite_chain.invoke({
                        "chat_history": chat_history_str,
                        "question": prompt
                    }).strip()
                else:
                    standalone_question = prompt

                # C. SEARCH MEMORY
                search_results = m.search(prompt, filters={"user_id": USER_ID})
                if isinstance(search_results, dict) and 'results' in search_results:
                    relevant_memories = search_results['results']
                elif isinstance(search_results, list):
                    relevant_memories = search_results
                else:
                    relevant_memories = []

                memory_context = "\n".join([mem.get('memory', str(mem)) for mem in relevant_memories]) if relevant_memories else "No relevant past memories found."

                # D. RETRIEVE DOCUMENTS
                docs = retriever.invoke(standalone_question)
                context = format_docs(docs)

                # E. BUILD DYNAMIC CHAIN BASED ON UI DROPDOWN
                dynamic_rag_chain = get_rag_chain(selected_model)

                # F. GENERATE RESPONSE (STREAMING)
                response_stream = dynamic_rag_chain.stream({
                    "question": prompt,
                    "context": context,
                    "memory": memory_context,
                    "chat_history": chat_history_str
                })

                # Stream only the visible answer to the UI.
                full_response = st.write_stream(_stream_visible_response(response_stream))
                full_response = _strip_thinking_content(full_response)

                latency = time.time() - start_time

                # G. SAVE TO MEMORY IN BACKGROUND (Fixes the UI freeze)
                conversation_text = f"User: {prompt}\nAssistant: {full_response}"
                threading.Thread(
                    target=save_memory_in_background,
                    args=(conversation_text, USER_ID),
                    daemon=True
                ).start()

                # H. FETCH SOURCES FOR DEBUG
                sources = []
                if show_debug:
                    for doc in docs:
                        sources.append({
                            "source": doc.metadata.get("source", "Unknown"),
                            "page": doc.metadata.get("page", "N/A"),
                            "preview": doc.page_content[:200] + "..."
                        })

                st.caption(f"⏱️ Latency: {latency:.2f}s | 🔍 Search: '{standalone_question}' | 🧠 Model: `{selected_model}`")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "sources": sources
                })
                upsert_chat_message(
                    st.session_state.active_chat_id,
                    "assistant",
                    full_response,
                    sources=sources,
                )
                st.session_state.chat_sessions = load_chat_sessions()

            except Exception as e:
                st.error(f" Pipeline Error: {e}")
