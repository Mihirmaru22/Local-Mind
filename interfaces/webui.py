# interfaces/webui.py
import os
import re
import time
import json
import streamlit as st
import httpx

# --- 1. DYNAMIC CONFIGURATION ---
API_URL = os.getenv("LOCALMIND_API_URL", "http://localhost:8000")

# --- 2. ENTERPRISE CSS ---
st.set_page_config(
    page_title="LocalMind | Enterprise AI", 
    layout="wide", 
    page_icon="",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Global */
    .block-container { padding-top: 2rem; }
    body { font-family: 'Inter', sans-serif; }
    
    /* Source Cards */
    .source-card {
        background-color: #f8f9fa; 
        border-radius: 8px; 
        padding: 12px 15px; 
        margin-bottom: 10px; 
        border-left: 4px solid #0d6efd; 
        box-shadow: 0 1px 3px rgba(0,0,0,0.08); 
        transition: transform 0.2s;
    }
    .source-card:hover { 
        transform: translateX(3px); 
        background-color: #e9ecef; 
    }
    
    /* Progress Bars */
    .stProgress > div > div > div > div { 
        background-color: #0d6efd; 
    }
    
    /* Stat Boxes */
    .stat-box {
        background-color: #ffffff; 
        border: 2px solid #dee2e6; 
        border-radius: 8px; 
        padding: 15px; 
        text-align: center; 
        margin-bottom: 10px;
    }
    .stat-box .stat-value {
        color: #0d6efd !important;
        font-size: 1.4em;
        font-weight: bold;
        display: block;
        margin-bottom: 4px;
    }
    .stat-box .stat-label {
        color: #6c757d !important;
        font-size: 0.8em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Chat Messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Status Indicators */
    .status-processing {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
    .status-complete {
        background-color: #d1e7dd;
        border: 1px solid #198754;
        border-radius: 8px;
        padding: 10px 15px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. HELPER FUNCTIONS ---
def api_get(endpoint):
    """Centralized GET request handler."""
    try:
        resp = httpx.get(f"{API_URL}{endpoint}", timeout=5.0)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def api_post(endpoint, **kwargs):
    """Centralized POST request handler."""
    try:
        resp = httpx.post(f"{API_URL}{endpoint}", timeout=kwargs.get("timeout", 30.0), **kwargs)
        return resp
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

# --- 4. SESSION STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
if "pipeline_status" not in st.session_state:
    st.session_state.pipeline_status = None

# --- 5. SIDEBAR: CONTROL PLANE ---
with st.sidebar:
    st.title("🏢 LOCAL-MIND")
    st.caption("Enterprise Intelligence Engine v3.0")
    st.divider()
    
    # A. DYNAMIC MODEL CONFIGURATION
    st.subheader("⚙️ Inference Engine")
    models_data = api_get("/models")
    
    if models_data:
        model_names = [m["name"] for m in models_data]
        # Try to select default model
        default_idx = 0
        for i, m in enumerate(model_names):
            if "qwen3.5:9b" in m or "qwen2.5:7b" in m:
                default_idx = i
                break
        
        selected_model = st.selectbox(
            "Select Active Model", 
            model_names, 
            index=default_idx
        )
        st.caption(f"🟢 Connected to Ollama ({len(models_data)} models)")
        
        # Show model sizes
        with st.expander("📊 Model Details"):
            for m in models_data:
                st.text(f"• {m['name']} ({m['size_gb']} GB)")
    else:
        st.error("❌ Backend Offline")
        selected_model = "offline-model"
        st.caption("Start FastAPI: `uvicorn interfaces.api:app`")
        
    st.divider()
    
    # B. KNOWLEDGE BASE MANAGER
    st.subheader("📂 Knowledge Base")
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        help="Max 200MB per file"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            with st.spinner(f"Uploading {file.name}..."):
                files = {"file": (file.name, file.getvalue(), file.type or "application/octet-stream")}
                resp = httpx.post(f"{API_URL}/upload", files=files, timeout=60.0)
                if resp.status_code == 200:
                    st.toast(f"✅ {file.name}", icon="✅")
                else:
                    st.toast(f"❌ {file.name}: {resp.text}", icon="❌")
        st.info("💡 Run `python run_ingest.py` to index new documents.")

    # Active Documents List
    docs = api_get("/documents")
    if docs:
        st.caption(f"**Indexed Documents:** {len(docs)}")
        for doc in docs:
            icon = "📄" if doc["filename"].endswith(".pdf") else "📝"
            st.markdown(f"{icon} **{doc['filename']}**")
            st.caption(f"   {doc['size_mb']} MB")
    else:
        st.warning("No documents indexed.")
            
    st.divider()
    
    # C. LIVE PIPELINE TELEMETRY
    st.subheader("📊 System Telemetry")
    stats = api_get("/stats")
    perf = api_get("/metrics")
    
    col1, col2 = st.columns(2)
    with col1:
        cpu = perf["cpu_percent"] if perf else 0
        st.metric("CPU", f"{cpu:.1f}%")
        st.progress(min(cpu / 100.0, 1.0))
    with col2:
        ram = perf["memory_percent"] if perf else 0
        st.metric("RAM", f"{ram:.1f}%")
        st.progress(min(ram / 100.0, 1.0))
        
    if stats:
        st.divider()
        st.caption("**Pipeline Architecture**")
        
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-value">{stats['total_chunks']}</span>
            <span class="stat-label">Vector Chunks</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-value" style="font-size: 0.9em;">{stats['embedding_model']}</span>
            <span class="stat-label">Embedding Model</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="stat-box">
            <span class="stat-value" style="font-size: 0.9em;">ChromaDB</span>
            <span class="stat-label">Vector Database</span>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # D. SESSION ACTIONS
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.session_state.pipeline_status = None
            st.rerun()
    with col2:
        if st.button("💾 Export Audit", use_container_width=True):
            chat_export = json.dumps(st.session_state.messages, indent=2)
            st.download_button(
                "Download JSON", 
                chat_export, 
                "localmind_audit.json", 
                "application/json"
            )

# --- 6. MAIN INTERFACE ---
st.title("💬 Enterprise Compliance Analyst")
st.caption("Air-gapped RAG Engine | Zero data leaves this machine.")

chat_col, source_col = st.columns([3, 1.2])

# LEFT: CHAT ENGINE
with chat_col:
    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🏢"):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a compliance, financial, or legal question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🏢"):
            status_placeholder = st.empty()
            placeholder = st.empty()
            full_response = ""
            
            # Show initial status
            status_placeholder.markdown(
                '<div class="status-processing">🔄 <b>Processing:</b> Embedding query & searching vector database...</div>', 
                unsafe_allow_html=True
            )
            
            payload = {
                "prompt": prompt,
                "chat_history": st.session_state.messages[:-1],
                "user_id": os.getenv("USER_ID", "enterprise_user"),
                "model_name": selected_model 
            }
            
            start_time = time.time()
            ttft = None
            
            try:
                with httpx.stream("POST", f"{API_URL}/query", json=payload, timeout=300) as response:
                    if response.status_code == 200:
                        # Update status once streaming starts
                        status_placeholder.markdown(
                            '<div class="status-processing">🔄 <b>Processing:</b> Generating response (CPU inference may take 30-90s)...</div>', 
                            unsafe_allow_html=True
                        )
                        
                        for chunk in response.iter_text():
                            if chunk:
                                if ttft is None: 
                                    ttft = time.time() - start_time
                                    # Update status on first token
                                    status_placeholder.markdown(
                                        '<div class="status-processing"> <b>Streaming:</b> First token received. Generating response...</div>', 
                                        unsafe_allow_html=True
                                    )
                                full_response += chunk
                                placeholder.markdown(full_response + "▌")
                        
                        placeholder.markdown(full_response)
                        
                        # Final status
                        total_time = time.time() - start_time
                        status_placeholder.markdown(
                            f'<div class="status-complete">✅ <b>Complete</b> in {total_time:.1f}s | TTFT: {ttft:.2f}s | Model: {selected_model}</div>', 
                            unsafe_allow_html=True
                        )
                        
                        # Parse Citations
                        citations = re.findall(r'\[Source:\s*(.*?),\s*Page:\s*(\d+)\]', full_response)
                        st.session_state.last_sources = list(set(citations))
                        
                    else:
                        status_placeholder.error(f"❌ API Error: {response.status_code}")
                        
            except httpx.ReadTimeout:
                status_placeholder.error("⏳ Request timed out (5 min limit).")
            except Exception as e:
                status_placeholder.error(f"❌ Connection error: {str(e)}")
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        st.rerun()

# RIGHT: VERIFIED SOURCES & SECURITY
with source_col:
    st.subheader(" Verified Citations")
    st.caption("Grounding sources for the latest response.")
    
    if st.session_state.last_sources:
        for source, page in st.session_state.last_sources:
            st.markdown(f"""
            <div class="source-card">
                <strong>📄 {source}</strong><br>
                <span style="color: #6c757d; font-size: 0.9em;">Page: {page}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Ask a question to see verified source citations here.")
        
    st.divider()
    
    st.subheader("🛡️ Security Posture")
    st.success("🔒 **Air-Gapped Mode Active**")
    st.caption("All processing, embedding, and inference occur locally. Zero external API telemetry.")
    
    st.divider()
    
    st.subheader("ℹ️ About LocalMind")
    st.caption("""
    **Architecture:** RAG with MMR Retrieval  
    **Models:** Qwen 3.5 (9B), Phi-4 Mini  
    **Embeddings:** Nomic Embed Text  
    **Vector DB:** ChromaDB (Local)  
    **License:** Apache 2.0
    """)