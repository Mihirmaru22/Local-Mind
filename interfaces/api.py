# interfaces/api.py
import os
import re
import time
import json
import shutil
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import httpx
import psutil

from config.settings import (
    PDF_DIR, DB_DIR, MAIN_MODEL, UTILITY_MODEL, 
    EMBEDDING_MODEL, OLLAMA_HOST, USER_ID
)
from retrieval.retriever import get_retriever
from retrieval.vectorstore import get_vectorstore
from retrieval.embeddings import get_embeddings
from llm.generator import get_rag_chain
from llm.query_rewriter import get_rewrite_chain
from llm.prompt import RAG_PROMPT
from memory.mem0_manager import save_memory_background

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("LocalMind.API")

# --- FastAPI App ---
app = FastAPI(
    title="LocalMind API",
    description="Enterprise-grade local RAG pipeline",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    prompt: str
    chat_history: List[dict] = []
    user_id: str = USER_ID
    model_name: Optional[str] = None

class DocumentInfo(BaseModel):
    filename: str
    status: str
    size_mb: float

# --- Helper Functions ---
def format_docs(docs):
    """Format retrieved documents into XML-tagged context."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "N/A")
        doc_type = doc.metadata.get("type", "text")
        doc_format = doc.metadata.get("format", "unknown")
        formatted.append(
            f"<document type='{doc_type}' format='{doc_format}' "
            f"source='{source}' page='{page}'>\n{doc.page_content}\n</document>"
        )
    return "\n\n".join(formatted)

# --- Endpoints ---

@app.get("/")
async def root():
    return {"status": "LocalMind API is running", "version": "3.0.0"}

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """Lists all indexed documents (PDFs + Markdown + TXT)"""
    files = []
    all_files = (
        list(PDF_DIR.glob("*.pdf")) + 
        list(PDF_DIR.glob("*.md")) + 
        list(PDF_DIR.glob("*.txt"))
    )
    
    for f in sorted(all_files, key=lambda x: x.name):
        try:
            files.append(DocumentInfo(
                filename=f.name,
                status="Indexed",
                size_mb=round(f.stat().st_size / (1024 * 1024), 2)
            ))
        except Exception as e:
            logger.warning(f"Could not read {f.name}: {e}")
            continue
            
    return files

@app.get("/metrics")
async def system_metrics():
    """Return real-time system performance metrics."""
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
    }

@app.get("/models")
async def get_available_models():
    """Fetches available models directly from the local Ollama instance."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
            if resp.status_code == 200:
                data = resp.json()
                return [
                    {
                        "name": m["name"], 
                        "size_gb": round(m.get("size", 0) / (1024**3), 2),
                        "modified": m.get("modified_at", "unknown")
                    } 
                    for m in data.get("models", [])
                ]
    except Exception as e:
        logger.error(f"Failed to fetch models from Ollama: {e}")
    return []

@app.get("/stats")
async def get_pipeline_stats():
    """Fetches real-time vector database statistics."""
    try:
        vs = get_vectorstore()
        count = vs._collection.count()
        return {
            "total_chunks": count,
            "embedding_model": EMBEDDING_MODEL,
            "vector_db": "ChromaDB (Local Persistent)",
            "retrieval_strategy": "MMR (k=4, fetch_k=20)",
            "corpus_directory": str(PDF_DIR)
        }
    except Exception as e:
        logger.error(f"Failed to fetch pipeline stats: {e}")
        return {
            "total_chunks": 0, 
            "embedding_model": "Unknown", 
            "vector_db": "Disconnected",
            "retrieval_strategy": "N/A",
            "corpus_directory": str(PDF_DIR)
        }

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Handles file uploads securely via the API."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    allowed_extensions = {".pdf", ".md", ".txt"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type. Allowed: {allowed_extensions}"
        )
    
    try:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        file_path = PDF_DIR / file.filename
        
        # Prevent overwriting
        counter = 1
        while file_path.exists():
            stem = Path(file.filename).stem
            file_path = PDF_DIR / f"{stem}_{counter}{file_ext}"
            counter += 1
            
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)
        logger.info(f"✅ Uploaded {file_path.name} ({size_mb} MB)")
        
        return {
            "status": "success", 
            "filename": file_path.name,
            "size_mb": size_mb,
            "message": "Run `python run_ingest.py` to index this document."
        }
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def stream_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Main RAG query endpoint with streaming response and pipeline stage logging.
    Implements 90% query rewriter bypass for optimal latency.
    """
    logger.info("="*60)
    logger.info(f"🔍 NEW QUERY: '{request.prompt[:50]}...'")
    logger.info(f"🧠 Model: {request.model_name or MAIN_MODEL}")
    logger.info(f"📜 Chat History: {len(request.chat_history)} messages")
    
    # === PHASE 1: PREPARE HISTORY ===
    history_str = "\n".join([
        f"{m['role']}: {m['content']}" 
        for m in request.chat_history[-6:]
    ]) if request.chat_history else "None"

    # === PHASE 2: SMART BYPASS LOGIC (90% Bypass) ===
    is_follow_up = bool(request.chat_history)
    should_rewrite = is_follow_up and (random.random() < 0.10)

    if should_rewrite:
        logger.info("🔄 STAGE 1: Query Rewriting (phi4-mini)...")
        rewrite_start = time.time()
        try:
            rewrite_chain = get_rewrite_chain()
            standalone_q = await rewrite_chain.ainvoke({
                "chat_history": history_str,
                "question": request.prompt
            })
            rewrite_time = time.time() - rewrite_start
            logger.info(f"✅ Rewritten: '{standalone_q}' ({rewrite_time:.2f}s)")
        except Exception as e:
            logger.warning(f"⚠️ Rewriter failed, using raw prompt: {e}")
            standalone_q = request.prompt
    else:
        standalone_q = request.prompt
        if is_follow_up:
            logger.info("⏭️ STAGE 1: Query Rewriting BYPASSED (90% logic)")
        else:
            logger.info("⏭️ STAGE 1: Query Rewriting BYPASSED (standalone query)")

    # === PHASE 3: RETRIEVAL ===
    logger.info("🔄 STAGE 2: Embedding & Retrieval (MMR)...")
    retrieval_start = time.time()
    
    try:
        retriever = get_retriever()
        docs = await retriever.ainvoke(standalone_q)
        retrieval_time = time.time() - retrieval_start
        logger.info(f"✅ Retrieved {len(docs)} chunks in {retrieval_time:.2f}s")
    except Exception as e:
        logger.error(f" Retrieval failed: {e}")
        docs = []
        retrieval_time = time.time() - retrieval_start
        
    context = format_docs(docs) if docs else "No relevant documents found."

    # === PHASE 4: MEMORY (DISABLED) ===
    memory_context = "No long-term memories active."

    # === PHASE 5: STREAMING GENERATION ===
    target_model = request.model_name or MAIN_MODEL
    logger.info(f" STAGE 3: Generation ({target_model})...")
    generation_start = time.time()
    first_token_time = None

    async def generate_stream():
        nonlocal first_token_time
        full_response = ""
        
        try:
            chain = get_rag_chain(target_model)
            
            async for token in chain.astream({
                "question": request.prompt,
                "context": context,
                "memory": memory_context,
                "chat_history": history_str
            }):
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - generation_start
                    logger.info(f"🚀 First Token (TTFT): {ttft:.2f}s")
                    
                full_response += token
                yield token
            
            generation_time = time.time() - generation_start
            total_time = time.time() - retrieval_start  # From retrieval start
            
            logger.info(f"✅ Generation complete in {generation_time:.2f}s")
            logger.info(f"📊 PIPELINE SUMMARY:")
            logger.info(f"   ├─ Rewriting:   {'BYPASSED' if not should_rewrite else f'{rewrite_time:.2f}s'}")
            logger.info(f"   ├─ Retrieval:   {retrieval_time:.2f}s ({len(docs)} chunks)")
            logger.info(f"   ├─ Generation:  {generation_time:.2f}s")
            logger.info(f"   └─ TOTAL:       {total_time:.2f}s")
            logger.info("="*60)
            
            # Background memory save (disabled)
            # background_tasks.add_task(
            #     save_memory_background,
            #     f"User: {request.prompt}\nAI: {full_response}",
            #     request.user_id
            # )
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            yield f"Error: {str(e)}"

    return StreamingResponse(generate_stream(), media_type="text/plain")

# --- Run directly (for testing) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)