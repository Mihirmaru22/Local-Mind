import os
import gc
import csv
import json
import time
import platform
import psutil
import pymupdf as fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from mem0 import Memory
from operator import itemgetter
from observability.file_tracker import build_chunk_tracking_entry, initialize_chunk_tracking_report, save_chunk_tracking_report

# Configure Mem0 to use local Ollama (Prevents OpenAI API key errors)
from mem0 import Memory
import os

# Since we are running locally, force the host to your local Mac's Ollama
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ==============================================================================
# THE TWO BRAINS (HARDCODED TO YOUR LOCAL OLLAMA LIST)
# ==============================================================================

# 1. THE MAIN BRAIN (Generates the final text)
# Pick your heaviest local model. You have minimax-m3:cloud and qwen3:4b. 
MAIN_MODEL = os.getenv("MAIN_MODEL", "minimax-m3:cloud")

# 2. THE UTILITY BRAIN (Rewrites vague queries & extracts Mem0 JSON)
# This needs to be tiny and fast. You have phi4-mini:latest. Perfect.
UTILITY_MODEL = "phi4-mini:latest"
# Force a completely fresh local database with correct 768 dimensions
mem0_config = {
    "llm": {
        "provider": "ollama",
        "config": {
            "model": UTILITY_MODEL, # <--- MUST BE phi4-mini:latest
            "temperature": 0,
        }
    },
    "embedder": {
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text", # <--- From your ollama list
        }
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": "mem0_ollama_fresh_v1", # Brand new collection name
            "path": "./mem0_qdrant_data",             # Brand new local folder
            "embedding_model_dims": 768               # Explicitly force 768 dimensions
        }
    }
}
m = Memory.from_config(mem0_config)
USER_ID = "mihirmaru"


# Try to import GPU monitoring
try:
    import GPUtil
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("  GPUtil not installed. GPU monitoring disabled. Install with: pip install GPUtil")

# ==============================================================================
# 1. CONFIGURATION (Hardware & Model Agnostic)
# ==============================================================================
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfs"
DB_DIR = BASE_DIR / "chroma_db_local"
METRICS_FILE = BASE_DIR / "rag_metrics_log.csv"
BENCHMARK_FILE = BASE_DIR / "benchmark.json"
SYSTEM_PERF_FILE = BASE_DIR / "system_performance_log.csv"
QUERY_PERF_FILE = BASE_DIR / "query_performance_log.csv"

print(f"PDF Directory: {PDF_DIR}")
print(f"Database Directory: {DB_DIR}")
MODEL_NAME = MAIN_MODEL
print(f"Initializing Pipeline with Model: {MODEL_NAME}")

embeddings = OllamaEmbeddings(model="nomic-embed-text")

llm = ChatOllama(
    model=MODEL_NAME,
    temperature=0,
    num_ctx=8192,
    keep_alive="5m",
    base_url=OLLAMA_HOST
)

# ==============================================================================
# 6.5 QUERY REWRITER (Fast Local Model to avoid cloud API delays)
# ==============================================================================
rewrite_prompt = ChatPromptTemplate.from_template("""
Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:""")

# FIX: Use the UTILITY brain for rewriting, NOT the main brain
rewrite_llm = ChatOllama(model=UTILITY_MODEL, temperature=0, base_url=OLLAMA_HOST)
rewrite_chain = rewrite_prompt | rewrite_llm | StrOutputParser()
# Metrics tracking variables
metrics = {
    "num_pdfs": 0,
    "num_chunks": 0,
    "ingestion_time": 0.0,
    "total_query_time": 0.0,
    "query_count": 0
}

# ==============================================================================
# 2. SYSTEM PERFORMANCE MONITORING
# ==============================================================================

def get_system_info():
    """Collect comprehensive system information."""
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "available_ram_gb": round(psutil.virtual_memory().available / (1024**3), 2),
        "disk_usage_percent": psutil.disk_usage('/').percent,
    }
    
    if HAS_GPU:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info["gpu_name"] = gpu.name
                info["gpu_memory_total_mb"] = gpu.memoryTotal
                info["gpu_memory_used_mb"] = gpu.memoryUsed
                info["gpu_utilization_percent"] = round(gpu.load * 100, 2)
        except Exception as e:
            info["gpu_error"] = str(e)
    
    return info

def get_current_performance_metrics():
    """Get real-time performance metrics."""
    metrics_snapshot = {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory_percent": psutil.virtual_memory().percent,
        "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
        "swap_percent": psutil.swap_memory().percent if psutil.swap_memory().total > 0 else 0,
    }
    
    if HAS_GPU:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                metrics_snapshot["gpu_utilization_percent"] = round(gpu.load * 100, 2)
                metrics_snapshot["gpu_memory_used_mb"] = gpu.memoryUsed
                metrics_snapshot["gpu_temperature_celsius"] = gpu.temperature
        except Exception:
            pass
    
    return metrics_snapshot

def log_system_performance(phase="general"):
    """Log system performance during different phases."""
    perf_metrics = get_current_performance_metrics()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n System Performance [{phase}]:")
    print(f"   CPU Usage: {perf_metrics['cpu_percent']:.1f}%")
    print(f"   RAM Usage: {perf_metrics['memory_percent']:.1f}% ({perf_metrics['memory_used_gb']:.2f} GB)")
    if 'gpu_utilization_percent' in perf_metrics:
        print(f"   GPU Usage: {perf_metrics['gpu_utilization_percent']:.1f}%")
        print(f"   GPU Memory: {perf_metrics['gpu_memory_used_mb']} MB")
        if 'gpu_temperature_celsius' in perf_metrics:
            print(f"   GPU Temp: {perf_metrics['gpu_temperature_celsius']}°C")
    
    # Save to CSV
    save_performance_to_csv(timestamp, phase, perf_metrics)
    
    return perf_metrics

def save_performance_to_csv(timestamp, phase, perf_metrics):
    """Append performance metrics to a separate CSV file."""
    file_exists = SYSTEM_PERF_FILE.is_file()
    
    with open(SYSTEM_PERF_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            headers = ["Timestamp", "Phase", "CPU_Percent", "RAM_Percent", 
                      "RAM_Used_GB", "Swap_Percent"]
            if HAS_GPU:
                headers.extend(["GPU_Utilization_Percent", "GPU_Memory_MB", 
                               "GPU_Temperature_C"])
            writer.writerow(headers)
        
        row = [timestamp, phase, 
               f"{perf_metrics['cpu_percent']:.2f}",
               f"{perf_metrics['memory_percent']:.2f}",
               f"{perf_metrics['memory_used_gb']:.2f}",
               f"{perf_metrics['swap_percent']:.2f}"]
        
        if HAS_GPU and 'gpu_utilization_percent' in perf_metrics:
            row.extend([
                f"{perf_metrics['gpu_utilization_percent']:.2f}",
                perf_metrics['gpu_memory_used_mb'],
                perf_metrics.get('gpu_temperature_celsius', 'N/A')
            ])
        
        writer.writerow(row)

def save_query_performance_to_csv(question_preview, latency, pre_perf, post_perf):
    """Save individual query performance metrics."""
    file_exists = QUERY_PERF_FILE.is_file()
    
    with open(QUERY_PERF_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            headers = ["Timestamp", "Question_Preview", "Latency_Sec",
                      "Pre_CPU_Percent", "Post_CPU_Percent",
                      "Pre_RAM_GB", "Post_RAM_GB"]
            if HAS_GPU:
                headers.extend(["Pre_GPU_Percent", "Post_GPU_Percent"])
            writer.writerow(headers)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = [timestamp, question_preview, f"{latency:.2f}",
               f"{pre_perf['cpu_percent']:.2f}", f"{post_perf['cpu_percent']:.2f}",
               f"{pre_perf['memory_used_gb']:.2f}", f"{post_perf['memory_used_gb']:.2f}"]
        
        if HAS_GPU and 'gpu_utilization_percent' in pre_perf:
            row.extend([
                f"{pre_perf['gpu_utilization_percent']:.2f}",
                f"{post_perf['gpu_utilization_percent']:.2f}"
            ])
        
        writer.writerow(row)

# ==============================================================================
# 3. CSV LOGGING FUNCTION
# ==============================================================================
def save_metrics_to_csv():
    file_exists = METRICS_FILE.is_file()
    avg_latency = metrics["total_query_time"] / metrics["query_count"] if metrics["query_count"] > 0 else 0.0
    
    with open(METRICS_FILE, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            writer.writerow([
                "Timestamp", "Model_Name", "Num_PDFs_Processed", "Total_Chunks_Created", 
                "Ingestion_Time_Sec", "Avg_Query_Latency_Sec", "Total_Queries_Asked", "DB_Path"
            ])
            
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            MODEL_NAME,
            metrics["num_pdfs"],
            metrics["num_chunks"],
            f"{metrics['ingestion_time']:.2f}",
            f"{avg_latency:.2f}",
            metrics["query_count"],
            str(DB_DIR)
        ])
    print(f"\n Session metrics successfully saved to '{METRICS_FILE.name}'")

# ==============================================================================
# 4. HYBRID PARSER (Text + Tabular Markdown)
# ==============================================================================
def convert_table_to_markdown(table_data):
    if not table_data or not table_data[0]: 
        return ""
    clean_data = [[str(cell).replace('\n', ' ').strip() if cell else "" for cell in row] for row in table_data]
    header = "| " + " | ".join(clean_data[0]) + " |"
    separator = "| " + " | ".join(["---" for _ in clean_data[0]]) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in clean_data[1:]]
    return "\n".join([header, separator] + rows)

def hybrid_pdf_parser(file_path):
    documents = []
    page_texts = {} 
    
    pdf_text = fitz.open(file_path)
    for page_num, page in enumerate(pdf_text):
        text = page.get_text("text").strip()
        page_texts[page_num] = text 
        if text:
            documents.append(Document(
                page_content=text,
                metadata={"source": file_path.name, "page": page_num + 1, "type": "text"}
            ))
            
    with pdfplumber.open(file_path) as pdf_tables:
        for page_num, page in enumerate(pdf_tables.pages):
            tables = page.extract_tables()
            for table in tables:
                md_table = convert_table_to_markdown(table)
                if md_table:
                    context_header = f"Context: {page_texts.get(page_num, '')[:300]}\n\n" if page_texts.get(page_num) else ""
                    documents.append(Document(
                        page_content=context_header + md_table,
                        metadata={"source": file_path.name, "page": page_num + 1, "type": "table"}
                    ))
    return documents

# ==============================================================================
# 5. INGESTION
# ==============================================================================
def ingest_documents():
    start_time = time.time() 
    
    # Log system info at start
    system_info = get_system_info()
    print("\n  System Information:")
    for key, value in system_info.items():
        print(f"   {key}: {value}")
    
    # Log performance before ingestion
    log_system_performance("pre_ingestion")
    
    if not DB_DIR.exists() or not any(DB_DIR.iterdir()):
        print(" Starting Batched Ingestion...")
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        if not pdf_files: 
            raise ValueError("No PDFs found.")
            
        vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        
        total_chunks = 0
        tracked_files = []
        initialize_chunk_tracking_report()
        print(" Tracking report initialized at logs/chunked_files.json")
        for idx, file_path in enumerate(pdf_files, start=1):
            print(f" Processing [{idx}/{len(pdf_files)}]: {file_path.name}")
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
                print(f" Indexed {file_path.name} | chars={tracked_files[-1]['characters_extracted']} | chunks={tracked_files[-1]['chunks_created']}")
                
                # Log performance every 5 files
                if idx % 5 == 0:
                    log_system_performance(f"ingestion_file_{idx}")
                
                gc.collect() 
            except Exception as e:
                print(f"  Skipping {file_path.name}: {e}")
                
        save_chunk_tracking_report(tracked_files, status="completed")
        print(" Chunk tracking saved to logs/chunked_files.json")
        metrics["num_pdfs"] = len(pdf_files)
        metrics["num_chunks"] = total_chunks
        print(" Ingestion Complete!")
    else:
        print(" Loading existing Vector DB...")
        pdf_files = list(PDF_DIR.glob("*.pdf"))
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
        tracked_files = []
        initialize_chunk_tracking_report()
        print(" Tracking report initialized at logs/chunked_files.json")
        for idx, file_path in enumerate(pdf_files, start=1):
            print(f" Processing [{idx}/{len(pdf_files)}]: {file_path.name}")
            try:
                docs = hybrid_pdf_parser(file_path)
                processed_docs = []
                for doc in docs:
                    if doc.metadata["type"] == "text":
                        processed_docs.extend(text_splitter.split_documents([doc]))
                    else:
                        processed_docs.append(doc)
                tracked_files.append(build_chunk_tracking_entry(file_path, docs, processed_docs))
                save_chunk_tracking_report(tracked_files, status="cached")
                print(f" Tracked {file_path.name} | chars={tracked_files[-1]['characters_extracted']} | chunks={tracked_files[-1]['chunks_created']}")
            except Exception as e:
                print(f"  Tracking skip {file_path.name}: {e}")
        metrics["num_pdfs"] = "N/A (Cached)"
        metrics["num_chunks"] = "N/A (Cached)"
        vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    metrics["ingestion_time"] = time.time() - start_time 
    
    # Log performance after ingestion
    log_system_performance("post_ingestion")
    
    return vectorstore

# ==============================================================================
# 6. RAG CHAIN SETUP (Updated for Chat History & Vague Questions)
# ==============================================================================
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        doc_type = doc.metadata.get('type', 'text')
        formatted.append(f"<document type='{doc_type}' source='{source}' page='{page}'>\n{doc.page_content}\n</document>")
    return "\n\n".join(formatted)

vectorstore = ingest_documents()
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 15})

# RELAXED PROMPT: Allows the bot to synthesize broad/vague questions instead of failing
prompt = ChatPromptTemplate.from_template("""/no_think
You are an AI assistant operating in a Retrieval-Augmented Generation (RAG) system.

PRIMARY OBJECTIVE

Your primary source of truth is the provided retrieved context. Use it to answer the user's question accurately, clearly, and helpfully.

CORE RULES

1. Ground answers in retrieved context.

   * Use the retrieved documents as the main source of information.
   * Extract facts, concepts, procedures, and details from the retrieved content.
   * Do not invent information that is not supported by the retrieved context.

2. Explain, don't just repeat.

   * Do not copy document text unless necessary.
   * Rewrite information in a natural, easy-to-understand way.
   * Adapt explanations to the user's level of understanding.
   * If the user asks for a beginner explanation, simplify the concepts.
   * If the user asks for a technical explanation, provide appropriate depth.

3. Use examples when helpful.

   * If the retrieved content describes a concept, process, or system, you may create illustrative examples to help understanding.
   * Examples must be clearly presented as examples and must not introduce unsupported factual claims.
   * Real-world analogies are encouraged when they improve clarity.

4. Answer the user's intent, not just the words.

   * Focus on what the user is trying to understand or accomplish.
   * User's greeting should be answered politely and enthusiastically, make user feel welcome and comfertable to ask their quetions.
   * Combine relevant information from multiple retrieved sources when appropriate.
   * Provide structured and complete answers.

5. Handle missing information honestly.

   * If the answer cannot be found in the retrieved context, say so clearly.
   * Do not guess.
   * Do not fabricate details.
   * Example:
     "I couldn't find information about that in the provided documents."

6. Separate document facts from general knowledge.

   * When a user asks something partially covered by documents and partially outside them:

     * First provide the document-supported answer.
     * Then clearly label any additional general knowledge.
   * Never present unsupported information as if it came from the documents.

7. Reduce hallucinations.

   * Never invent names, numbers, dates, specifications, requirements, procedures, policies, or technical details.
   * If uncertain, acknowledge uncertainty.
   * Prefer saying "I don't know" over making assumptions.

8. Preserve important details.

   * Keep critical numbers, configurations, requirements, versions, limitations, and constraints exactly as stated in the retrieved content.

9. Response style.

   * Be concise for simple questions.
   * Be detailed for complex questions.
   * Use bullet points, tables, and step-by-step explanations when useful.
   * Prioritize clarity and readability.
   * Complete answers are necessary, but do not add unsupported information just to be more comprehensive.
   * If the user asks multiple things in one question, answer every part.

10. Contradictory information.

    * If retrieved sources conflict:

      * Mention the conflict.
      * Explain the differing information.
      * Do not arbitrarily choose one unless evidence supports it.

11. Follow-up questions.

    * Use conversation history when available.
    * Maintain context across the conversation.
    * Ask clarifying questions when the user's request is ambiguous.

RESPONSE WORKFLOW

Step 1: Understand the user's intent.
Step 2: Search the retrieved context for relevant information.
Step 3: Determine whether the answer is fully supported, partially supported, or unsupported.
Step 4: Generate a clear and helpful response.
Step 5: If information is missing, explicitly state the limitation.
Step 6: Add examples or simplified explanations when they improve understanding.
Step 7: Verify that no unsupported claims are presented as facts.

IMPORTANT

* Retrieved documents are the primary source of truth.
* Helpful explanation is encouraged.
* Hallucination is not allowed.
* If the answer is not in the documents, say so.
* Explain concepts naturally rather than quoting documents verbatim.
* Prioritize accuracy over completeness when evidence is insufficient.

Use retrieved documents as the source of facts, but use your reasoning abilities to explain, summarize, compare, teach, and reorganize those facts in the most useful way for the user.

<memory>
{memory}
</memory>

<context>
{context}
</context>

<chat_history>
{chat_history}
</chat_history>

Question: {question}
""")

# Simplified chain: We will do retrieval manually in the loop to handle query rewriting
# DELETE THIS STATIC SHIT:
# llm = ChatOllama(model=MODEL_NAME, temperature=0, num_ctx=8192, keep_alive="5m", base_url=OLLAMA_HOST)
# rag_chain = prompt | llm | StrOutputParser()

def get_rag_chain(model_name: str = None):
    """Dynamically builds a RAG chain for the Main Generation Model."""
    # If no model is passed from the UI, default to the MAIN_MODEL
    target_model = model_name if model_name else MAIN_MODEL 
    
    dynamic_llm = ChatOllama(
        model=target_model,
        temperature=0,
        num_ctx=8192,
        keep_alive="5m",
        base_url=OLLAMA_HOST
    )
    return prompt | dynamic_llm | StrOutputParser()

# ==============================================================================
# 7. DEBUG INSPECTOR & BENCHMARK EVALUATOR
# ==============================================================================
def inspect_retrieval(question):
    """Prints the exact chunks the LLM will see to verify retrieval quality."""
    print(f"\n---  RETRIEVAL INSPECTOR for: '{question}' ---")
    docs = retriever.invoke(question)
    if not docs:
        print(" No documents retrieved!")
        return
    for i, doc in enumerate(docs):
        print(f"\n[Rank {i+1}] Source: {doc.metadata.get('source')} | Page: {doc.metadata.get('page')} | Type: {doc.metadata.get('type')}")
        print(f"Content Preview: {doc.page_content[:150]}...")
    print("--- END INSPECTOR ---\n")

def run_benchmark():
    """Runs the JSON benchmark to test retrieval accuracy."""
    if not BENCHMARK_FILE.is_file():
        print(f"\n Error: '{BENCHMARK_FILE.name}' not found. Please create it in the project root.")
        return

    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        benchmark_data = json.load(f)

    print(f"\n🧪 Starting Benchmark Evaluation ({len(benchmark_data)} questions)...")
    
    # Log performance before benchmark
    log_system_performance("pre_benchmark")
    
    correct_retrievals = 0

    for i, item in enumerate(benchmark_data):
        q = item["question"]
        expected_source = item.get("expected_source", "")
        expected_page = item.get("expected_page", None)

        docs = retriever.invoke(q)
        retrieved_sources = [doc.metadata.get('source') for doc in docs]
        retrieved_pages = [doc.metadata.get('page') for doc in docs]

        # Check if the expected source is in the top retrieved chunks
        if expected_source in retrieved_sources:
            correct_retrievals += 1
            status = "✅ PASS"
        else:
            status = " FAIL"

        print(f"[{i+1}/{len(benchmark_data)}] {status} | Q: '{q[:50]}...' | Expected: {expected_source} (p.{expected_page})")

    accuracy = (correct_retrievals / len(benchmark_data)) * 100 if benchmark_data else 0
    print(f"\n BENCHMARK RESULTS: {correct_retrievals}/{len(benchmark_data)} Correct ({accuracy:.1f}% Retrieval Accuracy)")
    print(" If accuracy < 80%, DO NOT upgrade the model. Fix your chunking/embeddings first.\n")
    
    # Log performance after benchmark
    log_system_performance("post_benchmark")

# ==============================================================================
# 8. EXECUTION LOOP (With Chat History & Query Rewriting)
# ==============================================================================
if __name__ == "__main__":
    # Log initial system state
    print("="*70)
    print(" RAG SYSTEM STARTUP")
    print("="*70)
    initial_perf = log_system_performance("startup")
    
    print("\n" + "="*70)
    print(f"  System Ready | Model: {MODEL_NAME}")
    print(" Commands: 'exit', 'benchmark', 'debug <question>', 'sysinfo', 'clear'")
    print("="*70)
    
    # Initialize Short-Term Chat History
    chat_history = []
    
    try:
        while True:
            user_input = input("\n Ask a question (or 'exit'): ").strip()
            if not user_input: 
                continue
            
            if user_input.lower() == 'exit': 
                break
            
            # Clear chat history command
            if user_input.lower() == 'clear':
                chat_history = []
                print("🧹 Chat history cleared.")
                continue
            
            # System Info Command
            if user_input.lower() == 'sysinfo':
                sys_info = get_system_info()
                current_perf = get_current_performance_metrics()
                print("\n" + "="*70)
                print("  SYSTEM INFORMATION")
                print("="*70)
                for key, value in sys_info.items():
                    print(f"   {key.replace('_', ' ').title()}: {value}")
                print("\n" + "="*70)
                print(" CURRENT PERFORMANCE")
                print("="*70)
                for key, value in current_perf.items():
                    print(f"   {key.replace('_', ' ').title()}: {value}")
                print("="*70)
                continue
            
            # Handle Benchmark Command
            if user_input.lower() == 'benchmark':
                run_benchmark()
                continue
                
            # Handle Debug Command
            if user_input.lower().startswith('debug '):
                debug_question = user_input[6:].strip()
                if debug_question:
                    inspect_retrieval(debug_question)
                continue

            # ======================================================================
            # NORMAL QUERY WITH VAGUE QUESTION HANDLING
            # ======================================================================
            print("\n Processing...")
            query_start = time.time() 
            
            # 1. FORMAT SHORT-TERM CHAT HISTORY
            chat_history_str = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in chat_history[-6:]]) if chat_history else "No previous conversation."
            
            # 2. REWRITE VAGUE QUESTIONS INTO STANDALONE QUESTIONS
            if chat_history:
                print(" Rewriting vague question for search...")
                standalone_question = rewrite_chain.invoke({
                    "chat_history": chat_history_str,
                    "question": user_input
                }).strip()
                print(f"   Search Query: '{standalone_question}'")
            else:
                standalone_question = user_input

            # 3. SEARCH LONG-TERM MEMORY (Mem0)
            print("Searching long-term memory...")
            search_results = m.search(user_input, filters={"user_id": USER_ID})
            
            if isinstance(search_results, dict) and 'results' in search_results:
                relevant_memories = search_results['results']
            elif isinstance(search_results, list):
                relevant_memories = search_results
            else:
                relevant_memories = []
            
            if relevant_memories:
                memory_context = "\n".join([mem.get('memory', str(mem)) for mem in relevant_memories])
                print(f"   Found {len(relevant_memories)} relevant memories.")
            else:
                memory_context = "No relevant past memories found."

            # 4. RETRIEVE DOCUMENTS (Using the REWRITTEN standalone question)
            docs = retriever.invoke(standalone_question)
            context = format_docs(docs)

            # 5. RUN RAG CHAIN (Using original question + context + memory + history)
                        # 5. RUN RAG CHAIN (Using original question + context + memory + history)
                        # 5. RUN RAG CHAIN (Using original question + context + memory + history)
            pre_query_perf = get_current_performance_metrics()
            
            # FIX: Build the chain dynamically for the terminal loop since we killed the global variable
            terminal_chain = get_rag_chain(MODEL_NAME)
            
            try:
                response = terminal_chain.invoke({
                    "question": user_input, 
                    "context": context,
                    "memory": memory_context,
                    "chat_history": chat_history_str
                })
                query_end = time.time() 
                
                latency = query_end - query_start
                metrics["total_query_time"] += latency
                metrics["query_count"] += 1
                
                # 6. UPDATE SHORT-TERM CHAT HISTORY
                chat_history.append({"role": "user", "content": user_input})
                chat_history.append({"role": "assistant", "content": response})
                
                # Keep history manageable (last 10 turns / 20 messages)
                if len(chat_history) > 20:
                    chat_history = chat_history[-20:]

                # 7. SAVE TO LONG-TERM MEMORY (Mem0)
                print(" Saving interaction to long-term memory...")
                conversation_text = f"User: {user_input}\nAssistant: {response}"
                m.add(conversation_text, user_id=USER_ID)
                
                # Log performance after query
                post_query_perf = get_current_performance_metrics()
                
                print(f"\n Answer:\n{response}")
                print(f"\n Latency: {latency:.2f} seconds")
                print(f"CPU: {pre_query_perf['cpu_percent']:.1f}% → {post_query_perf['cpu_percent']:.1f}%")
                print(f"RAM: {pre_query_perf['memory_used_gb']:.2f} GB → {post_query_perf['memory_used_gb']:.2f} GB")
                
                save_query_performance_to_csv(
                    user_input[:100], latency, pre_query_perf, post_query_perf
                )
                
            except Exception as e:
                print(f"\n Error: {e}")
                
    finally:
        # Final system state
        print("\n" + "="*70)
        print(" SHUTTING DOWN")
        print("="*70)
        log_system_performance("shutdown")
        save_metrics_to_csv()
        print("\n Goodbye!\n")