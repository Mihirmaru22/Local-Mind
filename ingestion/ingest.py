# ingestion/ingest.py
from pathlib import Path
from config.settings import PDF_DIR, DB_DIR
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from retrieval.embeddings import get_embeddings

# Import both parsers
from .parser import hybrid_pdf_parser, markdown_parser

def get_text_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)

def run_ingestion():
    print("🚀 Starting LocalMind Ingestion Pipeline...")
    print(f"📂 Scanning the {PDF_DIR} directory...")
    
    # 1. Glob for BOTH PDFs and Markdown files
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    md_files = list(PDF_DIR.glob("*.md"))
    all_files = pdf_files + md_files
    
    if not all_files:
        raise ValueError("No PDFs or Markdown files found in the data directory.")
        
    vectorstore = Chroma(persist_directory=str(DB_DIR), embedding_function=get_embeddings())
    text_splitter = get_text_splitter()
    
    total_chunks = 0
    
    print("Starting Batched Ingestion...")
    for file_path in all_files:
        try:
            # 2. ROUTING LOGIC: Send to the correct parser based on extension
            ext = file_path.suffix.lower()
            if ext == '.pdf':
                docs = hybrid_pdf_parser(file_path)
            elif ext == '.md':
                docs = markdown_parser(file_path)
            else:
                continue  # Skip unsupported files
                
            # 3. Chunking Logic
            processed_docs = []
            for doc in docs:
                # Split text, but keep tables intact
                if doc.metadata.get("type") == "text":
                    processed_docs.extend(text_splitter.split_documents([doc]))
                else:
                    processed_docs.append(doc) 
                    
            vectorstore.add_documents(processed_docs)
            total_chunks += len(processed_docs)
            print(f"✅ Indexed {file_path.name} ({ext})")
            
        except Exception as e:
            print(f"⚠️ Skipping {file_path.name}: {e}")
            
    print(f"🎉 Ingestion Complete! Indexed {total_chunks} total chunks.")
    return total_chunks