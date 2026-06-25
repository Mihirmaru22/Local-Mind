import fitz
import pdfplumber
from pathlib import Path
from langchain_core.documents import Document

def convert_table_to_markdown(table_data):
    if not table_data or not table_data[0]:
        return ""
    clean_data = [[str(cell).replace('\n', ' ').strip() if cell else "" for cell in row] for row in table_data]
    header = "| " + " | ".join(clean_data[0]) + " |"
    separator = "| " + " | ".join(["---" for _ in clean_data[0]]) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in clean_data[1:]]
    return "\n".join([header, separator] + rows)

def hybrid_pdf_parser(file_path: Path) -> list[Document]:
    documents = []
    page_texts = {} 
    
    with fitz.open(file_path) as pdf_text:
        for page_num, page in enumerate(pdf_text):
            text = page.get_text("text").strip()
            page_texts[page_num] = text 
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": file_path.name, "page": page_num + 1, "type": "text", "format": "pdf"}
                ))
                
    with pdfplumber.open(file_path) as pdf_tables:
        for page_num, page in enumerate(pdf_tables.pages):
            for table in page.extract_tables():
                md_table = convert_table_to_markdown(table)
                if md_table:
                    context_header = f"Context: {page_texts.get(page_num, '')[:300]}\n\n" if page_texts.get(page_num) else ""
                    documents.append(Document(
                        page_content=context_header + md_table,
                        metadata={"source": file_path.name, "page": page_num + 1, "type": "table", "format": "pdf"}
                    ))
    return documents

def markdown_parser(file_path: Path) -> list[Document]:
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        if text.strip():
            documents.append(Document(
                page_content=text,
                metadata={"source": file_path.name, "page": 1, "type": "text", "format": "markdown"}
            ))
    except Exception as e:
        print(f"Error parsing Markdown file {file_path.name}: {e}")
        
    return documents
