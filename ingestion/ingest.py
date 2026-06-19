from .parser import hybrid_pdf_parser
from .chunker import get_text_splitter
from retrieval.vectorstore import get_vectorstore
from config.settings import PDF_DIR
from observability.file_tracker import (
    build_chunk_tracking_entry,
    initialize_chunk_tracking_report,
    save_chunk_tracking_report,
)


def run_ingestion():
    print("Starting Batched Ingestion...")
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        raise ValueError("No PDFs found.")

    vectorstore = get_vectorstore()
    text_splitter = get_text_splitter()
    total_chunks = 0
    tracked_files = []

    initialize_chunk_tracking_report()
    print("Tracking report initialized at logs/chunked_files.json")

    for idx, file_path in enumerate(pdf_files, start=1):
        print(f"Processing [{idx}/{len(pdf_files)}]: {file_path.name}")
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

            print(
                f"Indexed {file_path.name} | chars={tracked_files[-1]['characters_extracted']} | chunks={tracked_files[-1]['chunks_created']}"
            )
        except Exception as e:
            print(f"Skipping {file_path.name}: {e}")

    save_chunk_tracking_report(tracked_files, status="completed")
    print("Chunk tracking saved to logs/chunked_files.json")
    print(f"Ingestion Complete! Total chunks: {total_chunks}")
    return total_chunks
