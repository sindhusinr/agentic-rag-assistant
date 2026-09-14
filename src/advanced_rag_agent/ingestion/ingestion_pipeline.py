from pathlib import Path

from advanced_rag_agent.ingestion.loader import load_pdf
from advanced_rag_agent.ingestion.chunker import chunk_documents
from advanced_rag_agent.ingestion.embedder import get_embedding_model
from advanced_rag_agent.ingestion.indexer import (
    delete_document_chunks,
    add_document_chunks,
    replace_document_chunks,
    remove_document_from_saved_chunks,
)
from advanced_rag_agent.ingestion.document_registry import (
    load_registry,
    save_registry,
    has_document_changed,
    update_document_registry,
)

def ingest_documents(data_dir: str = "data"):
    data_path = Path(data_dir)
    pdf_files = list(data_path.glob("*.pdf"))

    registry = load_registry()

    current_files = {pdf_path.name for pdf_path in pdf_files}
    registered_files = set(registry.keys())

    # Detect files removed from the data folder
    deleted_files = registered_files - current_files

    # Detect new or updated files
    changed_files = [
        pdf_path
        for pdf_path in pdf_files
        if has_document_changed(str(pdf_path), registry)
    ]

    # Nothing to update
    if not changed_files and not deleted_files:
        for pdf_path in pdf_files:
            print(f"Skipping unchanged file: {pdf_path.name}")
        return

    embeddings = get_embedding_model()

    # Remove deleted documents from indexes
    for source in deleted_files:
        print(f"Removing deleted file: {source}")

        delete_document_chunks(
            source=source,
            embeddings=embeddings,
        )

        remove_document_from_saved_chunks(
            source
        )

        registry.pop(
            source,
            None,
        )

    # Process new or changed documents
    for pdf_path in pdf_files:
        source = pdf_path.name

        if pdf_path not in changed_files:
            print(f"Skipping unchanged file: {source}")
            continue

        print(f"Processing changed file: {source}")

        documents = load_pdf(str(pdf_path))
        chunks = chunk_documents(documents)

        if not chunks:
            raise ValueError(
                f"No chunks created for {source}"
            )

        # Remove old vectors for this source
        delete_document_chunks(
            source=source,
            embeddings=embeddings,
        )

        # Add latest vectors
        add_document_chunks(
            chunks=chunks,
            embeddings=embeddings,
        )

        # Keep BM25 corpus synchronized
        replace_document_chunks(
            source=source,
            new_chunks=chunks,
        )

        # Update registry only after successful indexing
        update_document_registry(
            str(pdf_path),
            registry,
        )

        print(
            f"Indexed {source}: "
            f"{len(chunks)} chunks"
        )

    # Save registry once after all updates
    save_registry(registry)