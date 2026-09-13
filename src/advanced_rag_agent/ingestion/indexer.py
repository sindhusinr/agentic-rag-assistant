import pickle
from pathlib import Path

from langchain_chroma import Chroma
from advanced_rag_agent.config.settings import CHROMA_PATH

CHUNKS_PATH = Path("data/chunks.pkl")

def get_vector_store(embeddings):
    if not CHROMA_PATH:
        raise ValueError("CHROMA_PATH is not configured.")

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

def delete_document_chunks(source: str, embeddings):
    vector_store = get_vector_store(embeddings)

    # Remove old vectors for this document
    vector_store.delete(
        where={"source": source}
    )

def add_document_chunks(chunks, embeddings):
    if not chunks:
        raise ValueError("Cannot index empty chunks.")

    vector_store = get_vector_store(embeddings)

    # Add latest vectors for this document
    vector_store.add_documents(chunks)

    return vector_store

def load_saved_chunks():
    # First ingestion may not have chunks.pkl yet
    if not CHUNKS_PATH.exists():
        return []

    with open(CHUNKS_PATH, "rb") as file:
        return pickle.load(file)

def save_chunks(chunks):
    # Persist complete chunk corpus for BM25
    with open(CHUNKS_PATH, "wb") as file:
        pickle.dump(chunks, file)

def replace_document_chunks(source: str, new_chunks):
    existing_chunks = load_saved_chunks()

    # Remove old chunks for this document
    remaining_chunks = [
        chunk
        for chunk in existing_chunks
        if chunk.metadata.get("source") != source
    ]

    remaining_chunks.extend(new_chunks)

    save_chunks(remaining_chunks)

def remove_document_from_saved_chunks(source: str):
    existing_chunks = load_saved_chunks()

    # Remove deleted document from BM25 corpus
    remaining_chunks = [
        chunk
        for chunk in existing_chunks
        if chunk.metadata.get("source") != source
    ]

    save_chunks(remaining_chunks)