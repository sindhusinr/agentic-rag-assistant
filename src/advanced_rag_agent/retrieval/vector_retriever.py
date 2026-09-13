from langchain_chroma import Chroma

from advanced_rag_agent.config.settings import CHROMA_PATH
from advanced_rag_agent.ingestion.embedder import get_embedding_model

def get_vector_store():
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
    )

def get_retriever(k: int = 5):
    vector_store = get_vector_store()

    # Used later inside RAG pipeline
    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )

def vector_search(query: str, k: int = 5):
    vector_store = get_vector_store()

    # Useful for testing retrieval quality
    return vector_store.similarity_search_with_score(
        query=query,
        k=k,
    )