from advanced_rag_agent.retrieval.vector_retriever import get_retriever
from advanced_rag_agent.retrieval.bm25_retriever import create_bm25_retriever
from advanced_rag_agent.retrieval.chunk_store import load_chunks

def hybrid_search(query: str, k: int = 5, rrf_k: int = 60):
    chunks = load_chunks()

    vector_retriever = get_retriever(k)
    bm25_retriever = create_bm25_retriever(chunks, k)

    vector_results = vector_retriever.invoke(query)
    bm25_results = bm25_retriever.invoke(query)

    scores = {}
    documents = {}

    # Combine rankings instead of raw scores
    for results in [vector_results, bm25_results]:
        for rank, doc in enumerate(results, start=1):
            key = doc.page_content

            scores[key] = scores.get(
                key, 0
            ) + 1 / (rrf_k + rank)

            documents[key] = doc

    ranked_keys = sorted(
        scores,
        key=scores.get,
        reverse=True,
    )

    return [
        documents[key]
        for key in ranked_keys[:k]
    ]