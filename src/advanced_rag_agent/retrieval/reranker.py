from sentence_transformers import CrossEncoder

from advanced_rag_agent.config.settings import RERANK_MODEL

_reranker = None

def get_reranker():
    global _reranker

    if not RERANK_MODEL:
        raise ValueError("RERANK_MODEL is not configured.")

    # Load the reranker only when it is actually needed
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)

    return _reranker

def rerank_documents(query, documents, top_k=5):
    if not documents:
        return []

    reranker = get_reranker()

    # Cross-encoder scores each query-document pair directly
    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    # Higher cross-encoder score means more relevant
    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return [
        doc
        for doc, _ in ranked[:top_k]
    ]