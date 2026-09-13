from advanced_rag_agent.config.settings import TOP_K, FINAL_K
from advanced_rag_agent.retrieval.hybrid_retriever import hybrid_search
from advanced_rag_agent.retrieval.reranker_encoderbased import rerank_documents

queries = [
    "What is the probation period?",
    "How many earned leaves do employees get?",
    "What are the working hours?",
]

for query in queries:
    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    # Retrieve broader candidate set using Vector + BM25 + RRF
    candidates = hybrid_search(query, k=TOP_K)

    # Cross-encoder selects the most relevant final chunks
    results = rerank_documents(
        query=query,
        documents=candidates,
        top_k=FINAL_K,
    )

    for rank, document in enumerate(results, start=1):
        print(f"\nRANK: {rank}")
        print(f"Source: {document.metadata.get('source')}")
        print(f"Section: {document.metadata.get('section')}")
        print(f"Page: {document.metadata.get('page_start')}")
        print("-" * 80)
        print(document.page_content)