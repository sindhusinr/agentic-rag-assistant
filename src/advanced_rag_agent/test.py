from advanced_rag_agent.retrieval.hybrid_retriever import hybrid_search
from advanced_rag_agent.retrieval.reranker import rerank_documents
from advanced_rag_agent.graph.evidence_grader import grade_evidence
from advanced_rag_agent.graph.query_rewriter import rewrite_query

QUERY = "Can you summarize our company's leave policy?"

print("\n===== ORIGINAL QUERY =====")
print(QUERY)

# First retrieval
retrieved_docs = hybrid_search(
    query=QUERY,
    k=10,
)

print("\n===== HYBRID RESULTS =====")

for index, doc in enumerate(
    retrieved_docs,
    start=1,
):
    print(f"\n--- Document {index} ---")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Section: {doc.metadata.get('section')}")
    print(doc.page_content[:500])


# Rerank and keep final top 6
reranked_docs = rerank_documents(
    query=QUERY,
    documents=retrieved_docs,
    top_k=6,
)

print("\n===== RERANKED TOP 6 =====")

for index, doc in enumerate(
    reranked_docs,
    start=1,
):
    print(f"\n--- Document {index} ---")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Section: {doc.metadata.get('section')}")
    print(doc.page_content[:500])


# Grade first retrieval
decision = grade_evidence(
    query=QUERY,
    documents=reranked_docs,
)

print("\n===== FIRST EVIDENCE DECISION =====")
print(decision)


# Rewrite query
rewritten_query = rewrite_query(QUERY)

print("\n===== REWRITTEN QUERY =====")
print(rewritten_query)


# Retrieve again using rewritten query
retrieved_docs_2 = hybrid_search(
    query=rewritten_query,
    k=10,
)

reranked_docs_2 = rerank_documents(
    query=rewritten_query,
    documents=retrieved_docs_2,
    top_k=6,
)

print("\n===== SECOND RERANKED TOP 6 =====")

for index, doc in enumerate(
    reranked_docs_2,
    start=1,
):
    print(f"\n--- Document {index} ---")
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Section: {doc.metadata.get('section')}")
    print(doc.page_content[:500])


# Grade second retrieval
decision_2 = grade_evidence(
    query=rewritten_query,
    documents=reranked_docs_2,
)

print("\n===== SECOND EVIDENCE DECISION =====")
print(decision_2)