from advanced_rag_agent.evaluation.dataset import EVALUATION_DATASET
from advanced_rag_agent.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
)
from advanced_rag_agent.retrieval.hybrid_retriever import hybrid_search
from advanced_rag_agent.retrieval.reranker import rerank_documents
from advanced_rag_agent.config.settings import TOP_K

EVAL_K = 3

for index, example in enumerate(EVALUATION_DATASET, start=1):
    question = example["question"]
    relevant_sections = example["relevant_sections"]

    # Run the same hybrid retrieval used by the RAG application
    retrieved_docs = hybrid_search(question, k=TOP_K)

    # Evaluate the final ranking after cross-encoder reranking
    reranked_docs = rerank_documents(
        question,
        retrieved_docs,
        top_k=EVAL_K,
    )

    precision = precision_at_k(reranked_docs, relevant_sections, EVAL_K)
    recall = recall_at_k(reranked_docs, relevant_sections, EVAL_K)
    reciprocal_rank = mrr(reranked_docs, relevant_sections)

    print(f"\nQuestion {index}: {question}")

    # Print ranking so poor scores can be debugged easily
    for rank, doc in enumerate(reranked_docs, start=1):
        print(
            f"  Rank {rank}: "
            f"{doc.metadata.get('source')} | "
            f"{doc.metadata.get('section')}"
        )

    print(f"  Precision@{EVAL_K}: {precision:.3f}")
    print(f"  Recall@{EVAL_K}: {recall:.3f}")
    print(f"  MRR: {reciprocal_rank:.3f}")