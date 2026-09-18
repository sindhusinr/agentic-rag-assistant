from langsmith import Client

from advanced_rag_agent.retrieval.hybrid_retriever import hybrid_search
from advanced_rag_agent.retrieval.reranker import rerank_documents
from advanced_rag_agent.config.settings import TOP_K

DATASET_NAME = "agentic-rag-hr-evaluation"
EVAL_K = 3

client = Client()


def retrieval_target(inputs: dict) -> dict:
    question = inputs["question"]

    # Run the production retrieval pipeline
    retrieved_docs = hybrid_search(question, k=TOP_K)
    reranked_docs = rerank_documents(
        question,
        retrieved_docs,
        top_k=EVAL_K,
    )

    # Return section metadata so evaluators can compare with ground truth
    retrieved_sections = [
        {
            "source": doc.metadata.get("source"),
            "section": doc.metadata.get("section"),
        }
        for doc in reranked_docs
    ]

    return {
        "retrieved_sections": retrieved_sections,
    }
def _section_keys(sections):
    # Convert section dictionaries into comparable tuples
    return {
        (item["source"], item["section"])
        for item in sections
    }


def precision_evaluator(run, example):
    retrieved = run.outputs["retrieved_sections"]
    relevant = example.outputs["relevant_sections"]

    # Remove duplicate retrieved sections while preserving ranking
    unique_retrieved = list(dict.fromkeys(
        (item["source"], item["section"])
        for item in retrieved
    ))

    relevant_keys = _section_keys(relevant)
    hits = sum(section in relevant_keys for section in unique_retrieved)

    return {
        "key": "precision_at_3",
        "score": hits / len(unique_retrieved) if unique_retrieved else 0.0,
    }


def recall_evaluator(run, example):
    retrieved = _section_keys(run.outputs["retrieved_sections"])
    relevant = _section_keys(example.outputs["relevant_sections"])

    hits = len(retrieved & relevant)

    return {
        "key": "recall_at_3",
        "score": hits / len(relevant) if relevant else 0.0,
    }


def reciprocal_rank_evaluator(run, example):
    retrieved = run.outputs["retrieved_sections"]
    relevant = _section_keys(example.outputs["relevant_sections"])

    # Rank starts at 1 because reciprocal rank is 1 / first relevant rank
    for rank, item in enumerate(retrieved, start=1):
        key = (item["source"], item["section"])

        if key in relevant:
            return {
                "key": "reciprocal_rank",
                "score": 1 / rank,
            }

    return {
        "key": "reciprocal_rank",
        "score": 0.0,
    }
if __name__ == "__main__":
    # Run all dataset examples through the retrieval pipeline
    client.evaluate(
        retrieval_target,
        data=DATASET_NAME,
        evaluators=[
            precision_evaluator,
            recall_evaluator,
            reciprocal_rank_evaluator,
        ],
        experiment_prefix="retrieval-baseline",
        description="Hybrid retrieval + cross-encoder reranking evaluated at Top-3.",
    )