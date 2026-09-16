from langchain_core.documents import Document

from advanced_rag_agent.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mrr,
)

# Mock retrieval result: correct section appears at Rank 2
documents = [
    Document(
        page_content="Earned leave information",
        metadata={
            "source": "Leave policy.pdf",
            "section": "1. Earned Leave (EL)",
        },
    ),
    Document(
        page_content="Sick leave information",
        metadata={
            "source": "Leave policy.pdf",
            "section": "2. Sick Leave (SL)",
        },
    ),
    Document(
        page_content="General leave guidelines",
        metadata={
            "source": "Leave policy.pdf",
            "section": "B. General Guidelines",
        },
    ),
]

# Ground truth for our test question
relevant_sections = [
    {
        "source": "Leave policy.pdf",
        "section": "2. Sick Leave (SL)",
    }
]

k = 3

print("Precision@3:", precision_at_k(documents, relevant_sections, k))
print("Recall@3:", recall_at_k(documents, relevant_sections, k))
print("MRR:", mrr(documents, relevant_sections))