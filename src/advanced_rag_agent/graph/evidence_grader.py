from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from advanced_rag_agent.generation.llm import get_llm


class EvidenceDecision(BaseModel):
    # Keep the output limited to two valid decisions
    decision: Literal["sufficient", "insufficient"] = Field(
        description="Whether the retrieved evidence is sufficient to answer the query."
    )

GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an evidence grader for an HR policy RAG system.

Decide whether the retrieved context contains enough relevant information
to answer the user's question accurately.

Return "sufficient" when:
- the context directly answers a specific question, or
- the context contains enough relevant information to form a reliable answer,
- for broad overview questions, the context contains multiple relevant policy
  details that can support a useful summary.

Return "insufficient" when:
- the context is unrelated,
- the key information requested by the user is missing,
- the context is too limited to form a reliable answer,
- answering would require guessing or inventing policy details.

Important:
- An overview does not need to include every section of the policy unless the
  user explicitly asks for a complete or exhaustive list.
- Judge only whether the available context can support an accurate answer.
- Do not require information that the user did not ask for.

Use only the provided context.
Return only the structured decision.
""".strip(),
    ),
    (
        "human",
        """
Question:
{query}

Retrieved context:
{context}
""".strip(),
    ),
])

def format_documents(documents: list[Document]) -> str:
    # Combine retrieved chunks only for grading
    return "\n\n".join(
        doc.page_content
        for doc in documents
    )


def grade_evidence(
    query: str,
    documents: list[Document],
) -> str:
    # No retrieved documents means evidence is automatically insufficient
    if not documents:
        return "insufficient"

    llm = get_llm()
    grader_llm = llm.with_structured_output(EvidenceDecision)

    chain = GRADER_PROMPT | grader_llm

    context = format_documents(documents)

    result = chain.invoke({
        "query": query,
        "context": context,
    })

    return result.decision