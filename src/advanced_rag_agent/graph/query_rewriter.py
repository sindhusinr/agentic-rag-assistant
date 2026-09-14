from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from advanced_rag_agent.generation.llm import get_llm


class RewriteResult(BaseModel):
    # Keep rewrite output structured
    rewritten_query: str = Field(
        description="A query optimized for internal HR knowledge-base retrieval."
    )


REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You rewrite user questions to improve retrieval from an internal HR policy knowledge base.

Rules:
- Preserve the user's original intent.
- Keep the rewritten query concise.
- Use relevant HR terminology when helpful.
- Do not invent company-specific facts, policy names, entitlements, numbers, or conditions.
- Do not change a company-policy question into a public-law question.

For a specific question:
- Make only minimal changes.
- Focus on the exact requested policy information.

For a broad overview question:
- Expand it into useful retrieval concepts.
- Include neutral concepts such as overview, categories, eligibility,
  entitlements, application rules, and general guidelines when relevant.
- Do not invent specific leave types or benefits that the user did not mention.

Return only the structured rewritten query.
""".strip(),
    ),
    ("human", "Original query: {query}"),
])


def rewrite_query(query: str) -> str:
    llm = get_llm()

    # Structured output avoids parsing free-form LLM text
    rewrite_llm = llm.with_structured_output(
        RewriteResult
    )

    chain = REWRITE_PROMPT | rewrite_llm

    result = chain.invoke({
        "query": query,
    })

    return result.rewritten_query