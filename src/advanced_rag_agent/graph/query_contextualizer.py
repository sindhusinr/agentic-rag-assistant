from langchain_core.prompts import ChatPromptTemplate
from advanced_rag_agent.generation.llm import get_llm


CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
Rewrite the current user question as a standalone question using the
conversation history only when necessary.

Rules:
- Resolve references such as "it", "them", "that", "those", and other follow-up references.
- Preserve the user's original intent.
- Do not answer the question.
- Do not add information that is not present in the conversation.
- If the question already makes sense independently, return it unchanged.
- Return only the standalone question.
""".strip(),
    ),
    (
        "human",
        """
Conversation History:
{history}

Current Question:
{query}
""".strip(),
    ),
])
def contextualize_query(query: str, messages: list) -> str:
    # Last message is the current user question, so use only earlier messages
    previous_messages = messages[:-1]

    if not previous_messages:
        return query

    history_parts = []

    for message in previous_messages:
        content = getattr(message, "content", "")
        message_type = getattr(message, "type", "unknown")

        if content:
            history_parts.append(f"{message_type}: {content}")

    if not history_parts:
        return query

    history = "\n".join(history_parts)

    # Rewrite follow-up questions into standalone retrieval queries
    llm = get_llm()
    chain = CONTEXTUALIZE_PROMPT | llm

    response = chain.invoke({
        "history": history,
        "query": query,
    })

    return response.content.strip()