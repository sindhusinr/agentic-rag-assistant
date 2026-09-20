from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langgraph.graph.message import add_messages


class GraphState(TypedDict, total=False):
    # Conversation history
    messages: Annotated[list, add_messages]

    # Original and sanitized user query
    query: str
    sanitized_query: str
    
    # Standalone query created using conversation history
    contextualized_query: str

    # Guardrail result
    guardrail_allowed: bool
    guardrail_reason: str | None
    pii_entities: list[str]

    # Routing decision: "kb", "general", or "web"
    route: str

    # Retrieval results
    retrieved_docs: list[Document]
    reranked_docs: list[Document]

    # Web search results
    web_results: dict

    # Evidence grading result
    evidence_status: str

    # Query rewrite tracking
    rewritten_query: str
    rewrite_count: int

    # Cache information
    cache_hit: bool
    cached_response: dict | None

    # Final response
    final_answer: str
    sources: list[dict]