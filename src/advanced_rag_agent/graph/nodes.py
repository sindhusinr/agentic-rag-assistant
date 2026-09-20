from langchain_core.messages import AIMessage

from advanced_rag_agent.config.settings import TOP_K, FINAL_K, BROAD_K
from advanced_rag_agent.guardrails.guardrail_manager import (
    validate_input,
    validate_output,
)
from advanced_rag_agent.cache.semantic_cache import (
    get_cached_result,
    save_to_cache,
)
from advanced_rag_agent.ingestion.document_registry import (
    load_registry,
    get_kb_version,
)
from advanced_rag_agent.retrieval.hybrid_retriever import hybrid_search
from advanced_rag_agent.retrieval.reranker import rerank_documents
from advanced_rag_agent.retrieval.web_search import search_web
from advanced_rag_agent.graph.query_router import route_query
from advanced_rag_agent.graph.evidence_grader import grade_evidence
from advanced_rag_agent.graph.query_rewriter import rewrite_query
from advanced_rag_agent.graph.query_contextualizer import contextualize_query
from advanced_rag_agent.graph.state import GraphState
from advanced_rag_agent.generation.answer_generator import (
    generate_rag_response,
    generate_general_answer,
    generate_web_response,
    generate_insufficient_response,
)


def get_current_kb_version() -> str:
    # Cache must always use the current knowledge-base version
    registry = load_registry()
    return get_kb_version(registry)


def get_final_k(query: str) -> int:
    # Broad questions need more chunks to provide enough coverage
    query_lower = query.lower()

    broad_terms = [
        "overview",
        "summarize",
        "summary",
        "all leave",
        "leave types",
        "leave categories",
        "overall leave policy",
    ]

    if any(term in query_lower for term in broad_terms):
        return BROAD_K

    return FINAL_K


def get_contextualized_query(state: GraphState) -> str:
    # Downstream nodes should normally use the standalone query
    query = (
        state.get("contextualized_query")
        or state.get("sanitized_query")
    )

    if not query:
        raise ValueError("No contextualized query available.")

    return query


def get_retrieval_query(state: GraphState) -> str:
    # Retrieval retry takes priority over the contextualized query
    return (
        state.get("rewritten_query")
        or get_contextualized_query(state)
    )


def prepare_query_node(state: GraphState) -> dict:
    # Use explicit query when provided
    query = state.get("query")

    # Otherwise take the latest user message
    if not query:
        messages = state.get("messages", [])

        if not messages:
            raise ValueError("No user query found.")

        query = messages[-1].content

    # Reset turn-specific state so old checkpoint data does not leak
    return {
        "query": query,
        "sanitized_query": None,
        "contextualized_query": None,
        "guardrail_allowed": None,
        "guardrail_reason": None,
        "pii_entities": [],
        "route": None,
        "retrieved_docs": [],
        "reranked_docs": [],
        "evidence_status": None,
        "rewritten_query": None,
        "rewrite_count": 0,
        "web_results": {},
        "cache_hit": False,
        "cached_response": None,
        "final_answer": None,
        "sources": [],
    }


def input_guardrail_node(state: GraphState) -> dict:
    query = state.get("query")

    if not query:
        raise ValueError("No query available for input guardrail.")

    result = validate_input(query)

    if not result["allowed"]:
        return {
            "guardrail_allowed": False,
            "guardrail_reason": result["reason"],
            "pii_entities": [],
            "final_answer": (
                "I can't process that request because it violates "
                "the input safety rules."
            ),
            "sources": [],
        }

    return {
        "guardrail_allowed": True,
        "guardrail_reason": None,
        "sanitized_query": result["sanitized_text"],
        "pii_entities": result["pii_entities"],
    }


def contextualize_query_node(state: GraphState) -> dict:
    sanitized_query = state.get("sanitized_query")

    if not sanitized_query:
        raise ValueError(
            "No sanitized query available for contextualization."
        )

    # Resolve conversational references before routing/retrieval
    contextualized_query = contextualize_query(
        query=sanitized_query,
        messages=state.get("messages", []),
    )

    return {
        "contextualized_query": contextualized_query,
    }


def route_question_node(state: GraphState) -> dict:
    # Route using the standalone conversational query
    query = get_contextualized_query(state)
    route = route_query(query)

    return {
        "route": route,
    }


def cache_lookup_node(state: GraphState) -> dict:
    kb_version = get_current_kb_version()
    route = state.get("route")

    if not route:
        raise ValueError("No route available for cache lookup.")

    # Contextualized query avoids ambiguous follow-up cache keys
    query = get_contextualized_query(state)

    cached_response = get_cached_result(
        query=query,
        kb_version=kb_version,
        route=route,
    )

    if not cached_response:
        return {
            "cache_hit": False,
            "cached_response": None,
        }

    return {
        "cache_hit": True,
        "cached_response": cached_response,
        "final_answer": cached_response["answer"],
        "sources": cached_response.get("sources", []),
    }


def retrieve_kb_node(state: GraphState) -> dict:
    # Retry rewrite takes priority over contextualized query
    query = get_retrieval_query(state)

    documents = hybrid_search(
        query=query,
        k=TOP_K,
    )

    return {
        "retrieved_docs": documents,
    }


def rerank_kb_node(state: GraphState) -> dict:
    query = get_retrieval_query(state)
    final_k = get_final_k(query)

    documents = rerank_documents(
        query=query,
        documents=state.get("retrieved_docs", []),
        top_k=final_k,
    )

    return {
        "reranked_docs": documents,
    }


def grade_evidence_node(state: GraphState) -> dict:
    query = get_retrieval_query(state)

    decision = grade_evidence(
        query=query,
        documents=state.get("reranked_docs", []),
    )

    return {
        "evidence_status": decision,
    }


def rewrite_query_node(state: GraphState) -> dict:
    # Rewrite the standalone query, not the ambiguous original follow-up
    query = get_contextualized_query(state)
    rewritten_query = rewrite_query(query)

    return {
        "rewritten_query": rewritten_query,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
    }


def generate_kb_answer_node(state: GraphState) -> dict:
    # Answer using the resolved standalone question
    query = get_contextualized_query(state)

    result = generate_rag_response(
        query=query,
        documents=state.get("reranked_docs", []),
    )

    return {
        "final_answer": result["answer"],
        "sources": result["sources"],
    }


def generate_general_answer_node(state: GraphState) -> dict:
    query = get_contextualized_query(state)
    answer = generate_general_answer(query)

    return {
        "final_answer": answer,
        "sources": [],
    }


def web_search_node(state: GraphState) -> dict:
    # Search web using the resolved standalone question
    query = get_contextualized_query(state)
    results = search_web(query)

    return {
        "web_results": results,
    }


def generate_web_answer_node(state: GraphState) -> dict:
    query = get_contextualized_query(state)

    result = generate_web_response(
        query=query,
        web_results=state.get("web_results", {}),
    )

    return {
        "final_answer": result["answer"],
        "sources": result["sources"],
    }


def insufficient_answer_node(state: GraphState) -> dict:
    result = generate_insufficient_response()

    return {
        "final_answer": result["answer"],
        "sources": result["sources"],
    }


def output_guardrail_node(state: GraphState) -> dict:
    final_answer = state.get("final_answer")

    if not final_answer:
        raise ValueError("No final answer available for output guardrail.")

    result = validate_output(final_answer)

    return {
        "final_answer": result["sanitized_text"],
    }


def save_cache_node(state: GraphState) -> dict:
    # Do not cache blocked requests or existing cache hits
    if (
        not state.get("guardrail_allowed", False)
        or state.get("cache_hit", False)
    ):
        return {}

    route = state.get("route")

    if not route:
        return {}

    # Do not cache failed KB retrieval responses
    if (
        route == "kb"
        and state.get("evidence_status") == "insufficient"
    ):
        return {}

    final_answer = state.get("final_answer")

    if not final_answer:
        return {}

    kb_version = get_current_kb_version()
    query = get_contextualized_query(state)

    result = {
        "answer": final_answer,
        "sources": state.get("sources", []),
    }

    save_to_cache(
        query=query,
        result=result,
        kb_version=kb_version,
        route=route,
    )

    return {}


def final_message_node(state: GraphState) -> dict:
    final_answer = state.get("final_answer")

    if not final_answer:
        raise ValueError("No final answer available.")

    # Store assistant response alongside persisted HumanMessages
    return {
        "messages": [
            AIMessage(content=final_answer)
        ]
    }