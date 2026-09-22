from langgraph.graph import StateGraph, START, END

from advanced_rag_agent.graph.state import GraphState
from advanced_rag_agent.graph.memory import memory
from advanced_rag_agent.graph.nodes import (
    prepare_query_node,
    input_guardrail_node,
    contextualize_query_node,
    cache_lookup_node,
    route_question_node,
    retrieve_kb_node,
    rerank_kb_node,
    grade_evidence_node,
    rewrite_query_node,
    generate_kb_answer_node,
    generate_general_answer_node,
    web_search_node,
    generate_web_answer_node,
    insufficient_answer_node,
    output_guardrail_node,
    save_cache_node,
    final_message_node,
)


def route_after_guardrail(state: GraphState) -> str:
    # Block unsafe input before conversational contextualization
    if not state.get("guardrail_allowed", False):
        return "output_guardrail"

    return "contextualize_query"


def route_after_cache(state: GraphState) -> str:
    # Cache hit skips retrieval and generation
    if state.get("cache_hit", False):
        return "output_guardrail"

    route = state.get("route")

    if route == "general":
        return "generate_general_answer"

    if route == "web":
        return "web_search"

    return "retrieve_kb"


def route_after_evidence_grade(state: GraphState) -> str:
    if state.get("evidence_status") == "sufficient":
        return "generate_kb_answer"

    # Allow only one rewrite and retrieval retry
    if state.get("rewrite_count", 0) < 1:
        return "rewrite_query"

    return "insufficient_answer"


builder = StateGraph(GraphState)

# Register common nodes
builder.add_node("prepare_query", prepare_query_node)
builder.add_node("input_guardrail", input_guardrail_node)
builder.add_node("contextualize_query", contextualize_query_node)
builder.add_node("route_question", route_question_node)
builder.add_node("cache_lookup", cache_lookup_node)

# Internal KB branch
builder.add_node("retrieve_kb", retrieve_kb_node)
builder.add_node("rerank_kb", rerank_kb_node)
builder.add_node("grade_evidence", grade_evidence_node)
builder.add_node("rewrite_query", rewrite_query_node)
builder.add_node("generate_kb_answer", generate_kb_answer_node)

# General branch
builder.add_node("generate_general_answer",generate_general_answer_node)

# Web branch
builder.add_node("web_search", web_search_node)
builder.add_node("generate_web_answer",generate_web_answer_node)

# Shared final-response nodes
builder.add_node("insufficient_answer",insufficient_answer_node)
builder.add_node("output_guardrail",output_guardrail_node)
builder.add_node("save_cache",save_cache_node)
builder.add_node("final_message",final_message_node)

# Entry flow
builder.add_edge(START,"prepare_query",)
builder.add_edge("prepare_query","input_guardrail")

# Unsafe input stops here; valid input moves to contextualization
builder.add_conditional_edges(
    "input_guardrail",
    route_after_guardrail,
    {"contextualize_query": "contextualize_query","output_guardrail": "output_guardrail",}
)

# Convert conversational follow-up into a standalone query
builder.add_edge("contextualize_query","route_question")

# Route first so cache lookup knows which branch to search
builder.add_edge("route_question","cache_lookup")

# Cache decision
builder.add_conditional_edges(
    "cache_lookup",
    route_after_cache,
    {
        "retrieve_kb": "retrieve_kb",
        "generate_general_answer": "generate_general_answer",
        "web_search": "web_search",
        "output_guardrail": "output_guardrail",
    },
)

# Internal KB retrieval flow
builder.add_edge("retrieve_kb","rerank_kb")
builder.add_edge("rerank_kb","grade_evidence",)

# Evidence decision
builder.add_conditional_edges(
    "grade_evidence",
    route_after_evidence_grade,
    {
        "generate_kb_answer": "generate_kb_answer",
        "rewrite_query": "rewrite_query",
        "insufficient_answer": "insufficient_answer",
    },
)

# Retry KB retrieval once after query rewriting
builder.add_edge("rewrite_query", "retrieve_kb")

# Web flow
builder.add_edge("web_search", "generate_web_answer")

# All generated answers go through output guardrail
builder.add_edge("generate_kb_answer", "output_guardrail")
builder.add_edge("generate_general_answer", "output_guardrail")
builder.add_edge("generate_web_answer", "output_guardrail")
builder.add_edge("insufficient_answer", "output_guardrail")

# Sanitize before cache and final output
builder.add_edge("output_guardrail", "save_cache")
builder.add_edge("save_cache", "final_message")
builder.add_edge("final_message", END)

# Checkpointer preserves graph state for the same thread_id
graph = builder.compile(checkpointer=memory)