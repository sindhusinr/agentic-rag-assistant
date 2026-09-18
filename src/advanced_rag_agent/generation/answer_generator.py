from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from advanced_rag_agent.generation.llm import get_llm
from advanced_rag_agent.retrieval.context_builder import build_context


RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an HR policy assistant.

Answer the user's question using only the provided context.

Rules:
- Do not use outside knowledge.
- Do not guess or invent information.
- If the answer is not supported by the context, say that the available documents do not contain enough information.
- Keep the answer clear and concise.
- Do not create citations yourself; citations are handled separately by the application.
""".strip(),
    ),
    (
        "human",
        """
Question:
{query}

Context:
{context}
""".strip(),
    ),
])


GENERAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a helpful assistant.

Answer general knowledge questions accurately and concisely.
Prefer a short answer of 2-4 sentences unless the user explicitly asks
for a detailed explanation.

Do not answer company-specific HR policy questions because those
must be answered using the internal knowledge base.
""".strip(),
    ),
    ("human", "{query}"),
])


WEB_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a web research assistant.

Answer the user's question using only the provided web search results.

Rules:
- Use only information supported by the web results.
- Do not invent facts.
- Prefer recent and relevant information.
- If sources disagree, mention the uncertainty.
- Keep the answer concise, usually 3-6 sentences.
- Use bullet points only when they improve clarity.
- Do not create citations manually; sources are handled separately.
""".strip(),
    ),
    (
        "human",
        """
Question:
{query}

Web results:
{context}
""".strip(),
    ),
])


def generate_answer(
    query: str,
    documents: list[Document],
):
    if not documents:
        return (
            "The available documents do not contain enough "
            "information to answer this question."
        )

    llm = get_llm()
    context = build_context(documents)

    chain = RAG_PROMPT | llm

    response = chain.invoke({
        "query": query,
        "context": context,
    })

    return response.content


def generate_general_answer(query: str):
    llm = get_llm()

    chain = GENERAL_PROMPT | llm

    response = chain.invoke({
        "query": query,
    })

    return response.content


def build_web_context(web_results: dict) -> str:
    results = web_results.get("results", [])
    context_parts = []

    for result in results:
        title = result.get("title", "Unknown")
        url = result.get("url", "")
        content = result.get("content", "")

        # Give the LLM clean web-result context
        context_parts.append(
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {content}"
        )

    return "\n\n".join(context_parts)


def generate_web_answer(
    query: str,
    web_results: dict,
):
    results = web_results.get("results", [])

    if not results:
        return (
            "I couldn't find enough reliable web information "
            "to answer this question."
        )

    llm = get_llm()
    context = build_web_context(web_results)

    chain = WEB_PROMPT | llm

    response = chain.invoke({
        "query": query,
        "context": context,
    })

    return response.content


def build_sources(
    documents: list[Document],
) -> list[dict]:
    sources = []
    seen = set()

    for doc in documents:
        metadata = doc.metadata

        source = metadata.get("source", "Unknown")
        section = metadata.get("section", "Unknown")
        page_start = metadata.get("page_start", 0)
        page_end = metadata.get(
            "page_end",
            page_start,
        )

        # Convert zero-based PDF pages to human-readable pages
        page_start += 1
        page_end += 1

        key = (
            source,
            section,
            page_start,
            page_end,
        )

        # Avoid displaying duplicate sources
        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "source": source,
            "section": section,
            "page_start": page_start,
            "page_end": page_end,
        })

    return sources


def build_web_sources(
    web_results: dict,
) -> list[dict]:
    sources = []
    seen = set()

    for result in web_results.get("results", []):
        url = result.get("url", "")
        title = result.get(
            "title",
            "Unknown",
        )

        if not url or url in seen:
            continue

        seen.add(url)

        sources.append({
            "title": title,
            "url": url,
        })

    return sources


def generate_rag_response(
    query: str,
    documents: list[Document],
) -> dict:
    answer = generate_answer(
        query=query,
        documents=documents,
    )

    # Sources currently come from retrieved/reranked context
    sources = build_sources(documents)

    return {
        "answer": answer,
        "sources": sources,
    }


def generate_web_response(
    query: str,
    web_results: dict,
) -> dict:
    answer = generate_web_answer(
        query=query,
        web_results=web_results,
    )

    sources = build_web_sources(
        web_results
    )

    return {
        "answer": answer,
        "sources": sources,
    }


def generate_insufficient_response() -> dict:
    # Do not guess when internal evidence is unavailable
    return {
        "answer": (
            "I couldn't find enough information in the available HR policy "
            "documents to answer this question reliably."
        ),
        "sources": [],
    }