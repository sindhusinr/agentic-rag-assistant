import uuid

import streamlit as st

from advanced_rag_agent.graph.rag_graph import graph


@st.cache_resource
def load_graph():
    # Avoid rebuilding the graph on every Streamlit rerun
    return graph


graph_instance = load_graph()


st.set_page_config(
    page_title="Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic RAG Assistant")


# Store UI conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Keep one LangGraph thread for the conversation
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def render_sources(sources):
    """Render KB citations or web links."""
    if not sources:
        return

    st.markdown("**Sources**")

    for source in sources:
        # Web source
        if source.get("url"):
            title = source.get("title", "Web source")
            url = source["url"]
            st.markdown(f"- [{title}]({url})")
            continue

        # Internal KB source
        file_name = source.get("source", "Unknown")
        section = source.get("section", "Unknown")
        page_start = source.get("page_start")
        page_end = source.get("page_end")

        if page_start is None:
            page_text = ""
        elif page_end and page_end != page_start:
            page_text = f", pages {page_start}-{page_end}"
        else:
            page_text = f", page {page_start}"

        st.markdown(
            f"- **{file_name}** — {section}{page_text}"
        )


def get_text(content):
    """Extract text safely from streamed content."""
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text = ""

        for item in content:
            if isinstance(item, str):
                text += item
            elif isinstance(item, dict):
                text += item.get("text", "")

        return text

    return ""


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:
    st.header("Agentic RAG")

    if st.button("🗑️ New Chat"):
        # New thread prevents previous graph state being reused
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


# ==========================================================
# CHAT HISTORY
# ==========================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            render_sources(
                message.get("sources", [])
            )


# ==========================================================
# CHAT INPUT
# ==========================================================

question = st.chat_input(
    "Ask a question..."
)


if question:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    with st.chat_message("assistant"):
        placeholder = st.empty()

        streamed_answer = ""
        final_answer = ""
        sources = []

        # Only stream user-facing generation nodes
        answer_nodes = {
            "generate_kb_answer",
            "generate_general_answer",
            "generate_web_answer",
        }

        try:
            placeholder.markdown("Thinking...")

            for mode, chunk in graph_instance.stream(
                {"query": question},
                config=config,
                stream_mode=[
                    "messages",
                    "updates",
                ],
            ):

                # Stream LLM tokens
                if mode == "messages":
                    message, metadata = chunk

                    node = metadata.get(
                        "langgraph_node"
                    )

                    if node not in answer_nodes:
                        continue

                    text = get_text(
                        message.content
                    )

                    if text:
                        streamed_answer += text

                        placeholder.markdown(
                            streamed_answer + "▌"
                        )

                # Capture final graph state updates
                elif mode == "updates":
                    for _, update in chunk.items():

                        if not isinstance(
                            update,
                            dict,
                        ):
                            continue

                        if update.get(
                            "final_answer"
                        ):
                            final_answer = update[
                                "final_answer"
                            ]

                        if (
                            "sources" in update
                            and update["sources"] is not None
                        ):
                            sources = update[
                                "sources"
                            ]

            # Guardrailed final answer takes priority
            if final_answer:
                answer = final_answer
            elif streamed_answer:
                answer = streamed_answer
            else:
                answer = "No response generated."

            placeholder.markdown(answer)

            render_sources(sources)

        except Exception as error:
            answer = (
                "Sorry, something went wrong while "
                "processing your request."
            )

            sources = []

            placeholder.markdown(answer)

            # Keep technical error visible during development
            st.error(str(error))

    # Save final response in Streamlit history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )