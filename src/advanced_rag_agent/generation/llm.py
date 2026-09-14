from langchain_groq import ChatGroq

from advanced_rag_agent.config.settings import GROQ_API_KEY, LLM_MODEL

_llm = None

def get_llm():
    global _llm

    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")

    if not LLM_MODEL:
        raise ValueError("LLM_MODEL is not configured.")

    # Create the LLM client only when generation is needed
    if _llm is None:
        _llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=0.2,
            streaming=True,
        )

    return _llm