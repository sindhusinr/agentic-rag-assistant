from langchain_tavily import TavilySearch

from advanced_rag_agent.config.settings import TAVILY_API_KEY


_web_search = None


def get_web_search():
    global _web_search

    if not TAVILY_API_KEY:
        raise ValueError("TAVILY_API_KEY is not configured.")

    # Create Tavily only when web search is needed
    if _web_search is None:
        _web_search = TavilySearch(
            max_results=5,
            tavily_api_key=TAVILY_API_KEY,
        )

    return _web_search


def search_web(query: str):
    web_search = get_web_search()

    return web_search.invoke({
        "query": query,
    })