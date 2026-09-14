from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from advanced_rag_agent.generation.llm import get_llm


class RouteDecision(BaseModel):
    # Restrict routing to supported graph branches
    route: Literal["kb", "general", "web"] = Field(
        description="Route the query to internal KB, general LLM, or web search."
    )


ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a query router for an HR assistant.

Route to "kb" when the question requires company-specific internal information,
such as:
- employee handbook
- leave policy
- probation
- working hours
- attendance
- employee benefits
- parental leave
- internal HR policies

Route to "web" when the question requires current or external information,
such as:
- latest news
- recent legal or regulatory changes
- current government policies
- current market information
- information that may have changed recently

Route to "general" when the question is general knowledge that does not require
internal company documents or current web information.

Important:
- Never route company-specific policy questions to the web.
- If a company-specific answer is unavailable in the KB, the system must not
  use public web information as a replacement.

Return only the structured route decision.
""".strip(),
    ),
    ("human", "{query}"),
])


def route_query(query: str) -> str:
    llm = get_llm()

    # Structured output prevents free-form routing responses
    router_llm = llm.with_structured_output(RouteDecision)
    chain = ROUTER_PROMPT | router_llm

    result = chain.invoke({
        "query": query,
    })

    return result.route