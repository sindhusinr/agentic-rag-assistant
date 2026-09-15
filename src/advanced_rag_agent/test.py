from advanced_rag_agent.graph.rag_graph import graph

# Export the latest LangGraph workflow as Mermaid
mermaid = graph.get_graph().draw_mermaid()

with open("graph.mmd", "w", encoding="utf-8") as file:
    file.write(mermaid)

print("graph.mmd updated successfully.")