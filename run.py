# run.py
"""
Manual smoke-test for the LangGraph pipeline.
"""

from flow import build_graph
from langchain_core.messages import HumanMessage

if __name__ == "__main__":
    graph = build_graph()
    # Graph expects state: {"messages": [...]}
    result = graph.invoke({"messages": [HumanMessage("Blue")]})
    print("\nFinal assistant message:\n", result["messages"][-1].content)
