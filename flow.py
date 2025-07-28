"""
LangGraph flow definition using a FAISS retriever and fake LLM.
"""

from typing import Annotated, TypedDict, Literal

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage

from fake_llm import fake_llm
from vectorstore import build_store

# 1. Shared state structure
class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    context: str

# 2. Setup FAISS retriever
vs_retriever = build_store().as_retriever(search_kwargs={"k": 3})

# 3. Node: retrieve context using the last user message
def retrieve_context(state: GraphState) -> GraphState:
    user_msg = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    question = user_msg.content
    docs = vs_retriever.invoke(question)
    context_str = "\n".join(d.page_content for d in docs)
    return {
        "messages": state["messages"],
        "context": context_str
    }

# 4. Node: generate assistant reply using retrieved context
def generate_reply(state: GraphState) -> GraphState:
    user_msg = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))
    context = state.get("context", "")
    prompt = f"Context:\n{context}\nUser:\n{user_msg.content}"
    ai_reply = fake_llm(prompt)
    return {
        "messages": state["messages"] + [AIMessage(content=ai_reply)],
        "context": context
    }

# Transition decision
def _decide(state: GraphState) -> Literal["respond"]:
    return "respond"

# Graph builder
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("retrieve", retrieve_context)
    g.add_node("respond", generate_reply)

    g.add_conditional_edges("retrieve", _decide, {"respond": "respond"})
    g.add_conditional_edges("respond", lambda *_: END, {END: END})

    g.set_entry_point("retrieve")
    return g.compile()

__all__ = ["build_graph"]
