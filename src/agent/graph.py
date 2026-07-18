"""
The Civic Navigator agent, wired as an actual LangGraph state machine:

    classify -> retrieve -> confidence_gate --[low confidence]--> answer (clarifying Q) -> END
                                             \\--[high confidence]-> answer -> complaint_draft -> END

This replaces the old single-shot "embed -> stuff prompt -> one LLM call" pipeline
in civic_qa.answer_query with a graph that actually makes decisions:
  - classifies intent before touching the vector store
  - can refuse to guess when it doesn't have a real answer
  - only drafts a complaint when the user is actually reporting something
"""
import logging

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    answer_node,
    classify_node,
    complaint_draft_node,
    confidence_gate_node,
    retrieve_node,
)
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

_compiled_graph = None


def _route_after_gate(state: AgentState) -> str:
    return "skip_draft" if state.get("needs_clarification") else "draft"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("confidence_gate", confidence_gate_node)
    graph.add_node("answer", answer_node)
    graph.add_node("complaint_draft", complaint_draft_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "confidence_gate")
    graph.add_edge("confidence_gate", "answer")
    graph.add_conditional_edges(
        "answer",
        _route_after_gate,
        {"draft": "complaint_draft", "skip_draft": END},
    )
    graph.add_edge("complaint_draft", END)

    return graph.compile()


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_agent(query: str) -> dict:
    graph = _get_graph()
    final_state = graph.invoke({"query": query})

    return {
        "query": query,
        "answer": final_state.get("answer", ""),
        "citations": final_state.get("citations", []),
        "chunks_retrieved": final_state.get("chunks_retrieved", 0),
        "issue_type": final_state.get("issue_type"),
        "authority": final_state.get("authority"),
        "confidence": final_state.get("confidence", "high"),
        "needs_clarification": final_state.get("needs_clarification", False),
        "clarifying_question": final_state.get("clarifying_question"),
        "complaint_draft": final_state.get("complaint_draft"),
    }
