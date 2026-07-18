"""
The Civic Navigator agent, wired as an actual LangGraph state machine:

    classify -> fetch_live_data (if air_pollution) -> retrieve -> confidence_gate
        --[low: insufficient_context & retry==0]--> reformulate -> retrieve -> confidence_gate
        --[low: off_topic]------------------------------> answer (clarifying Q) -> END
        --[high confidence]-----------------------------> answer -> complaint_draft -> END

This replaces the old single-shot "embed -> stuff prompt -> one LLM call" pipeline
in civic_qa.answer_query with a graph that actually makes decisions:
  - classifies intent before touching the vector store
  - can refuse to guess when it doesn't have a real answer
  - retries with a reformulated query once if the first retrieval was empty
  - calls a live AQI API when the issue is air pollution
  - only drafts a complaint when the user is actually reporting something
"""
import logging

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    answer_node,
    classify_node,
    complaint_draft_node,
    confidence_gate_node,
    fetch_live_data_node,
    reformulate_query_node,
    retrieve_node,
)
from src.agent.state import AgentState

logger = logging.getLogger(__name__)

_compiled_graph = None


def _route_after_gate(state: AgentState) -> str:
    low_reason = state.get("low_confidence_reason")
    retry_count = state.get("retry_count", 0)
    if low_reason == "insufficient_context" and retry_count == 0:
        return "reformulate"
    return "answer"


def _route_after_classify(state: AgentState) -> str:
    if state.get("issue_type") == "air_pollution":
        return "fetch_live_data"
    return "retrieve"


def _route_after_answer(state: AgentState) -> str:
    return "skip_draft" if state.get("needs_clarification") else "draft"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("fetch_live_data", fetch_live_data_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("confidence_gate", confidence_gate_node)
    graph.add_node("reformulate", reformulate_query_node)
    graph.add_node("answer", answer_node)
    graph.add_node("complaint_draft", complaint_draft_node)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        _route_after_classify,
        {"fetch_live_data": "fetch_live_data", "retrieve": "retrieve"},
    )
    graph.add_edge("fetch_live_data", "retrieve")
    graph.add_edge("retrieve", "confidence_gate")

    graph.add_conditional_edges(
        "confidence_gate",
        _route_after_gate,
        {"reformulate": "reformulate", "answer": "answer"},
    )

    graph.add_edge("reformulate", "retrieve")

    graph.add_conditional_edges(
        "answer",
        _route_after_answer,
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

    initial_state: AgentState = {
        "query": query,
        "retry_count": 0,
        "used_reformulation": False,
    }
    final_state = graph.invoke(initial_state)

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
        "used_reformulation": bool(final_state.get("used_reformulation")),
        "original_query": final_state.get("original_query", query),
        "reformulated_query": final_state.get("reformulated_query"),
        "live_data": final_state.get("live_data"),
    }
