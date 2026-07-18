"""
Shared state passed between nodes in the Civic Navigator agent graph.
"""
from typing import Any, Optional, TypedDict


class AgentState(TypedDict, total=False):
    # input
    query: str

    # classify_node output
    issue_type: Optional[str]          # e.g. "garbage", "pothole_road", "water_supply"
    location_hint: Optional[str]       # free-text location mentioned by the user, if any
    is_civic_related: bool             # False for off-topic queries (weather, jokes, etc.)
    wants_complaint_draft: bool        # True if the user is trying to report/file something

    # retrieve_node output (RAG over ingested docs — used for informational answers)
    docs: list
    citations: list
    chunks_retrieved: int

    # confidence_gate_node output
    confidence: str                    # "high" | "low"
    needs_clarification: bool
    clarifying_question: Optional[str]

    # answer_node output
    authority: Optional[dict]          # deterministic routing match, if any (with citation)
    answer: str

    # complaint_draft_node output
    complaint_draft: Optional[str]
