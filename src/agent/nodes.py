import logging
import re

from src.agent import civic_qa
from src.agent.authority_map import resolve_authority
from src.agent.llm import call_llm, call_llm_structured
from src.agent.state import AgentState
from src.agent.tools import fetch_live_aqi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. classify_node — figure out intent before doing anything else.
# ---------------------------------------------------------------------------

CLASSIFY_SYSTEM_PROMPT = """You classify a citizen's message about Delhi civic issues.

Output ONLY valid JSON, no prose, with this exact shape:
{
  "is_civic_related": true/false,
  "issue_type": one of ["garbage","pothole_road","streetlight","water_supply",
                          "waterlogging_drainage","air_pollution","emergency",
                          "general_info", null],
  "location_hint": "<any area/locality/ward the user mentioned, else null>",
  "wants_complaint_draft": true/false
}

Rules:
- "general_info" means the user is asking an informational question (rules, policy,
  status of a scheme) rather than reporting a specific problem.
- wants_complaint_draft is true only if the user describes an actual problem they
  want to report/file a complaint about (not just asking how the system works).
- is_civic_related is false only for messages with nothing to do with Delhi civic
  matters at all (e.g. small talk, unrelated topics).
"""

# Lightweight keyword fallback used when there's no LLM key configured, so the
# agent still behaves deterministically rather than refusing to run.
_KEYWORD_MAP = [
    (r"\b(garbage|trash|waste (not )?collect)", "garbage"),
    (r"\b(pothole|road (is )?damaged|broken road)", "pothole_road"),
    (r"\b(streetlight|street light|lamp post)", "streetlight"),
    (r"\b(water (supply|not coming|shortage)|no water)", "water_supply"),
    (r"\b(waterlog|drainage|sewage|sewer)", "waterlogging_drainage"),
    (r"\b(pollution|smog|aqi|air quality)", "air_pollution"),
    (r"\b(emergency|fire|accident|police)", "emergency"),
]

_REPORT_VERBS = re.compile(r"\b(report|complain|file a? ?complaint|not (been )?collected|"
                            r"isn'?t working|broken|not working|leak(ing)?|overflow)\b", re.I)


def _keyword_classify(query: str) -> dict:
    q = query.lower()
    issue_type = "general_info"
    for pattern, label in _KEYWORD_MAP:
        if re.search(pattern, q, re.I):
            issue_type = label
            break
    return {
        "is_civic_related": True,  # can't cheaply rule this out without an LLM; default permissive
        "issue_type": issue_type,
        "location_hint": None,
        "wants_complaint_draft": bool(_REPORT_VERBS.search(q)) and issue_type != "general_info",
    }


def classify_node(state: AgentState) -> AgentState:
    query = state["query"]
    result = call_llm_structured(
        prompt=f"Citizen message: {query}",
        system_prompt=CLASSIFY_SYSTEM_PROMPT,
    )
    if result is None:
        logger.info("classify_node: no LLM available, using keyword fallback")
        result = _keyword_classify(query)

    return {
        **state,
        "issue_type": result.get("issue_type"),
        "location_hint": result.get("location_hint"),
        "is_civic_related": bool(result.get("is_civic_related", True)),
        "wants_complaint_draft": bool(result.get("wants_complaint_draft", False)),
        "original_query": state.get("original_query", query),
    }


# ---------------------------------------------------------------------------
# 2. retrieve_node — RAG over the ingested, cited documents (reuses civic_qa).
# ---------------------------------------------------------------------------

DISTANCE_THRESHOLD = 1.0  # cosine distance; lower = more similar


def retrieve_node(state: AgentState) -> AgentState:
    vectorstore = civic_qa._get_vectorstore()
    if vectorstore is None:
        return {**state, "docs": [], "citations": [], "chunks_retrieved": 0}

    try:
        scored = vectorstore.similarity_search_with_score(state["query"], k=5)
    except Exception as e:
        logger.error(f"retrieve_node: retrieval failed: {e}")
        return {**state, "docs": [], "citations": [], "chunks_retrieved": 0}

    # Filter by distance threshold so low-relevance results don't count
    docs = [doc for doc, score in scored if score < DISTANCE_THRESHOLD]

    context_text, citations = civic_qa.format_context(docs) if docs else ("", [])
    return {
        **state,
        "docs": docs,
        "citations": citations,
        "chunks_retrieved": len(docs),
        "_context_text": context_text,
    }


# ---------------------------------------------------------------------------
# 3. fetch_live_data_node — live AQI call (only for air_pollution).
# ---------------------------------------------------------------------------

def fetch_live_data_node(state: AgentState) -> AgentState:
    live = fetch_live_aqi()
    return {**state, "live_data": live}


# ---------------------------------------------------------------------------
# 4. confidence_gate_node — decide whether we actually know enough to answer.
# ---------------------------------------------------------------------------

def confidence_gate_node(state: AgentState) -> AgentState:
    is_civic_related = state.get("is_civic_related", True)
    chunks_retrieved = state.get("chunks_retrieved", 0)
    issue_type = state.get("issue_type")
    authority = resolve_authority(issue_type)

    if not is_civic_related:
        return {
            **state,
            "confidence": "low",
            "needs_clarification": True,
            "clarifying_question": (
                "I can only help with Delhi civic issues — waste management, roads, "
                "water, streetlights, air pollution, and where to report them. "
                "Could you rephrase your question around one of those?"
            ),
            "authority": None,
            "low_confidence_reason": "off_topic",
        }

    # If we have neither a deterministic authority match nor any retrieved context,
    # we genuinely don't know the answer — say so instead of letting the LLM guess.
    if authority is None and chunks_retrieved == 0:
        return {
            **state,
            "confidence": "low",
            "needs_clarification": True,
            "clarifying_question": (
                "I don't have enough information on that specific topic yet. Could you "
                "tell me a bit more, or ask about waste management, roads, water, "
                "streetlights, or air pollution in Delhi?"
            ),
            "authority": None,
            "low_confidence_reason": "insufficient_context",
        }

    return {
        **state,
        "confidence": "high",
        "needs_clarification": False,
        "clarifying_question": None,
        "authority": authority,
        "low_confidence_reason": None,
    }


# ---------------------------------------------------------------------------
# 5. reformulate_query_node — retry with a better query when retrieval was empty.
# ---------------------------------------------------------------------------

REFORMULATE_SYSTEM_PROMPT = (
    "Rewrite this citizen's civic query into a more specific search query likely to "
    "match official Delhi government documents about waste management, roads, water, "
    "air pollution, or civic helplines. Output ONLY the rewritten query, no explanation."
)

_KEYWORD_EXPANSIONS = {
    "SWM": "Solid Waste Management",
    "AQI": "Air Quality Index",
    "MCD": "MCD Municipal Corporation of Delhi",
    "DJ": "Delhi Jal Board",
    "DPCC": "Delhi Pollution Control Committee",
    "PWD": "Public Works Department",
    "NDMC": "New Delhi Municipal Council",
    "CAQM": "Commission for Air Quality Management",
    "CPCB": "Central Pollution Control Board",
    "RWA": "Resident Welfare Association",
    "BWG": "bulk waste generator",
    "EBWGR": "Extended Bulk Waste Generator Responsibility",
    "WTE": "Waste to Energy",
    "MRF": "Material Recovery Facility",
    "RDF": "Refuse Derived Fuel",
    "NGT": "National Green Tribunal",
    "IEC": "Information Education Communication",
    "EV": "electric vehicle",
}


def _keyword_expand(query: str) -> str:
    expanded = query
    for abbr, full in _KEYWORD_EXPANSIONS.items():
        expanded = re.sub(rf"\b{abbr}\b", full, expanded, flags=re.I)
    return expanded + " Delhi MCD official"


def reformulate_query_node(state: AgentState) -> AgentState:
    original_query = state.get("original_query", state["query"])
    current_query = state["query"]

    llm_response = call_llm(
        prompt=f"Original query: {original_query}",
        system_prompt=REFORMULATE_SYSTEM_PROMPT,
    )

    if llm_response:
        reformulated = llm_response.strip().strip('"\'')
    else:
        reformulated = _keyword_expand(current_query)

    logger.info(f"reformulate_query_node: '{current_query[:60]}' -> '{reformulated[:60]}'")
    return {
        **state,
        "query": reformulated,
        "reformulated_query": reformulated,
        "retry_count": state.get("retry_count", 0) + 1,
        "used_reformulation": True,
    }


# ---------------------------------------------------------------------------
# 6. answer_node — either return the clarifying question, or produce a grounded answer.
# ---------------------------------------------------------------------------

def answer_node(state: AgentState) -> AgentState:
    if state.get("needs_clarification"):
        return {**state, "answer": state["clarifying_question"], "complaint_draft": None}

    authority = state.get("authority")
    context_text = state.get("_context_text", "")
    citations = state.get("citations", [])
    live_data = state.get("live_data")

    # Build system prompt dynamically to avoid mentioning "authority match" when none exists
    base_prompt = (
        "You are the Delhi Civic Sense Navigator, an assistant that answers "
        "questions about civic issues in Delhi, India, using ONLY the retrieved context "
        "and any live data provided.\n\n"
        "Rules:\n"
        "1. Answer strictly from the retrieved context below. Extract EVERY specific fact, "
        "number, name, helpline, date, and rule mentioned in the context that's relevant.\n"
        "2. Cite sources inline using numbered brackets like [1], [2], matching the SOURCES list.\n"
        "3. Be concise (2-4 short paragraphs). Include helpline numbers/portals when relevant.\n"
        "4. NEVER say 'the context is insufficient' or 'I don't have enough information' — "
        "the context DOES contain information, extract it.\n"
    )
    if authority:
        system_prompt = base_prompt + (
            "5. An AUTHORITY MATCH is provided below. Lead with it plainly, "
            "stating the responsible authority and how to report, before adding background.\n"
        )
    else:
        system_prompt = base_prompt + (
            "5. AUTHORITY MATCH: none — do not invent one, just cite background sources normally.\n"
        )

    authority_block = ""
    if authority:
        authority_block = (
            f"AUTHORITY MATCH (deterministic, from {authority['name']}):\n"
            f"- Responsible authority: {authority['authority']}\n"
            f"- How to report: {authority['channel']}\n"
            f"- Contact: {authority['contact']}\n"
            + (f"- Portal: {authority['portal']}\n" if authority.get("portal") else "")
        )

    # Live data block — prepended prominently if present
    live_data_block = ""
    if live_data:
        live_data_block = (
            "LIVE DATA (fetched just now, not from static documents): "
            f"Delhi AQI is {live_data.get('aqi', 'N/A')} "
            f"({live_data.get('dominant_pollutant', 'unknown')} dominant), "
            f"reported at {live_data.get('timestamp', 'unknown time')} "
            f"by station '{live_data.get('station', 'unknown')}'. "
            "Source: waqi.info (World Air Quality Index project, aggregating "
            "CPCB/DPCC monitoring stations).\n\n"
        )

    sources_list = "\n".join(f'[{c["id"]}] {c["source"]}: {c["url"]}' for c in citations)
    prompt = f"""## User Question:
{state['query']}

{live_data_block}{authority_block}

## Retrieved Context:
{context_text if context_text else '(no additional background retrieved)'}

## SOURCES:
{sources_list if sources_list else '(none)'}

Answer the question. If LIVE DATA is present, lead with it first, then add
any useful background from the retrieved context with inline citations like [1].
If an AUTHORITY MATCH is given, state it clearly after any live data."""

    llm_response = call_llm(prompt=prompt, system_prompt=system_prompt)

    if llm_response:
        answer = llm_response.strip()
    elif live_data:
        # deterministic fallback surfacing live data
        answer = (
            f"Current Delhi AQI: {live_data.get('aqi', 'N/A')} "
            f"({live_data.get('dominant_pollutant', 'unknown')}), "
            f"as of {live_data.get('timestamp', 'unknown time')}. "
            f"Source: waqi.info — {live_data.get('source_url', '')}"
        )
    elif authority:
        answer = (
            f"This falls under **{authority['authority']}**.\n\n"
            f"How to report: {authority['channel']}\n"
            f"Contact: {authority['contact']}\n"
            + (f"Portal: {authority['portal']}\n" if authority.get("portal") else "")
            + f"\nSource: {authority['name']} — {authority['url']}"
        )
    else:
        parts = []
        for i, doc in enumerate(state.get("docs", []), 1):
            source = doc.metadata.get("source", "Unknown")
            parts.append(f"Based on {source}: {doc.page_content[:400]}")
        answer = "\n\n".join(parts) if parts else "I couldn't find enough information to answer that."

    return {**state, "answer": answer}


# ---------------------------------------------------------------------------
# 7. complaint_draft_node — only runs if the user actually wants to file something.
# ---------------------------------------------------------------------------

DRAFT_SYSTEM_PROMPT = """Write a short, polite, factual civic complaint (max 120 words)
a Delhi citizen could submit as-is. Include: the issue, the location if given (or note
that location should be added), and a request for resolution within a reasonable
timeframe. Do not invent facts the user didn't provide. Output plain text only."""


def complaint_draft_node(state: AgentState) -> AgentState:
    if not state.get("wants_complaint_draft") or state.get("needs_clarification"):
        return {**state, "complaint_draft": None}

    authority = state.get("authority")
    location = state.get("location_hint") or "[add your locality/ward here]"
    prompt = (
        f"Issue reported by citizen: {state['query']}\n"
        f"Location: {location}\n"
        f"Responsible authority: {authority['authority'] if authority else 'unspecified'}\n"
        f"Contact channel: {authority['channel'] if authority else 'unspecified'}"
    )

    draft = call_llm(prompt=prompt, system_prompt=DRAFT_SYSTEM_PROMPT)
    if not draft:
        auth_name = authority["authority"] if authority else "the relevant department"
        draft = (
            f"To: {auth_name}\n"
            f"Subject: Civic complaint — {state.get('issue_type', 'issue')}\n\n"
            f"I would like to report the following issue: {state['query']}\n"
            f"Location: {location}\n"
            f"Kindly look into this and resolve it at the earliest. "
            f"Please share a reference/complaint ID for tracking."
        )

    return {**state, "complaint_draft": draft.strip()}
