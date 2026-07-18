"""
Deterministic issue -> authority routing table.

Why this exists instead of relying purely on vector retrieval: for "who do I contact"
questions, a citizen needs a correct answer every time, not a plausible-sounding one.
Every entry here is drawn directly from an already-ingested, dated source
(src/data/raw/delhi_helplines_20260718.md), not invented. If a query's issue type
isn't in this table, the agent must say so honestly instead of guessing — see
nodes.confidence_gate_node.

Known limitation (stated on purpose, not hidden): this table routes by issue type only,
not by MCD zone/ward. Ward-level jurisdiction data was not ingested, so the agent does
not claim to know which of MCD's 12 zones a location falls into.
"""

_HELPLINE_SOURCE = {
    "name": "Delhi Government Helpline Center",
    "url": "https://delhi.gov.in/page/helpline-center",
}

AUTHORITY_MAP: dict[str, dict] = {
    "garbage": {
        "authority": "MCD (Municipal Corporation of Delhi)",
        "channel": "MCD 311 mobile app, MCD helpline, or the online grievance portal",
        "contact": "155305 (24/7) · mcd-ithelpdesk@mcd.nic.in",
        "portal": "https://mcdonline.nic.in/portal",
        **_HELPLINE_SOURCE,
    },
    "pothole_road": {
        "authority": "MCD for local/internal roads, PWD for major arterial roads",
        "channel": "MCD 311 app for local roads; PWD toll-free line for PWD-maintained roads",
        "contact": "MCD: 155305 · PWD: 1800-110-093 / 1908 / complaint@pwddelhi.com",
        "portal": "https://mcdonline.nic.in/portal",
        **_HELPLINE_SOURCE,
    },
    "streetlight": {
        "authority": "MCD (Municipal Corporation of Delhi)",
        "channel": "MCD 311 mobile app or MCD helpline",
        "contact": "155305 (24/7) · mcd-ithelpdesk@mcd.nic.in",
        "portal": "https://mcdonline.nic.in/portal",
        **_HELPLINE_SOURCE,
    },
    "water_supply": {
        "authority": "Delhi Jal Board (via the Delhi Govt Water Helpline)",
        "channel": "Water Helpline",
        "contact": "1916",
        "portal": None,
        **_HELPLINE_SOURCE,
    },
    "waterlogging_drainage": {
        "authority": "Water Helpline, with DDMA for disaster-level waterlogging",
        "channel": "Water Helpline; DDMA Disaster Management Helpline for severe/monsoon flooding",
        "contact": "Water: 1916 · DDMA: 1077",
        "portal": None,
        **_HELPLINE_SOURCE,
    },
    "air_pollution": {
        "authority": "Delhi Pollution Control Committee (DPCC) / CAQM",
        "channel": "DPCC website (complaints/monitoring); CAQM is the Delhi-NCR nodal body",
        "contact": "https://dpcc.delhi.gov.in/",
        "portal": "https://dpcc.delhi.gov.in/",
        **_HELPLINE_SOURCE,
    },
    "emergency": {
        "authority": "Delhi Police / Fire / Ambulance",
        "channel": "Direct emergency call",
        "contact": "Police: 100 · Fire: 101 · Ambulance: 102",
        "portal": None,
        **_HELPLINE_SOURCE,
    },
}

# Maps free-text issue labels the classifier might emit to a canonical key above.
ISSUE_ALIASES = {
    "garbage": "garbage", "waste": "garbage", "trash": "garbage",
    "pothole": "pothole_road", "road": "pothole_road", "pothole_road": "pothole_road",
    "streetlight": "streetlight", "street_light": "streetlight",
    "water": "water_supply", "water_supply": "water_supply",
    "waterlogging": "waterlogging_drainage", "drainage": "waterlogging_drainage",
    "sewage": "waterlogging_drainage",
    "air_pollution": "air_pollution", "pollution": "air_pollution",
    "emergency": "emergency", "police": "emergency", "fire": "emergency",
}


def resolve_authority(issue_type: str | None) -> dict | None:
    if not issue_type:
        return None
    key = ISSUE_ALIASES.get(issue_type.strip().lower())
    return AUTHORITY_MAP.get(key) if key else None
