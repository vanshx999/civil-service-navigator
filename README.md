# Delhi Civic Sense Navigator

An agent that routes Delhi citizens to the correct civic authority for their issue,
cites the government/official source for that routing, and drafts a submittable
complaint on request — instead of just answering questions from a vector index.

## Why this is an agent, not a chatbot

The pipeline makes decisions, not just retrieval + one LLM call:

```
classify → retrieve → confidence_gate → answer → complaint_draft
```

- **classify** — extracts issue type, location hint, and whether the user wants a
  complaint drafted, before touching the knowledge base.
- **retrieve** — RAG over ingested, dated, sourced documents (see below).
- **confidence_gate** — if the query is off-topic, or there's neither a deterministic
  authority match nor any retrieved context, the agent **asks a clarifying question
  instead of guessing an answer.**
- **answer** — leads with a deterministic, cited authority match (`src/agent/authority_map.py`)
  when one exists, then adds retrieved background with inline citations.
- **complaint_draft** — only runs if the user is actually reporting a problem; drafts
  a short, submittable complaint using only facts the user provided.

Built with LangGraph (`src/agent/graph.py`) as a real state machine with conditional
branching — see `src/agent/nodes.py` for each node's logic.

## Data sources & credit

Every fact this agent surfaces is traceable to one of these sources, ingested with a
retrieval timestamp (`src/data/raw/*.md`):

| Source | Credit |
|---|---|
| mcdonline.nic.in | Municipal Corporation of Delhi (MCD) |
| delhi.gov.in/page/helpline-center | Government of NCT of Delhi — Helpline Center |
| pib.gov.in | Press Information Bureau, Government of India |
| dpcc.delhi.gov.in | Delhi Pollution Control Committee |
| smartcities data portal | Ministry of Housing & Urban Affairs |
| Times of India, The Hindu (news articles) | Respective publications, linked and dated |

The full list with URLs is served live at `GET /api/sources`.

## Known limitations (stated honestly, not hidden)

- **No ward/zone-level jurisdiction data.** MCD is split into 12 zones, but this
  agent does not have a street→zone lookup, so it cannot tell you *which* MCD zone
  office handles your specific address — only that MCD (vs. NDMC, PWD, DJB, etc.)
  is the right body. Location hints are used for the complaint draft text, not for
  fine-grained routing.
- **NDMC and Delhi Cantonment Board are separate bodies** not covered by this
  agent's MCD-focused sources. A query about Lutyens' Delhi / New Delhi area may
  route incorrectly to MCD; this is a known gap, not a silent one.
- The deterministic authority table (`authority_map.py`) covers the most common
  issue types (garbage, potholes, streetlights, water, drainage, air pollution,
  emergencies) — anything outside that list falls back to RAG-only, cited, best-effort
  answers, or a clarifying question if there isn't enough retrieved context.

## Running it

```
pip install -r requirements.txt
python src/data/ingest.py   # builds the Chroma index from src/data/raw/*.md
uvicorn src.main:app --reload --port 8000
```

Set `GROQ_API_KEY` for LLM-backed classification/answering; without it, the agent
still runs end-to-end using deterministic keyword classification and templated
answers, so a missing key degrades gracefully instead of failing the demo.
