import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent.civic_qa import list_sources
from src.agent.graph import run_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Delhi Civic Sense Navigator",
    description=(
        "LangGraph agent for Delhi civic issues: classifies intent, routes to the "
        "correct authority with a cited source, asks for clarification instead of "
        "guessing when confidence is low, retries with reformulated queries, calls "
        "live AQI data, and drafts complaints on request."
    ),
    version="3.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent.parent / "static"


class CivicQuery(BaseModel):
    query: str


class CivicResponse(BaseModel):
    answer: str
    citations: list[dict]
    query: str
    chunks_retrieved: int = 0
    issue_type: str | None = None
    authority: dict | None = None
    confidence: str = "high"
    needs_clarification: bool = False
    clarifying_question: str | None = None
    complaint_draft: str | None = None
    used_reformulation: bool = False
    original_query: str = ""
    reformulated_query: str | None = None
    live_data: dict | None = None


@app.get("/")
async def serve_frontend():
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Delhi Civic Sense Navigator</h1><p>Frontend not found.</p>")


@app.post("/api/ask", response_model=CivicResponse)
async def civic_ask(req: CivicQuery):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    logger.info(f"Civic query: {req.query[:100]}...")
    result = run_agent(req.query.strip())
    return CivicResponse(
        answer=result["answer"],
        citations=result["citations"],
        query=result["query"],
        chunks_retrieved=result.get("chunks_retrieved", 0),
        issue_type=result.get("issue_type"),
        authority=result.get("authority"),
        confidence=result.get("confidence", "high"),
        needs_clarification=result.get("needs_clarification", False),
        clarifying_question=result.get("clarifying_question"),
        complaint_draft=result.get("complaint_draft"),
        used_reformulation=bool(result.get("used_reformulation")),
        original_query=result.get("original_query", req.query),
        reformulated_query=result.get("reformulated_query"),
        live_data=result.get("live_data"),
    )


@app.get("/api/sources")
async def get_sources():
    sources = list_sources()
    return {"sources": sources, "count": len(sources)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent": "Delhi Civic Sense Navigator v3.1 (LangGraph + live AQI + query reformulation)",
        "timestamp": datetime.now().isoformat(),
        "llm_configured": bool(os.getenv("GROQ_API_KEY")),
        "waqi_configured": bool(os.getenv("WAQI_API_TOKEN")),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
