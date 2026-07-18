import logging
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.agent.civic_qa import answer_query, list_sources

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Delhi Civic Sense Navigator",
    description="AI agent that answers questions about civic issues in Delhi with cited sources + web search fallback",
    version="2.1.0",
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
    web_search_used: bool = False


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
    result = answer_query(req.query.strip())
    return CivicResponse(
        answer=result["answer"],
        citations=result["citations"],
        query=result["query"],
        chunks_retrieved=result.get("chunks_retrieved", 0),
        web_search_used=result.get("web_search_used", False),
    )


@app.get("/api/sources")
async def get_sources():
    sources = list_sources()
    return {"sources": sources, "count": len(sources)}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "agent": "Delhi Civic Sense Navigator v2.1",
        "timestamp": datetime.now().isoformat(),
        "llm_configured": bool(os.getenv("GROQ_API_KEY")),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
