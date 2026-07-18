from typing import Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    id: int
    source: str
    url: str


class Authority(BaseModel):
    authority: str
    channel: str
    contact: str
    portal: Optional[str] = None
    source_name: str
    source_url: str


class CivicQuery(BaseModel):
    query: str


class CivicResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    chunks_retrieved: int = 0

    issue_type: Optional[str] = None
    authority: Optional[Authority] = None

    confidence: str = "high"                 # "high" | "low"
    needs_clarification: bool = False
    clarifying_question: Optional[str] = None

    complaint_draft: Optional[str] = None
