"""
Retrieval + formatting helpers shared by the agent graph (src/agent/graph.py).

The old single-shot `answer_query` function that used to live here has been
replaced by the LangGraph pipeline in graph.py — this module now only owns the
Chroma retriever and context formatting, which the graph's nodes reuse.
"""
import os
import logging

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")
COLLECTION_NAME = "delhi_civic_sense"
RETRIEVAL_K = 6

_retriever = None


def _get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever

    if not os.path.exists(CHROMA_DIR):
        logger.warning(f"ChromaDB not found at {CHROMA_DIR}. Run ingest.py first.")
        return None

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )

        _retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K},
        )
        logger.info(f"Retriever initialized from {CHROMA_DIR}")
        return _retriever
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        return None


def format_context(docs: list[Document]) -> tuple[str, list[dict]]:
    contexts = []
    seen_urls = set()
    citations = []

    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:1500]
        source_url = doc.metadata.get("source_url", "")
        source_name = doc.metadata.get("source", "Unknown")

        if source_url and source_url not in seen_urls:
            seen_urls.add(source_url)
            citations.append({
                "id": i,
                "source": source_name,
                "url": source_url,
            })

        contexts.append(f"[Citation {i}] Source: {source_name}\nURL: {source_url}\n\n{content}")

    return "\n\n---\n\n".join(contexts), citations


def list_sources() -> list[dict]:
    retriever = _get_retriever()
    if retriever is None:
        return []

    try:
        vectorstore = getattr(retriever, "vectorstore", None)
        if vectorstore is None:
            return []

        collection_data = vectorstore.get()
        sources_seen = {}
        for meta in collection_data.get("metadatas", []):
            if not meta:
                continue
            url = meta.get("source_url", "")
            name = meta.get("source", "")
            if url and url not in sources_seen:
                sources_seen[url] = name
        result = [{"source": name, "url": url} for url, name in sources_seen.items()]
        return result
    except Exception as e:
        logger.warning(f"Failed to list sources: {e}")
        return []
