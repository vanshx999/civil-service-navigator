import os
import logging
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.agent.config import settings
from src.agent import es_store
from src.agent.llm import call_llm

logger = logging.getLogger(__name__)

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")
COLLECTION_NAME = "delhi_civic_sense"
RETRIEVAL_K = 5
DISTANCE_THRESHOLD = 1.0

_retriever = None
_vectorstore = None
_use_es = False
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_vectorstore():
    global _vectorstore, _use_es

    if _vectorstore is not None:
        return _vectorstore

    # Try Elasticsearch first if configured
    es = es_store.get_es_store()
    if es is not None:
        _vectorstore = es
        _use_es = True
        logger.info("Using Elasticsearch as vector store")
        return _vectorstore

    # Fall back to Chroma
    if not os.path.exists(CHROMA_DIR):
        logger.warning(f"ChromaDB not found at {CHROMA_DIR}")
        return None
    try:
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=_get_embeddings(),
            persist_directory=CHROMA_DIR,
        )
        _use_es = False
        logger.info("Using Chroma as vector store")
        return _vectorstore
    except Exception as e:
        logger.error(f"Vectorstore init failed: {e}")
        return None


def _get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        return None
    try:
        _retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K},
        )
        return _retriever
    except Exception as e:
        logger.error(f"Retriever init failed: {e}")
        return None


QA_SYSTEM_PROMPT = """You're a Delhi local who knows the city's civic systems. Answer using only the info below.

RULES:
1. Sound like a real person. Short sentences, natural flow. No robotic phrases like "based on the provided context" or "as an AI".
2. Start with the answer. Use numbers, names, dates from the context. Cite them like [1], [2].
3. If helplines are mentioned, give the numbers. If rules are mentioned, sum them up plainly.
4. Never make anything up. If it's not in the context, skip it.
5. Keep it short — 2-4 paragraphs. No bullet points or markdown.

Write like you're telling a neighbour."""


def format_context(docs: list[Document]) -> tuple[str, list[dict]]:
    contexts = []
    seen_urls = set()
    citations = []

    for i, doc in enumerate(docs, 1):
        content = doc.page_content[:2000]
        source_url = doc.metadata.get("source_url", "")
        source_name = doc.metadata.get("source", "Unknown")

        if source_url and source_url not in seen_urls:
            seen_urls.add(source_url)
            citations.append({"id": i, "source": source_name, "url": source_url})

        contexts.append(f"[{i}] (Source: {source_name})\n{content}")

    return "\n\n".join(contexts), citations


def web_search(query: str, max_results: int = 5) -> list[dict]:
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return []


def answer_query(query: str) -> dict:
    retriever = _get_retriever()
    all_citations = []
    web_results_used = False

    if retriever:
        try:
            docs = retriever.invoke(query)
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            docs = []
    else:
        docs = []

    if docs:
        context_text, citations = format_context(docs)
        all_citations = citations

        prompt = f"""Someone in Delhi is asking: {query}

Here's what I found about it:
{context_text}

Sources:
{chr(10).join(f'[{c["id"]}] {c["source"]}: {c["url"]}' for c in citations)}

Answer them directly. Use the info above, cite with [1] [2], keep it conversational."""
    else:
        prompt = ""
        context_text = ""

    llm_response = None
    if prompt:
        llm_response = call_llm(prompt=prompt, system_prompt=QA_SYSTEM_PROMPT)

    needs_web = False
    answer = (llm_response or "").strip()

    if not answer:
        needs_web = True
    elif len(answer) < 100:
        needs_web = True
    elif "there is no" in answer.lower() or "does not contain" in answer.lower() or "not provide" in answer.lower() or "not available" in answer.lower() or "no direct information" in answer.lower():
        needs_web = True
        logger.info(f"LLM gave weak answer for '{query[:60]}', falling back to web search")

    if needs_web:
        logger.info(f"Web searching: {query[:80]}")
        web_results = web_search(f"Delhi civic {query}", max_results=5)

        if web_results:
            web_citations = []
            web_contexts = []
            for i, r in enumerate(web_results, 1):
                web_contexts.append(f"[Web {i}] {r['title']}\n{r['snippet']}\nURL: {r['url']}")
                web_citations.append({"id": i, "source": r["title"][:60], "url": r["url"]})

            web_context = "\n\n".join(web_contexts)
            web_citations_text = "\n".join(f'[{c["id"]}] {c["source"]}: {c["url"]}' for c in web_citations)
            all_citations = web_citations

            web_prompt = f"""Someone in Delhi is asking: {query}

I found these from the web:
{web_context}

Citations:
{web_citations_text}

Give them a straightforward answer with the details above. Cite sources as [1], [2]. Keep it natural."""
            web_answer = call_llm(prompt=web_prompt, system_prompt=QA_SYSTEM_PROMPT)
            if web_answer and len(web_answer.strip()) > 50:
                answer = web_answer.strip()
                web_results_used = True
            else:
                parts = []
                for r in web_results[:3]:
                    parts.append(f"{r['title']}: {r['snippet']} ({r['url']})")
                answer = "Came across this online:\n\n" + "\n\n".join(parts)
                web_results_used = True

    if not answer:
        answer = "Couldn't find anything on that. Try asking about garbage collection, helplines, pollution, or road repairs in Delhi."

    return {
        "answer": answer,
        "citations": all_citations,
        "query": query,
        "chunks_retrieved": len(docs) if docs else 0,
        "web_search_used": web_results_used,
    }


def list_sources() -> list[dict]:
    retriever = _get_retriever()
    if retriever is None:
        return []
    try:
        if _use_es:
            es = es_store.get_es_store()
            if es is None:
                return []
            resp = es.client.search(
                index=es.index_name,
                body={"size": 0, "aggs": {"sources": {"terms": {"field": "metadata.source.keyword", "size": 50}}}},
            )
            buckets = resp.get("aggregations", {}).get("sources", {}).get("buckets", [])
            return [{"source": b["key"], "url": ""} for b in buckets]

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
        return [{"source": name, "url": url} for url, name in sources_seen.items()]
    except Exception as e:
        logger.warning(f"Failed to list sources: {e}")
        return []
