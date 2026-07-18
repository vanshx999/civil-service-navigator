import os
import logging
from typing import Optional

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.agent.llm import call_llm

logger = logging.getLogger(__name__)

CHROMA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma")
COLLECTION_NAME = "delhi_civic_sense"
RETRIEVAL_K = 5

_retriever = None
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore
    if not os.path.exists(CHROMA_DIR):
        logger.warning(f"ChromaDB not found at {CHROMA_DIR}")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_DIR,
        )
        return _vectorstore
    except Exception as e:
        logger.error(f"Vectorstore init failed: {e}")
        return None


def _get_retriever():
    global _retriever
    if _retriever is not None:
        return _retriever
    if not os.path.exists(CHROMA_DIR):
        logger.warning(f"ChromaDB not found at {CHROMA_DIR}")
        return None
    try:
        vectorstore = _get_vectorstore()
        if vectorstore is None:
            return None
        _retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": RETRIEVAL_K},
        )
        return _retriever
    except Exception as e:
        logger.error(f"Retriever init failed: {e}")
        return None


QA_SYSTEM_PROMPT = """You are an AI that answers questions about Delhi civic issues using the provided context. You ONLY extract facts from the context below — you never make up information.

INSTRUCTIONS:
1. Read the retrieved context CAREFULLY. It contains official documents from Delhi Government, MCD, PIB, DPCC, and news sources.
2. Extract EVERY specific fact, number, name, helpline, date, and rule mentioned in the context that's relevant to the question.
3. Cite each fact with the corresponding [N] number from the context.
4. If the context lists helplines — mention ALL of them with their numbers.
5. If the context describes rules — summarize them using specific details from the context.
6. Be thorough and specific. List actual phone numbers, dates, department names, URLs.
7. NEVER say "the context doesn't contain information" — the context DOES contain information, extract it.

OUTPUT FORMAT:
Start directly with your answer. Use inline citations like [1], [2]. End with "Sources:" section.

EXAMPLE OUTPUT:
The Solid Waste Management Rules 2026 were notified by the Ministry of Environment [1] and will take effect on April 1, 2026 [1]. They replace the 2016 rules [1].

Sources:
[1] PIB - https://pib.gov.in/PressReleasePage.aspx?PRID=2219676
"""


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

    # Step 1: Try vector DB retrieval
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

        prompt = f"""Answer this Delhi civic question using ONLY the context below.

QUESTION: {query}

CONTEXT:
{context_text}

CITATIONS:
{chr(10).join(f'[{c["id"]}] {c["source"]}: {c["url"]}' for c in citations)}

Now write your answer. Extract every specific detail from the context."""
    else:
        prompt = ""
        context_text = ""

    llm_response = None
    if prompt:
        llm_response = call_llm(prompt=prompt, system_prompt=QA_SYSTEM_PROMPT)

    # Step 2: If LLM gave a weak answer or no context, fall back to web search
    answer = (llm_response or "").strip()
    needs_web = False

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

            web_prompt = f"""Answer this Delhi civic question using the web search results below.

QUESTION: {query}

WEB RESULTS:
{web_context}

CITATIONS:
{web_citations_text}

Write a helpful answer. Always include specific numbers, names, and URLs."""
            web_answer = call_llm(prompt=web_prompt, system_prompt=QA_SYSTEM_PROMPT)
            if web_answer and len(web_answer.strip()) > 50:
                answer = web_answer.strip()
                web_results_used = True
            else:
                # Raw fallback
                parts = []
                for r in web_results[:3]:
                    parts.append(f"{r['title']}: {r['snippet']} ({r['url']})")
                answer = "Here's what I found from web sources:\n\n" + "\n\n".join(parts)
                web_results_used = True

    if not answer:
        answer = "I couldn't find information about that. Try asking about waste management, helplines, air pollution, or reporting civic issues in Delhi."

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
