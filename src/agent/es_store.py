import logging
from typing import Optional

from langchain_elasticsearch import ElasticsearchStore
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.agent.config import settings

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ES_INDEX = settings.ES_INDEX_NAME or "delhi_civic_docs"
DISTANCE_STRATEGY = "COSINE"  # matches all-MiniLM-L6-v2 normalization

_es_store: Optional[ElasticsearchStore] = None
_embeddings: Optional[HuggingFaceEmbeddings] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_es_store() -> Optional[ElasticsearchStore]:
    global _es_store
    if _es_store is not None:
        return _es_store

    url = settings.ES_URL
    if not url:
        logger.info("ES_URL not set — Elasticsearch is disabled, will fall back to Chroma")
        return None

    try:
        es_kwargs = {"es_url": url, "index_name": ES_INDEX, "embedding": _get_embeddings()}

        if settings.ES_API_KEY:
            es_kwargs["es_api_key"] = settings.ES_API_KEY
        elif settings.ES_PASSWORD:
            es_kwargs["es_user"] = settings.ES_USERNAME
            es_kwargs["es_password"] = settings.ES_PASSWORD

        _es_store = ElasticsearchStore(**es_kwargs)
        # Ping to verify connectivity
        _es_store.client.info()
        logger.info(f"Connected to Elasticsearch at {url} (index={ES_INDEX})")
        return _es_store
    except Exception as e:
        logger.warning(f"Elasticsearch connection failed: {e}")
        return None


def index_documents(docs: list, batch_size: int = 50) -> int:
    """Index a list of LangChain Document objects into Elasticsearch."""
    store = get_es_store()
    if store is None:
        logger.error("ES store unavailable — cannot index")
        return 0

    try:
        uuids = [str(hash(doc.page_content[:200])) for doc in docs]
        store.add_documents(docs, ids=uuids, batch_size=batch_size)
        count = len(docs)
        logger.info(f"Indexed {count} documents to ES index '{ES_INDEX}'")
        return count
    except Exception as e:
        logger.error(f"ES indexing failed: {e}")
        return 0


def delete_index():
    store = get_es_store()
    if store is None:
        return
    try:
        store.delete_index()
        logger.info(f"Deleted ES index '{ES_INDEX}'")
    except Exception as e:
        logger.warning(f"Could not delete ES index: {e}")


def similarity_search(query: str, k: int = 5):
    store = get_es_store()
    if store is None:
        return []
    return store.similarity_search(query, k=k)


def similarity_search_with_score(query: str, k: int = 5):
    store = get_es_store()
    if store is None:
        return []
    return store.similarity_search_with_score(query, k=k)
