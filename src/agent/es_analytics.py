import logging
from datetime import datetime, timezone
from typing import Optional

from elasticsearch import Elasticsearch

from src.agent.config import settings

logger = logging.getLogger(__name__)

_es_client: Optional[Elasticsearch] = None
ANALYTICS_INDEX = settings.ES_ANALYTICS_INDEX or "delhi_civic_analytics"


def _get_client() -> Optional[Elasticsearch]:
    global _es_client
    if _es_client is not None:
        return _es_client

    url = settings.ES_URL
    if not url:
        return None

    try:
        kwargs = {"hosts": [url]}
        if settings.ES_API_KEY:
            kwargs["api_key"] = settings.ES_API_KEY
        elif settings.ES_PASSWORD:
            kwargs["basic_auth"] = (settings.ES_USERNAME, settings.ES_PASSWORD)

        _es_client = Elasticsearch(**kwargs)
        _es_client.info()
        return _es_client
    except Exception as e:
        logger.warning(f"ES analytics client failed: {e}")
        return None


def log_query(
    query: str,
    answer: str,
    issue_type: Optional[str],
    confidence: str,
    chunks_retrieved: int,
    used_reformulation: bool,
    live_data: Optional[dict],
    response_time_ms: int,
    error: Optional[str] = None,
):
    client = _get_client()
    if client is None:
        return

    doc = {
        "@timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer_length": len(answer),
        "issue_type": issue_type,
        "confidence": confidence,
        "chunks_retrieved": chunks_retrieved,
        "used_reformulation": used_reformulation,
        "has_live_data": live_data is not None,
        "response_time_ms": response_time_ms,
        "error": error,
    }

    try:
        client.index(index=ANALYTICS_INDEX, document=doc)
    except Exception as e:
        logger.warning(f"Failed to log query to ES: {e}")


def ensure_analytics_index():
    """Create the analytics index template if it doesn't exist."""
    client = _get_client()
    if client is None:
        return

    mapping = {
        "mappings": {
            "properties": {
                "@timestamp": {"type": "date"},
                "query": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "answer_length": {"type": "integer"},
                "issue_type": {"type": "keyword"},
                "confidence": {"type": "keyword"},
                "chunks_retrieved": {"type": "integer"},
                "used_reformulation": {"type": "boolean"},
                "has_live_data": {"type": "boolean"},
                "response_time_ms": {"type": "integer"},
                "error": {"type": "text"},
            }
        }
    }

    try:
        if not client.indices.exists(index=ANALYTICS_INDEX):
            client.indices.create(index=ANALYTICS_INDEX, body=mapping)
            logger.info(f"Created analytics index '{ANALYTICS_INDEX}'")
    except Exception as e:
        logger.warning(f"Failed to create analytics index: {e}")
