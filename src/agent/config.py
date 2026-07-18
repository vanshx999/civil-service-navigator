import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    dotenv_path = Path(__file__).parent.parent.parent / ".env"
    load_dotenv(dotenv_path)
except ImportError:
    pass

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    WAQI_API_TOKEN: str = os.getenv("WAQI_API_TOKEN", "")
    MAX_RETRIES: int = 3
    OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
    MAX_TASKS: int = 10
    AGENT_NAME: str = "PlanExecute-v1"

    # Elasticsearch — set ES_URL or ELASTICSEARCH_URL to enable; falls back to Chroma
    ES_URL: str = os.getenv("ES_URL", os.getenv("ELASTICSEARCH_URL", ""))
    ES_API_KEY: str = os.getenv("ES_API_KEY", os.getenv("ELASTICSEARCH_API_KEY", ""))
    ES_USERNAME: str = os.getenv("ES_USERNAME", "elastic")
    ES_PASSWORD: str = os.getenv("ES_PASSWORD", "")
    ES_INDEX_NAME: str = os.getenv("ES_INDEX_NAME", "delhi_civic_docs")
    ES_ANALYTICS_INDEX: str = os.getenv("ES_ANALYTICS_INDEX", "delhi_civic_analytics")

settings = Settings()
