import os

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"
    GROQ_MODEL: str = "llama-3.3-70b-versatile"
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    MAX_RETRIES: int = 3
    OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")
    MAX_TASKS: int = 10
    AGENT_NAME: str = "PlanExecute-v1"

settings = Settings()
