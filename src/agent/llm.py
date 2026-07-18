import json
import logging
import time

import httpx

from src.agent.config import settings

logger = logging.getLogger(__name__)

_last_call = 0.0


def _rate_limit():
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < 1.5:
        time.sleep(1.5 - elapsed)
    _last_call = time.time()


def call_llm(prompt: str, system_prompt: str, model: str | None = None) -> str | None:
    """Generic Groq chat-completion call. Returns None if no API key is set or the
    call fails, so callers can fall back to deterministic behavior instead of crashing."""
    if not settings.GROQ_API_KEY:
        logger.info("No GROQ_API_KEY set — using deterministic fallback")
        return None

    _rate_limit()

    try:
        with httpx.Client(timeout=90.0) as client:
            resp = client.post(
                settings.GROQ_BASE_URL,
                headers={
                    "Authorization": f"Bearer {settings.GROQ_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model or settings.GROQ_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.warning(f"LLM call failed: {e}")
        return None


def call_llm_structured(prompt: str, system_prompt: str, model: str | None = None) -> dict | None:
    """Same as call_llm but parses the response as JSON. Returns None on any failure
    (missing key, malformed JSON, etc.) so callers must have a deterministic fallback."""
    raw = call_llm(prompt, system_prompt, model=model)
    if not raw:
        return None
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            cleaned = cleaned.rsplit("```", 1)[0]
        return json.loads(cleaned)
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")
        return None
