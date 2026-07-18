import logging

import httpx

from src.agent.config import settings

logger = logging.getLogger(__name__)


def fetch_live_aqi(city: str = "delhi") -> dict | None:
    token = settings.WAQI_API_TOKEN
    if not token:
        logger.info("fetch_live_aqi: no WAQI_API_TOKEN configured, skipping live AQI")
        return None

    url = f"https://api.waqi.info/feed/{city}/?token={token}"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        logger.warning(f"fetch_live_aqi: request failed: {e}")
        return None

    if data.get("status") != "ok":
        logger.warning(f"fetch_live_aqi: WAQI returned status {data.get('status')}")
        return None

    raw = data.get("data", {})
    if not raw:
        logger.warning("fetch_live_aqi: no data in response")
        return None

    return {
        "aqi": raw.get("aqi"),
        "dominant_pollutant": raw.get("dominentpol"),
        "station": raw.get("city", {}).get("name"),
        "timestamp": raw.get("time", {}).get("s"),
        "source_url": "https://waqi.info/",
    }
