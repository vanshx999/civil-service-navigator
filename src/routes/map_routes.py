import logging
import random
import math
from datetime import datetime, timedelta
from fastapi import APIRouter, Query
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# Realistic Delhi landmark coordinates for mock issue generation
DELHI_HOTSPOTS = [
    # (lat, lon, area_name)
    (28.6139, 77.2090, "Connaught Place"),
    (28.7041, 77.1025, "Sadar Bazar"),
    (28.5677, 77.2105, "Hauz Khas"),
    (28.5447, 77.1905, "Saket"),
    (28.6276, 77.2161, "Chandni Chowk"),
    (28.5893, 77.0770, "Dwarka Sector 6"),
    (28.5355, 77.2580, "Lajpat Nagar"),
    (28.4595, 77.0266, "Gurgaon Border"),
    (28.7499, 77.1167, "Rohini Sector 15"),
    (28.6944, 77.1425, "Pitampura"),
    (28.5682, 77.2189, "Greater Kailash"),
    (28.6312, 77.3007, "Mayur Vihar"),
    (28.5085, 77.2324, "Jamia Nagar"),
    (28.5802, 77.2182, "Panchsheel Park"),
    (28.6592, 77.2380, "Civil Lines"),
    (28.7264, 77.1937, "Model Town"),
    (28.4916, 77.2155, "Kalkaji"),
    (28.6631, 77.2880, "Shahdara"),
    (28.5553, 77.2706, "Okhla"),
    (28.6783, 77.1828, "Karol Bagh"),
    (28.6380, 77.2765, "Preet Vihar"),
    (28.7714, 77.1257, "Bawana"),
    (28.6183, 77.1678, "Moti Nagar"),
    (28.6934, 77.0958, "Paschim Vihar"),
    (28.6429, 77.0297, "Najafgarh"),
    (28.5989, 77.1891, "RK Puram"),
    (28.6489, 77.1358, "Janakpuri"),
    (28.5783, 77.2100, "Vasant Kunj"),
    (28.7047, 77.2890, "Yamuna Vihar"),
    (28.4639, 77.0886, "Kapashera"),
]

CATEGORIES = [
    {"id": "garbage", "label": "Garbage", "color": "#22c55e", "icon": "🗑️"},
    {"id": "road", "label": "Roads", "color": "#eab308", "icon": "🛣️"},
    {"id": "water", "label": "Water", "color": "#3b82f6", "icon": "💧"},
    {"id": "electricity", "label": "Electricity", "color": "#a855f7", "icon": "⚡"},
    {"id": "air_quality", "label": "Air Quality", "color": "#ef4444", "icon": "🌫️"},
    {"id": "emergency", "label": "Emergency", "color": "#000000", "icon": "🚨"},
]

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
STATUSES = ["active", "in_progress", "resolved", "pending"]

HELPLINES = {
    "garbage": "MCD 311 Helpline — 155311",
    "road": "PWD Road Helpline — 1800-11-0793",
    "water": "DJ Jal Board — 1916",
    "electricity": "BSES — 011-39999666 / BYPL — 011-39999777",
    "air_quality": "DPCC — https://dpcc.delhi.gov.in/",
    "emergency": "Police — 100 / Fire — 101 / Ambulance — 102",
}

ISSUE_TITLES = {
    "garbage": [
        "Garbage not collected in {area} for {n} days",
        "Overflowing dumpster at {area} market",
        "Illegal dumping near {area}",
        "Burning garbage in {area} causing smoke",
    ],
    "road": [
        "Large pothole on {area} main road",
        "Road cave-in near {area} metro station",
        "Damaged footpath on {area} street",
        "Waterlogged underpass at {area}",
    ],
    "water": [
        "No water supply in {area} since yesterday",
        "Broken water main flooding {area} road",
        "Contaminated water reported in {area}",
        "Sewage overflow in {area} colony",
    ],
    "electricity": [
        "Power outage in {area} for {n} hours",
        "Snapped power line near {area}",
        "Transformer explosion at {area}",
        "Voltage fluctuation in {area} damaging appliances",
    ],
    "air_quality": [
        "AQI Very Poor ({aqi}) at {area}",
        "Dense smog reported in {area}",
        "Construction dust choking {area}",
        "Stubble burning smoke in {area}",
    ],
    "emergency": [
        "Fire reported at {area} market",
        "Building collapse near {area}",
        "Gas leak detected in {area}",
        "Flooding emergency in {area}",
    ],
}


def _generate_issues(count: int = 80) -> list[dict]:
    now = datetime.now()
    issues = []
    for i in range(count):
        hotspot = random.choice(DELHI_HOTSPOTS)
        lat, lon, area = hotspot
        cat = random.choice(CATEGORIES)
        severity = random.choice(SEVERITY_LEVELS)
        status = random.choice(STATUSES)
        aqi = random.randint(150, 450) if cat["id"] == "air_quality" else 0
        days = random.randint(1, 7)
        n = random.choice([2, 3, 5, 7])
        title_template = random.choice(ISSUE_TITLES[cat["id"]])
        title = title_template.format(area=area, aqi=aqi, n=n)

        issues.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon + random.uniform(-0.02, 0.02), lat + random.uniform(-0.02, 0.02)],
            },
            "properties": {
                "id": f"issue_{i:04d}",
                "category": cat["id"],
                "category_label": cat["label"],
                "color": cat["color"],
                "icon": cat["icon"],
                "title": title,
                "description": f"{cat['label']} issue reported in {area}. Severity: {severity}. Status: {status}.",
                "severity": severity,
                "status": status,
                "area": area,
                "lat": lat,
                "lon": lon,
                "helpline": HELPLINES.get(cat["id"], ""),
                "source": "Citizen Report / Mock Data",
                "source_url": "https://delhi.gov.in/",
                "confidence": round(random.uniform(0.6, 0.99), 2),
                "reported_at": (now - timedelta(hours=random.randint(1, 168))).isoformat(),
                "estimated_resolution": (now + timedelta(days=random.randint(1, 14))).isoformat(),
                "upvotes": random.randint(0, 50),
            },
        })
    return issues


_ISSUE_CACHE = None


def _get_issues():
    global _ISSUE_CACHE
    if _ISSUE_CACHE is None:
        _ISSUE_CACHE = _generate_issues(80)
    return _ISSUE_CACHE


@router.get("/api/map/issues")
async def get_issues(
    category: str = "",
    severity: str = "",
    status: str = "",
    limit: int = 200,
):
    issues = _get_issues()
    if category:
        issues = [i for i in issues if i["properties"]["category"] == category]
    if severity:
        issues = [i for i in issues if i["properties"]["severity"] == severity]
    if status:
        issues = [i for i in issues if i["properties"]["status"] == status]
    return {
        "type": "FeatureCollection",
        "features": issues[:limit],
        "total": len(issues),
    }


@router.get("/api/map/categories")
async def get_categories():
    return {"categories": CATEGORIES}


@router.get("/api/nearby")
async def get_nearby(
    lat: float = Query(28.6139),
    lon: float = Query(77.2090),
    radius: float = Query(2.0),
    limit: int = Query(20),
):
    issues = _get_issues()
    nearby = []
    for issue in issues:
        p = issue["properties"]
        ilat, ilon = p["lat"], p["lon"]
        dist = math.sqrt((lat - ilat) ** 2 + (lon - ilon) ** 2) * 111
        if dist <= radius:
            nearby.append({**issue, "properties": {**p, "distance_km": round(dist, 2)}})
    nearby.sort(key=lambda x: x["properties"]["distance_km"])
    return {"features": nearby[:limit], "total": len(nearby)}


@router.get("/api/map/stats")
async def get_stats():
    issues = _get_issues()
    stats = {}
    for issue in issues:
        cat = issue["properties"]["category"]
        stats[cat] = stats.get(cat, 0) + 1
    return {
        "total_issues": len(issues),
        "by_category": stats,
        "active": sum(1 for i in issues if i["properties"]["status"] == "active"),
        "resolved": sum(1 for i in issues if i["properties"]["status"] == "resolved"),
    }
