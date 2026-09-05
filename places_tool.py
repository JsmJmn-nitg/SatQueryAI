# places_tool.py
from __future__ import annotations
import os
import requests
from functools import lru_cache

GOOGLE_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()

@lru_cache(maxsize=512)
def autocomplete_places(query: str, limit: int = 5):
    query = (query or "").strip()
    if len(query) < 3:
        return []

    if GOOGLE_KEY:
        return _google_autocomplete(query, limit=limit)
    else:
        return _osm_autocomplete(query, limit=limit)

def geocode_place(label: str):
    """
    Returns (lat, lon, formatted_label).
    If GOOGLE key present: uses Google Geocoding API.
    Else: uses OSM Nominatim search.
    """
    label = (label or "").strip()
    if not label:
        return None, None, ""

    if GOOGLE_KEY:
        return _google_geocode(label)
    else:
        return _osm_geocode(label)


def _google_autocomplete(query: str, limit: int = 5):
    url = "https://maps.googleapis.com/maps/api/place/autocomplete/json"
    params = {
        "input": query,
        "key": GOOGLE_KEY,
        "types": "geocode",
    }
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    data = r.json()
    preds = data.get("predictions", [])[:limit]
    # Return description strings (simple UX)
    return [p.get("description", "").strip() for p in preds if p.get("description")]

def _google_geocode(label: str):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": label, "key": GOOGLE_KEY}
    r = requests.get(url, params=params, timeout=8)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])
    if not results:
        return None, None, label
    loc = results[0]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"]), results[0].get("formatted_address", label)

def _osm_autocomplete(query: str, limit: int = 5):
    # Nominatim usage policy: keep requests low; we cache + only fire after 3 chars
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": query,
        "format": "json",
        "addressdetails": 1,
        "limit": limit,
    }
    headers = {"User-Agent": "SatQueryAI-Demo/1.0 (hackathon UI)"}
    r = requests.get(url, params=params, headers=headers, timeout=8)
    r.raise_for_status()
    items = r.json()
    return [it.get("display_name", "").strip() for it in items if it.get("display_name")]

def _osm_geocode(label: str):
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": label, "format": "json", "limit": 1}
    headers = {"User-Agent": "SatQueryAI-Demo/1.0 (hackathon UI)"}
    r = requests.get(url, params=params, headers=headers, timeout=8)
    r.raise_for_status()
    items = r.json()
    if not items:
        return None, None, label
    lat = float(items[0]["lat"])
    lon = float(items[0]["lon"])
    return lat, lon, items[0].get("display_name", label)
