"""Validation of user-submitted crime reports before they enter the admin queue."""
from __future__ import annotations
import re
from datetime import datetime

CRIME_KEYWORDS = {
    "theft", "robbery", "burglary", "assault", "battery", "murder", "homicide",
    "kidnap", "harassment", "stalking", "vandalism", "fraud", "snatching",
    "chain", "mugging", "molestation", "stabbing", "shooting", "arson", "narcotics",
}
MAX_TEXT = 1000


def has_crime_keyword(text: str) -> bool:
    """Whole-word keyword match (boundary matching, so 'battery' != 'batteryless')."""
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & CRIME_KEYWORDS)


def validate_report(report: dict, now: datetime | None = None) -> tuple[bool, list[str]]:
    """Return (is_valid, errors). Valid reports go to the pending queue."""
    now = now or datetime.now()
    errors = []
    lat, lon = report.get("latitude"), report.get("longitude")
    if not isinstance(lat, (int, float)) or not -90 <= lat <= 90:
        errors.append("latitude missing or out of range")
    if not isinstance(lon, (int, float)) or not -180 <= lon <= 180:
        errors.append("longitude missing or out of range")

    text = (report.get("description") or "").strip()
    if not text:
        errors.append("description is empty")
    elif len(text) > MAX_TEXT:
        errors.append(f"description exceeds {MAX_TEXT} characters")
    elif not (has_crime_keyword(text) or has_crime_keyword(report.get("category") or "")):
        errors.append("content does not appear to describe a crime")

    when = report.get("date")
    if when is not None:
        try:
            d = when if isinstance(when, datetime) else datetime.fromisoformat(str(when))
            if d > now:
                errors.append("incident date is in the future")
        except ValueError:
            errors.append("incident date is not a valid ISO date")
    else:
        errors.append("incident date is missing")
    return (not errors), errors
