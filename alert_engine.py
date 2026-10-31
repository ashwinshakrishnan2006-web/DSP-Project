"""Haversine-based proximity alerting to high-risk zones."""
from __future__ import annotations
import math
from typing import Iterable, Sequence

EARTH_RADIUS_KM = 6371.0088
DEFAULT_RADIUS_KM = 1.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def nearest_zone(user: Sequence[float], zones: Iterable[Sequence[float]]):
    """Return (distance_km, zone) for the closest zone, or (None, None)."""
    best_d, best_z = None, None
    for z in zones:
        d = haversine_km(user[0], user[1], z[0], z[1])
        if best_d is None or d < best_d:
            best_d, best_z = d, z
    return best_d, best_z


def check_alert(user, high_risk_zones, radius_km: float = DEFAULT_RADIUS_KM) -> dict:
    """Eq. (1): alert = 1 if min distance to a high-risk zone <= radius."""
    if radius_km <= 0:
        raise ValueError("radius_km must be positive")
    d, z = nearest_zone(user, high_risk_zones)
    triggered = d is not None and d <= radius_km
    return {"alert": int(triggered), "distance_km": d, "zone": z if triggered else None}


def coverage_rate(users, high_risk_zones, radius_km: float) -> float:
    """Fraction of user locations that would trigger an alert."""
    users = list(users)
    if not users:
        return 0.0
    hits = sum(check_alert(u, high_risk_zones, radius_km)["alert"] for u in users)
    return hits / len(users)
