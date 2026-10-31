"""
CrimeSense – Proximity Alert Module
=====================================
Evaluates whether a user's current location is within a configurable
radius of any High-risk grid cell centroid.

Alert function (Eq. 1 from the paper):

    a(u) = 1  if  min dist(u, r) ≤ θ  for r ∈ H
    a(u) = 0  otherwise

where
  u  = user location (lat, lon)
  H  = set of High-risk cell centroids
  θ  = alert radius in km (default 1.0 km)
  dist(·,·) = haversine great-circle distance

Coverage analysis
-----------------
Alert coverage is defined as the fraction of simulated user locations
that would trigger an alert at a given radius:

    coverage(θ) = |{u : a(u,θ) = 1}| / |U|

The paper reports coverage at θ ∈ {0.5, 1.0, 2.0} km.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.haversine import haversine, haversine_vectorized

logger = logging.getLogger(__name__)

DEFAULT_RADIUS_KM = 1.0
COVERAGE_RADII_KM = [0.5, 1.0, 2.0]   # as in the paper


@dataclass
class AlertResult:
    """Result of a single proximity check."""
    user_lat: float
    user_lon: float
    alert_triggered: bool
    nearest_high_risk_lat: Optional[float]
    nearest_high_risk_lon: Optional[float]
    nearest_distance_km: Optional[float]
    radius_km: float


class ProximityAlertEngine:
    """
    Continuously evaluates user proximity to High-risk grid cells.

    Parameters
    ----------
    grid_df : pd.DataFrame
        Grid DataFrame produced by RiskLabeler.fit().  Must contain columns:
        'centroid_lat', 'centroid_lon', 'risk_label'.
    radius_km : float
        Alert radius in km. Default 1.0 km.
    """

    def __init__(self, grid_df: pd.DataFrame, radius_km: float = DEFAULT_RADIUS_KM):
        self.radius_km = radius_km
        self._high_risk = self._extract_high_risk(grid_df)
        logger.info(
            "ProximityAlertEngine: %d High-risk cells loaded, radius=%.1f km",
            len(self._high_risk), self.radius_km,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, user_lat: float, user_lon: float) -> AlertResult:
        """
        Check whether a user location is within the alert radius of any
        High-risk cell.

        Parameters
        ----------
        user_lat, user_lon : float   User coordinates in decimal degrees.

        Returns
        -------
        AlertResult
        """
        if self._high_risk.empty:
            return AlertResult(
                user_lat=user_lat, user_lon=user_lon,
                alert_triggered=False,
                nearest_high_risk_lat=None, nearest_high_risk_lon=None,
                nearest_distance_km=None, radius_km=self.radius_km,
            )

        distances = haversine_vectorized(
            user_lat, user_lon,
            self._high_risk["centroid_lat"].values,
            self._high_risk["centroid_lon"].values,
        )
        min_idx = int(np.argmin(distances))
        min_dist = float(distances[min_idx])

        nearest = self._high_risk.iloc[min_idx]
        return AlertResult(
            user_lat=user_lat,
            user_lon=user_lon,
            alert_triggered=min_dist <= self.radius_km,
            nearest_high_risk_lat=float(nearest["centroid_lat"]),
            nearest_high_risk_lon=float(nearest["centroid_lon"]),
            nearest_distance_km=min_dist,
            radius_km=self.radius_km,
        )

    def coverage_analysis(
        self,
        user_locations: pd.DataFrame,
        radii_km: Optional[list[float]] = None,
    ) -> pd.DataFrame:
        """
        Compute alert coverage for a set of simulated user locations.

        Parameters
        ----------
        user_locations : pd.DataFrame
            Must contain 'latitude' and 'longitude' columns.
        radii_km : list[float], optional
            Radii to evaluate. Defaults to [0.5, 1.0, 2.0].

        Returns
        -------
        pd.DataFrame   Columns: radius_km, n_alerted, total, coverage_pct.
        """
        radii_km = radii_km or COVERAGE_RADII_KM

        if self._high_risk.empty:
            return pd.DataFrame({"radius_km": radii_km, "coverage_pct": [0.0] * len(radii_km)})

        high_lats = self._high_risk["centroid_lat"].values
        high_lons = self._high_risk["centroid_lon"].values

        # Pre-compute minimum distance for every user location
        min_distances = np.array([
            haversine_vectorized(row["latitude"], row["longitude"], high_lats, high_lons).min()
            for _, row in user_locations.iterrows()
        ])

        records = []
        for r in radii_km:
            n_alerted = int((min_distances <= r).sum())
            total = len(min_distances)
            records.append({
                "radius_km": r,
                "n_alerted": n_alerted,
                "total": total,
                "coverage_pct": round(100 * n_alerted / total, 2) if total > 0 else 0.0,
            })
            logger.info("Coverage @ %.1f km: %.1f%%", r, records[-1]["coverage_pct"])

        return pd.DataFrame(records)

    def update_grid(self, grid_df: pd.DataFrame) -> None:
        """Hot-update the High-risk cell list (e.g., after new approved reports)."""
        self._high_risk = self._extract_high_risk(grid_df)
        logger.info("ProximityAlertEngine: updated to %d High-risk cells", len(self._high_risk))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_high_risk(grid_df: pd.DataFrame) -> pd.DataFrame:
        """Filter and return only High-risk rows."""
        return (
            grid_df[grid_df["risk_label"] == "High"][["centroid_lat", "centroid_lon"]]
            .dropna()
            .reset_index(drop=True)
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def check_alert(
    user_lat: float,
    user_lon: float,
    grid_df: pd.DataFrame,
    radius_km: float = DEFAULT_RADIUS_KM,
) -> AlertResult:
    """One-call proximity alert check."""
    engine = ProximityAlertEngine(grid_df, radius_km=radius_km)
    return engine.check(user_lat, user_lon)


def simulate_coverage(
    grid_df: pd.DataFrame,
    n_simulated: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate random user locations within Chicago and compute coverage.

    Parameters
    ----------
    grid_df : pd.DataFrame
    n_simulated : int   Number of random user locations.
    seed : int

    Returns
    -------
    pd.DataFrame  Coverage table for radii [0.5, 1.0, 2.0] km.
    """
    rng = np.random.default_rng(seed)
    lats = rng.uniform(41.60, 42.10, n_simulated)
    lons = rng.uniform(-87.95, -87.50, n_simulated)
    user_locs = pd.DataFrame({"latitude": lats, "longitude": lons})

    engine = ProximityAlertEngine(grid_df)
    return engine.coverage_analysis(user_locs)
