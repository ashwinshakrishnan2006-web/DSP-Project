"""
CrimeSense – DBSCAN Hotspot Detection
======================================
Applies DBSCAN with the haversine metric to detect spatial crime hotspots
in cleaned crime records.

Key design decisions (matching the paper):
- eps    = 0.3 km  (300 m neighbourhood radius)
- min_samples = 50 (minimum cluster density)
- metric = haversine (great-circle distance, expects radians)
- algorithm = ball_tree (required for haversine in sklearn)

Produces:
- cluster labels for every input record (-1 = noise)
- hotspot summary DataFrame (cluster id, centroid, size, crime-type mix)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EARTH_RADIUS_KM = 6371.0
DEFAULT_EPS_KM = 0.3          # 300 m
DEFAULT_MIN_SAMPLES = 50


@dataclass
class DBSCANConfig:
    """Hyper-parameters for the DBSCAN hotspot detector."""
    eps_km: float = DEFAULT_EPS_KM
    min_samples: int = DEFAULT_MIN_SAMPLES
    n_jobs: int = -1            # Use all CPU cores


@dataclass
class HotspotSummary:
    """Results returned by DBSCANHotspotDetector.fit()."""
    labels: np.ndarray                      # per-record cluster id (-1 = noise)
    n_clusters: int                         # number of hotspot clusters found
    noise_fraction: float                   # fraction of records labelled noise
    hotspots: pd.DataFrame                  # cluster-level summary table
    config: DBSCANConfig = field(default_factory=DBSCANConfig)


class DBSCANHotspotDetector:
    """
    Detect spatial crime hotspots using DBSCAN.

    Parameters
    ----------
    config : DBSCANConfig, optional
        Hyper-parameters. Defaults to paper values (eps=0.3 km, min_samples=50).
    """

    def __init__(self, config: Optional[DBSCANConfig] = None):
        self.config = config or DBSCANConfig()
        self._model: Optional[DBSCAN] = None

    def fit(self, df: pd.DataFrame) -> HotspotSummary:
        """
        Run DBSCAN on the crime records.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned crime data with 'latitude' and 'longitude' columns.

        Returns
        -------
        HotspotSummary
        """
        coords = df[["latitude", "longitude"]].to_numpy()
        coords_rad = np.radians(coords)

        eps_rad = self.config.eps_km / EARTH_RADIUS_KM

        logger.info(
            "Running DBSCAN: eps=%.1f km (%.6f rad), min_samples=%d, n=%d",
            self.config.eps_km, eps_rad, self.config.min_samples, len(coords_rad),
        )

        self._model = DBSCAN(
            eps=eps_rad,
            min_samples=self.config.min_samples,
            metric="haversine",
            algorithm="ball_tree",
            n_jobs=self.config.n_jobs,
        )
        labels = self._model.fit_predict(coords_rad)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_fraction = float((labels == -1).sum()) / len(labels)

        logger.info(
            "DBSCAN complete: %d clusters, %.1f%% noise",
            n_clusters, noise_fraction * 100,
        )

        hotspots = self._build_summary(df, labels)
        return HotspotSummary(
            labels=labels,
            n_clusters=n_clusters,
            noise_fraction=noise_fraction,
            hotspots=hotspots,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
        """Build a cluster-level summary DataFrame."""
        df = df.copy()
        df["cluster_id"] = labels

        records = []
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            subset = df[df["cluster_id"] == cid]
            row = {
                "cluster_id": cid,
                "size": len(subset),
                "centroid_lat": subset["latitude"].mean(),
                "centroid_lon": subset["longitude"].mean(),
                "lat_std": subset["latitude"].std(),
                "lon_std": subset["longitude"].std(),
            }
            # Crime type diversity
            if "primary_type" in subset.columns:
                row["crime_types"] = subset["primary_type"].nunique()
                row["top_crime"] = subset["primary_type"].mode().iloc[0]
            # Night fraction
            if "is_night" in subset.columns:
                row["night_fraction"] = subset["is_night"].mean()
            # Arrest fraction
            if "arrest" in subset.columns:
                row["arrest_fraction"] = subset["arrest"].mean()
            records.append(row)

        return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def detect_hotspots(
    df: pd.DataFrame,
    eps_km: float = DEFAULT_EPS_KM,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> HotspotSummary:
    """
    One-call DBSCAN hotspot detection.

    Parameters
    ----------
    df : pd.DataFrame   Cleaned crime DataFrame.
    eps_km : float      Neighbourhood radius in km.
    min_samples : int   Minimum points to form a dense region.

    Returns
    -------
    HotspotSummary
    """
    config = DBSCANConfig(eps_km=eps_km, min_samples=min_samples)
    return DBSCANHotspotDetector(config).fit(df)
