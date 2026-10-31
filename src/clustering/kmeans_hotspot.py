"""
CrimeSense – K-Means Hotspot Detection (comparison baseline)
=============================================================
Partitions crime records into k geographic clusters using K-Means.
K-Means acts as a comparison baseline against DBSCAN:
- It covers the entire city (no noise points).
- It requires a preset k.
- Cluster boundaries are Voronoi-based, not density-driven.

Produces:
- cluster labels for every input record
- cluster summary DataFrame (centroid, size, crime-type mix)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans

logger = logging.getLogger(__name__)

DEFAULT_K = 32          # Match number of DBSCAN clusters from paper


@dataclass
class KMeansConfig:
    """Hyper-parameters for K-Means hotspot detection."""
    k: int = DEFAULT_K
    random_state: int = 42
    n_init: int = 10
    max_iter: int = 300
    use_minibatch: bool = False    # Use MiniBatchKMeans for large datasets


@dataclass
class KMeansResult:
    """Results returned by KMeansHotspotDetector.fit()."""
    labels: np.ndarray
    centroids: np.ndarray           # shape (k, 2) – lat/lon
    inertia: float                  # within-cluster sum of squares
    hotspots: pd.DataFrame
    config: KMeansConfig = field(default_factory=KMeansConfig)


class KMeansHotspotDetector:
    """
    Detect spatial crime clusters using K-Means.

    Parameters
    ----------
    config : KMeansConfig, optional
    """

    def __init__(self, config: Optional[KMeansConfig] = None):
        self.config = config or KMeansConfig()
        self._model = None

    def fit(self, df: pd.DataFrame) -> KMeansResult:
        """
        Run K-Means on the crime records.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned crime data with 'latitude' and 'longitude' columns.

        Returns
        -------
        KMeansResult
        """
        coords = df[["latitude", "longitude"]].to_numpy()

        logger.info(
            "Running K-Means: k=%d, n=%d, mini_batch=%s",
            self.config.k, len(coords), self.config.use_minibatch,
        )

        if self.config.use_minibatch:
            self._model = MiniBatchKMeans(
                n_clusters=self.config.k,
                random_state=self.config.random_state,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
            )
        else:
            self._model = KMeans(
                n_clusters=self.config.k,
                random_state=self.config.random_state,
                n_init=self.config.n_init,
                max_iter=self.config.max_iter,
            )

        labels = self._model.fit_predict(coords)
        logger.info("K-Means complete. Inertia: %.2f", self._model.inertia_)

        hotspots = self._build_summary(df, labels, self._model.cluster_centers_)
        return KMeansResult(
            labels=labels,
            centroids=self._model.cluster_centers_,
            inertia=self._model.inertia_,
            hotspots=hotspots,
            config=self.config,
        )

    def elbow_analysis(self, df: pd.DataFrame, k_range: range = range(5, 51, 5)) -> dict:
        """
        Compute inertia for a range of k values (Elbow method).

        Parameters
        ----------
        df : pd.DataFrame   Cleaned crime data.
        k_range : range     Range of k values to evaluate.

        Returns
        -------
        dict  {k: inertia}
        """
        coords = df[["latitude", "longitude"]].to_numpy()
        results = {}
        for k in k_range:
            km = KMeans(
                n_clusters=k,
                random_state=self.config.random_state,
                n_init=5,
                max_iter=100,
            )
            km.fit(coords)
            results[k] = km.inertia_
            logger.info("Elbow k=%d → inertia=%.2f", k, km.inertia_)
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(
        df: pd.DataFrame,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> pd.DataFrame:
        """Build a cluster-level summary DataFrame."""
        df = df.copy()
        df["cluster_id"] = labels

        records = []
        for cid in range(centroids.shape[0]):
            subset = df[df["cluster_id"] == cid]
            row = {
                "cluster_id": cid,
                "size": len(subset),
                "centroid_lat": centroids[cid, 0],
                "centroid_lon": centroids[cid, 1],
            }
            if "primary_type" in subset.columns and len(subset) > 0:
                row["crime_types"] = subset["primary_type"].nunique()
                row["top_crime"] = subset["primary_type"].mode().iloc[0]
            if "is_night" in subset.columns:
                row["night_fraction"] = subset["is_night"].mean()
            if "arrest" in subset.columns:
                row["arrest_fraction"] = subset["arrest"].mean()
            records.append(row)

        return pd.DataFrame(records).sort_values("size", ascending=False)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def kmeans_hotspots(df: pd.DataFrame, k: int = DEFAULT_K) -> KMeansResult:
    """One-call K-Means hotspot detection."""
    return KMeansHotspotDetector(KMeansConfig(k=k)).fit(df)
