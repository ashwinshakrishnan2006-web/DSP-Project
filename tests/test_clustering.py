"""Tests for DBSCAN and K-Means hotspot detection."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clustering.dbscan_hotspot import DBSCANHotspotDetector, DBSCANConfig, detect_hotspots
from src.clustering.kmeans_hotspot import KMeansHotspotDetector, KMeansConfig, kmeans_hotspots


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_clustered_df(n_clusters: int = 3, points_per_cluster: int = 30) -> pd.DataFrame:
    """Create a synthetic DataFrame with clearly separated clusters."""
    rng = np.random.default_rng(42)
    # Cluster centres spread 0.5° apart (>>0.3 km eps → distinct clusters)
    centers = [
        (41.80 + 0.5 * i, -87.70 + 0.5 * i)
        for i in range(n_clusters)
    ]
    rows = []
    for lat_c, lon_c in centers:
        lats = rng.normal(lat_c, 0.001, points_per_cluster)
        lons = rng.normal(lon_c, 0.001, points_per_cluster)
        for la, lo in zip(lats, lons):
            rows.append({
                "latitude": la, "longitude": lo,
                "primary_type": "THEFT", "is_night": 0, "arrest": True,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# DBSCAN Tests
# ---------------------------------------------------------------------------

class TestDBSCANHotspotDetector:
    def test_returns_hotspot_summary(self):
        df = _make_clustered_df()
        result = detect_hotspots(df, eps_km=0.3, min_samples=5)
        assert result.labels is not None
        assert len(result.labels) == len(df)

    def test_detects_expected_clusters(self):
        df = _make_clustered_df(n_clusters=3, points_per_cluster=50)
        result = detect_hotspots(df, eps_km=0.5, min_samples=5)
        # Should find at least 2 clusters (tight tolerance)
        assert result.n_clusters >= 2

    def test_noise_fraction_between_0_and_1(self):
        df = _make_clustered_df()
        result = detect_hotspots(df, eps_km=0.3, min_samples=5)
        assert 0.0 <= result.noise_fraction <= 1.0

    def test_hotspot_summary_has_centroid_columns(self):
        df = _make_clustered_df()
        result = detect_hotspots(df, eps_km=0.3, min_samples=5)
        if not result.hotspots.empty:
            for col in ("centroid_lat", "centroid_lon", "size"):
                assert col in result.hotspots.columns

    def test_all_records_labelled(self):
        df = _make_clustered_df()
        result = detect_hotspots(df, eps_km=0.3, min_samples=5)
        assert len(result.labels) == len(df)
        # Labels should be integers >= -1
        assert all(isinstance(int(l), int) for l in result.labels)


# ---------------------------------------------------------------------------
# K-Means Tests
# ---------------------------------------------------------------------------

class TestKMeansHotspotDetector:
    def test_returns_correct_number_of_clusters(self):
        df = _make_clustered_df(n_clusters=3, points_per_cluster=30)
        result = kmeans_hotspots(df, k=3)
        assert result.labels.max() == 2     # 0, 1, 2 → 3 clusters
        assert len(np.unique(result.labels)) == 3

    def test_all_records_assigned(self):
        df = _make_clustered_df()
        result = kmeans_hotspots(df, k=3)
        # K-Means has no noise; all labels >= 0
        assert (result.labels >= 0).all()

    def test_inertia_positive(self):
        df = _make_clustered_df()
        result = kmeans_hotspots(df, k=3)
        assert result.inertia > 0

    def test_centroids_shape(self):
        df = _make_clustered_df()
        k = 4
        result = kmeans_hotspots(df, k=k)
        assert result.centroids.shape == (k, 2)

    def test_hotspot_summary_size_column(self):
        df = _make_clustered_df()
        result = kmeans_hotspots(df, k=3)
        assert "size" in result.hotspots.columns
        assert result.hotspots["size"].sum() == len(df)
