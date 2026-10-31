"""Tests for the proximity alerting module."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.alerting.proximity_alert import ProximityAlertEngine, check_alert, simulate_coverage
from src.utils.haversine import haversine, haversine_vectorized


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_grid_df() -> pd.DataFrame:
    """Minimal grid DataFrame with Low/Medium/High cells."""
    return pd.DataFrame({
        "cell_lat": [41.80, 41.81, 41.82, 41.83, 41.84],
        "cell_lon": [-87.70, -87.69, -87.68, -87.67, -87.66],
        "centroid_lat": [41.8022, 41.8122, 41.8222, 41.8322, 41.8422],
        "centroid_lon": [-87.6977, -87.6877, -87.6777, -87.6677, -87.6577],
        "risk_label": ["Low", "Medium", "High", "High", "Low"],
        "incident_count": [5, 25, 80, 95, 3],
    })


# ---------------------------------------------------------------------------
# Haversine Tests
# ---------------------------------------------------------------------------

class TestHaversine:
    def test_zero_distance(self):
        assert haversine(41.85, -87.65, 41.85, -87.65) == pytest.approx(0.0)

    def test_known_distance(self):
        # Chicago O'Hare to Midway (roughly 26 km)
        d = haversine(41.9742, -87.9073, 41.7868, -87.7522)
        assert 20 < d < 35

    def test_vectorized_matches_scalar(self):
        lat2 = np.array([41.90, 41.80, 41.70])
        lon2 = np.array([-87.70, -87.65, -87.60])
        vec_result = haversine_vectorized(41.85, -87.65, lat2, lon2)
        scalar_results = [haversine(41.85, -87.65, la, lo) for la, lo in zip(lat2, lon2)]
        np.testing.assert_allclose(vec_result, scalar_results, rtol=1e-6)


# ---------------------------------------------------------------------------
# ProximityAlertEngine Tests
# ---------------------------------------------------------------------------

class TestProximityAlertEngine:
    def test_alert_triggered_when_close_to_high_risk(self):
        grid = _make_grid_df()
        engine = ProximityAlertEngine(grid, radius_km=2.0)
        # High-risk centroid at ~41.8222, -87.6777
        result = engine.check(41.8222, -87.6777)   # exactly on the centroid
        assert result.alert_triggered

    def test_no_alert_when_far_from_high_risk(self):
        grid = _make_grid_df()
        engine = ProximityAlertEngine(grid, radius_km=0.1)
        # Far from all High-risk centroids (10° away)
        result = engine.check(31.00, -77.00)
        assert not result.alert_triggered

    def test_nearest_distance_is_non_negative(self):
        grid = _make_grid_df()
        engine = ProximityAlertEngine(grid, radius_km=1.0)
        result = engine.check(41.85, -87.65)
        assert result.nearest_distance_km is not None
        assert result.nearest_distance_km >= 0

    def test_empty_grid_no_alert(self):
        empty_grid = pd.DataFrame(columns=["centroid_lat", "centroid_lon", "risk_label"])
        engine = ProximityAlertEngine(empty_grid, radius_km=1.0)
        result = engine.check(41.85, -87.65)
        assert not result.alert_triggered

    def test_check_alert_convenience_function(self):
        grid = _make_grid_df()
        result = check_alert(41.85, -87.65, grid, radius_km=5.0)
        assert hasattr(result, "alert_triggered")


class TestCoverageAnalysis:
    def test_coverage_returns_dataframe(self):
        grid = _make_grid_df()
        engine = ProximityAlertEngine(grid, radius_km=1.0)
        user_locs = pd.DataFrame({
            "latitude": [41.82, 41.85, 41.90],
            "longitude": [-87.68, -87.65, -87.62],
        })
        cov = engine.coverage_analysis(user_locs, radii_km=[0.5, 1.0, 2.0])
        assert len(cov) == 3
        assert "coverage_pct" in cov.columns

    def test_coverage_pct_between_0_and_100(self):
        grid = _make_grid_df()
        cov = simulate_coverage(grid, n_simulated=200)
        assert (cov["coverage_pct"] >= 0).all()
        assert (cov["coverage_pct"] <= 100).all()

    def test_larger_radius_gives_greater_or_equal_coverage(self):
        grid = _make_grid_df()
        cov = simulate_coverage(grid, n_simulated=500)
        # Coverage must be monotonically non-decreasing with radius
        coverages = cov.sort_values("radius_km")["coverage_pct"].values
        for i in range(len(coverages) - 1):
            assert coverages[i] <= coverages[i + 1] + 1e-6   # allow float noise
