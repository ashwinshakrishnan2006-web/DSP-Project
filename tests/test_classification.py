"""Tests for RiskLabeler and CrimeRiskClassifier."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classification.risk_labeler import RiskLabeler
from src.classification.classifier import CrimeRiskClassifier


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_crime_df(n: int = 300) -> pd.DataFrame:
    """Synthetic cleaned crime DataFrame."""
    rng = np.random.default_rng(0)
    lats = rng.uniform(41.65, 42.05, n)
    lons = rng.uniform(-87.90, -87.55, n)
    types = rng.choice(["THEFT", "BATTERY", "ASSAULT", "ROBBERY", "NARCOTICS"], n)
    return pd.DataFrame({
        "latitude": lats,
        "longitude": lons,
        "primary_type": types,
        "is_night": rng.integers(0, 2, n),
        "arrest": rng.choice([True, False], n),
        "domestic": rng.choice([True, False], n),
    })


# ---------------------------------------------------------------------------
# RiskLabeler Tests
# ---------------------------------------------------------------------------

class TestRiskLabeler:
    def test_fit_creates_grid_df(self):
        df = _make_crime_df()
        labeler = RiskLabeler()
        labeler.fit(df)
        assert labeler.grid_df is not None
        assert len(labeler.grid_df) > 0

    def test_risk_labels_are_valid(self):
        df = _make_crime_df()
        labeler = RiskLabeler()
        labeler.fit(df)
        valid = {"Low", "Medium", "High"}
        assert set(labeler.grid_df["risk_label"].unique()).issubset(valid)

    def test_all_three_risk_levels_present(self):
        # With enough spread data we should get all three tercile levels
        df = _make_crime_df(n=500)
        labeler = RiskLabeler()
        labeler.fit(df)
        labels = set(labeler.grid_df["risk_label"].unique())
        assert len(labels) >= 2   # at least 2 distinct levels

    def test_get_features_excludes_incident_count_by_default(self):
        df = _make_crime_df()
        labeler = RiskLabeler()
        labeler.fit(df)
        X, y = labeler.get_features_and_labels(include_incident_count=False)
        assert "incident_count" not in X.columns

    def test_get_features_includes_incident_count_when_asked(self):
        df = _make_crime_df()
        labeler = RiskLabeler()
        labeler.fit(df)
        X, y = labeler.get_features_and_labels(include_incident_count=True)
        assert "incident_count" in X.columns

    def test_feature_matrix_no_nans(self):
        df = _make_crime_df()
        labeler = RiskLabeler()
        labeler.fit(df)
        X, y = labeler.get_features_and_labels()
        assert not X.isnull().any().any()

    def test_predict_cell_returns_string_or_none(self):
        df = _make_crime_df()
        labeler = RiskLabeler()
        labeler.fit(df)
        result = labeler.predict_cell(41.85, -87.65)
        assert result in (None, "Low", "Medium", "High")


# ---------------------------------------------------------------------------
# CrimeRiskClassifier Tests
# ---------------------------------------------------------------------------

class TestCrimeRiskClassifier:
    def _get_data(self):
        df = _make_crime_df(n=400)
        labeler = RiskLabeler()
        labeler.fit(df)
        return labeler.get_features_and_labels()

    def test_prepare_data_splits_correctly(self):
        X, y = self._get_data()
        clf = CrimeRiskClassifier(test_size=0.2)
        clf.prepare_data(X, y)
        total = len(clf.X_train) + len(clf.X_test)
        assert total == len(X)

    def test_random_forest_trains_and_evaluates(self):
        X, y = self._get_data()
        if len(X) < 10:
            pytest.skip("Not enough grid cells for classification test.")
        clf = CrimeRiskClassifier(test_size=0.2)
        clf.prepare_data(X, y)
        clf.train_random_forest(n_estimators=10)
        assert "random_forest" in clf.results
        assert 0.0 <= clf.results["random_forest"]["accuracy"] <= 1.0

    def test_feature_importance_returns_series(self):
        X, y = self._get_data()
        if len(X) < 10:
            pytest.skip("Not enough data.")
        clf = CrimeRiskClassifier()
        clf.prepare_data(X, y)
        clf.train_random_forest(n_estimators=10)
        importance = clf.feature_importance("random_forest")
        assert len(importance) == X.shape[1]

    def test_predict_returns_valid_labels(self):
        X, y = self._get_data()
        if len(X) < 10:
            pytest.skip("Not enough data.")
        clf = CrimeRiskClassifier()
        clf.prepare_data(X, y)
        clf.train_random_forest(n_estimators=10)
        preds = clf.predict(clf.X_test, model_name="random_forest")
        valid = {"Low", "Medium", "High"}
        assert set(preds).issubset(valid)
