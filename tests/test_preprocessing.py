"""Tests for the data preprocessing module."""
import io
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.cleaner import CrimeDataCleaner, load_and_clean


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_sample_csv(tmp_path: Path) -> Path:
    """Write a minimal Chicago-format CSV to a temp path."""
    data = {
        "Case Number": ["JG000001", "JG000002", "JG000001", "JG000003", "JG000004"],
        "Date": [
            "01/15/2023 02:00:00 AM",
            "06/20/2023 14:30:00 PM",
            "01/15/2023 02:00:00 AM",    # duplicate
            "11/01/2023 23:00:00 PM",    # night hour
            "03/10/2023 09:00:00 AM",
        ],
        "Primary Type": ["THEFT", "battery", "THEFT", "ASSAULT", "ROBBERY"],
        "Arrest": [True, False, True, False, True],
        "Domestic": [False, True, False, False, False],
        "Latitude": [41.8500, 41.9000, 41.8500, 999.0, 41.7800],   # 999 → invalid
        "Longitude": [-87.6500, -87.7000, -87.6500, -87.6500, -87.8000],
    }
    df = pd.DataFrame(data)
    p = tmp_path / "test_crimes.csv"
    df.to_csv(p, index=False)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCrimeDataCleaner:
    def test_load_and_clean_returns_dataframe(self, tmp_path):
        csv_path = _make_sample_csv(tmp_path)
        cleaner = CrimeDataCleaner(csv_path)
        cleaner.load().clean()
        assert isinstance(cleaner.df, pd.DataFrame)

    def test_duplicate_removal(self, tmp_path):
        csv_path = _make_sample_csv(tmp_path)
        df = CrimeDataCleaner(csv_path).load().clean().get()
        # JG000001 appears twice; should be deduplicated
        assert df["case_number"].duplicated().sum() == 0

    def test_invalid_coordinates_dropped(self, tmp_path):
        csv_path = _make_sample_csv(tmp_path)
        df = CrimeDataCleaner(csv_path).load().clean().get()
        # Record with lat=999 should be dropped
        assert (df["latitude"] > 100).sum() == 0

    def test_primary_type_normalised(self, tmp_path):
        csv_path = _make_sample_csv(tmp_path)
        df = CrimeDataCleaner(csv_path).load().clean().get()
        # All types must be upper-case (no lowercase)
        assert all(pt == pt.upper() for pt in df["primary_type"].dropna())

    def test_temporal_features_created(self, tmp_path):
        csv_path = _make_sample_csv(tmp_path)
        df = CrimeDataCleaner(csv_path).load().clean().get()
        for col in ("hour", "month", "day_of_week", "is_night"):
            assert col in df.columns, f"Missing column: {col}"

    def test_is_night_correct(self, tmp_path):
        csv_path = _make_sample_csv(tmp_path)
        df = CrimeDataCleaner(csv_path).load().clean().get()
        night_rows = df[df["hour"] >= 22]
        assert (night_rows["is_night"] == 1).all()
        early_rows = df[df["hour"] < 6]
        assert (early_rows["is_night"] == 1).all()


class TestValidateUserReport:
    def test_valid_report(self):
        report = {
            "primary_type": "THEFT",
            "description": "Simple theft from auto",
            "latitude": "41.85",
            "longitude": "-87.65",
        }
        ok, msg = CrimeDataCleaner.validate_user_report(report)
        assert ok, msg

    def test_missing_coordinates(self):
        report = {"primary_type": "BATTERY"}
        ok, msg = CrimeDataCleaner.validate_user_report(report)
        assert not ok

    def test_out_of_bounds_coordinates(self):
        report = {
            "primary_type": "ASSAULT",
            "latitude": "35.00",     # outside Chicago bbox
            "longitude": "-87.65",
        }
        ok, msg = CrimeDataCleaner.validate_user_report(report)
        assert not ok

    def test_non_crime_description(self):
        report = {
            "primary_type": "PIZZA DELIVERY",
            "description": "Fast food order",
            "latitude": "41.85",
            "longitude": "-87.65",
        }
        ok, msg = CrimeDataCleaner.validate_user_report(report)
        assert not ok
