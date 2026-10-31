import pandas as pd
import pytest

from src.privacy import remove_sensitive_columns, round_coordinates
from src.quality import audit_dataset


def test_audit_dataset_reports_missing_duplicates_and_bad_coordinates():
    frame = pd.DataFrame({
        "latitude": [12.9, 999.0, 12.9],
        "longitude": [79.1, 79.0, 79.1],
        "victim_name": ["A", None, "A"],
    })
    details, summary = audit_dataset(frame)

    assert summary["row_count"] == 3
    assert summary["duplicate_rows"] == 1
    assert summary["invalid_coordinate_rows"] == 1
    assert "victim_name" in summary["possible_identifier_columns"]
    assert "missing_percent" in details.columns


def test_remove_sensitive_columns_returns_copy_and_names():
    frame = pd.DataFrame({"crime_type": ["theft"], "email": ["a@example.com"]})
    cleaned, dropped = remove_sensitive_columns(frame)

    assert list(cleaned.columns) == ["crime_type"]
    assert dropped == ["email"]
    assert "email" in frame.columns  # input is not mutated


def test_remove_sensitive_columns_accepts_extra_names():
    frame = pd.DataFrame({"case_ref": ["X1"], "crime_type": ["theft"]})
    cleaned, dropped = remove_sensitive_columns(frame, ["case_ref"])

    assert "case_ref" not in cleaned.columns
    assert dropped == ["case_ref"]


def test_round_coordinates():
    frame = pd.DataFrame({"latitude": [12.34567], "longitude": [79.12345]})
    rounded = round_coordinates(frame, decimals=2)

    assert rounded.loc[0, "latitude"] == pytest.approx(12.35)
    assert rounded.loc[0, "longitude"] == pytest.approx(79.12)


def test_round_coordinates_rejects_missing_columns():
    with pytest.raises(ValueError):
        round_coordinates(pd.DataFrame({"lat": [12.3]}))
