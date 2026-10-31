"""Dataset quality checks for the CrimeSense project.

These checks describe data completeness and coordinate validity. They do not
judge whether a crime report is true or whether a location is dangerous.
"""
from __future__ import annotations

import pandas as pd

SENSITIVE_NAME_HINTS = (
    "name", "phone", "email", "address", "victim", "suspect",
    "person_id", "national_id",
)


def audit_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return per-column quality details and a dataset-level summary."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    rows = []
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        rows.append({
            "column": str(column),
            "dtype": str(series.dtype),
            "missing_count": missing,
            "missing_percent": round(100 * missing / len(df), 2) if len(df) else 0.0,
            "unique_values": unique,
            "possible_identifier": any(
                hint in str(column).strip().lower() for hint in SENSITIVE_NAME_HINTS
            ),
        })

    details = pd.DataFrame(rows, columns=[
        "column", "dtype", "missing_count", "missing_percent",
        "unique_values", "possible_identifier",
    ])

    duplicate_rows = int(df.duplicated().sum())
    summary = {
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "duplicate_rows": duplicate_rows,
        "columns_with_missing_values": int((df.isna().any()).sum()),
        "possible_identifier_columns": details.loc[
            details["possible_identifier"], "column"
        ].tolist(),
    }

    if {"latitude", "longitude"}.issubset(df.columns):
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        valid = lat.between(-90, 90) & lon.between(-180, 180)
        summary["invalid_coordinate_rows"] = int((~valid).sum())
    else:
        summary["invalid_coordinate_rows"] = None

    return details, summary
