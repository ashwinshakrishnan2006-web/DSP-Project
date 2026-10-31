"""Privacy helpers for preparing classroom demo datasets.

These helpers are conservative conveniences, not a guarantee of anonymisation.
Review every exported dataset before sharing it.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_SENSITIVE_COLUMNS = {
    "name", "full_name", "phone", "phone_number", "email",
    "address", "home_address", "victim_name", "suspect_name",
    "national_id", "person_id",
}


def remove_sensitive_columns(
    df: pd.DataFrame,
    extra_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Drop likely direct-identifier columns; return a copy and dropped names."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    requested = {str(name).strip().lower() for name in DEFAULT_SENSITIVE_COLUMNS}
    requested.update(str(name).strip().lower() for name in (extra_columns or []))

    drop = [
        column for column in df.columns
        if str(column).strip().lower() in requested
    ]
    return df.drop(columns=drop).copy(), [str(column) for column in drop]


def round_coordinates(
    df: pd.DataFrame,
    decimals: int = 2,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
) -> pd.DataFrame:
    """Return a copy with coordinates rounded to reduce location precision."""
    if decimals < 0 or decimals > 6:
        raise ValueError("decimals must be between 0 and 6")

    missing = [
        name for name in (latitude_column, longitude_column)
        if name not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing coordinate column(s): {', '.join(missing)}")

    out = df.copy()
    out[latitude_column] = pd.to_numeric(
        out[latitude_column], errors="coerce"
    ).round(decimals)
    out[longitude_column] = pd.to_numeric(
        out[longitude_column], errors="coerce"
    ).round(decimals)
    return out
