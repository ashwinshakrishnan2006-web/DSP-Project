"""Temporal analysis of crime records: by hour, weekday and month."""
from __future__ import annotations
import pandas as pd

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Return a copy with hour, weekday, month and is_night columns.

    Rows whose date cannot be parsed are dropped.
    """
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out["hour"] = out[date_col].dt.hour
    out["weekday"] = out[date_col].dt.dayofweek
    out["month"] = out[date_col].dt.month
    out["is_night"] = ((out["hour"] >= 20) | (out["hour"] < 5)).astype(int)
    return out


def _counts(series: pd.Series, index) -> pd.Series:
    return series.value_counts().reindex(index, fill_value=0).sort_index()


def crimes_by_hour(df: pd.DataFrame) -> pd.Series:
    return _counts(df["hour"], range(24))


def crimes_by_weekday(df: pd.DataFrame) -> pd.Series:
    s = _counts(df["weekday"], range(7))
    s.index = WEEKDAYS
    return s


def crimes_by_month(df: pd.DataFrame) -> pd.Series:
    return _counts(df["month"], range(1, 13))


def peak_period(series: pd.Series):
    """Return (label, count) of the busiest period in a counts series."""
    if series.empty or series.sum() == 0:
        return None, 0
    return series.idxmax(), int(series.max())


def night_fraction(df: pd.DataFrame) -> float:
    return float(df["is_night"].mean()) if len(df) else 0.0
