"""Crime category statistics and summaries."""
from __future__ import annotations
import pandas as pd


def top_crime_types(df: pd.DataFrame, n: int = 10, type_col="primary_type") -> pd.DataFrame:
    vc = df[type_col].value_counts().head(n)
    out = vc.rename("count").to_frame()
    out["share_pct"] = (out["count"] / len(df) * 100).round(2)
    return out


def arrest_rate(df: pd.DataFrame, arrest_col="arrest") -> float:
    return float(df[arrest_col].astype(float).mean()) if len(df) else 0.0


def arrest_rate_by_type(df, type_col="primary_type", arrest_col="arrest") -> pd.Series:
    return (df.assign(_a=df[arrest_col].astype(float))
              .groupby(type_col)["_a"].mean().sort_values(ascending=False))


def dataset_summary(df: pd.DataFrame, type_col="primary_type", date_col="date") -> dict:
    dates = pd.to_datetime(df[date_col], errors="coerce")
    return {
        "records": len(df),
        "crime_types": int(df[type_col].nunique()),
        "date_min": dates.min(),
        "date_max": dates.max(),
        "arrest_rate": arrest_rate(df),
    }
