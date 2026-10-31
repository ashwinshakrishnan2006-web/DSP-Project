from __future__ import annotations
import pandas as pd

ALIASES = {
    "latitude": ["latitude", "lat", "y", "LATITUDE", "Latitude"],
    "longitude": ["longitude", "lon", "lng", "long", "x", "LONGITUDE", "Longitude"],
    "date": ["date", "datetime", "date_time", "reported_at", "incident_date", "Date"],
    "crime_type": ["crime_type", "type", "category", "primary_type", "Primary Type", "offense"],
}

def guess_column(columns, target):
    lookup = {str(c).strip().lower(): c for c in columns}
    for alias in ALIASES[target]:
        if alias.lower() in lookup:
            return lookup[alias.lower()]
    return None

def clean_crime_data(raw: pd.DataFrame, latitude_col: str, longitude_col: str,
                     date_col: str | None = None, type_col: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Return a cleaned copy and transparent cleaning statistics."""
    if latitude_col not in raw or longitude_col not in raw:
        raise ValueError("Choose valid latitude and longitude columns.")
    df = raw.copy()
    before = len(df)
    df["latitude"] = pd.to_numeric(df[latitude_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[longitude_col], errors="coerce")
    invalid = df["latitude"].isna() | df["longitude"].isna()
    invalid |= ~df["latitude"].between(-90, 90) | ~df["longitude"].between(-180, 180)
    invalid_count = int(invalid.sum())
    df = df.loc[~invalid].copy()
    dup_count = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    if date_col and date_col in df:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    else:
        df["date"] = pd.NaT
    if type_col and type_col in df:
        df["crime_type"] = df[type_col].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    else:
        df["crime_type"] = "Unspecified"
    stats = {"input_rows": before, "invalid_coordinate_rows": invalid_count,
             "duplicate_rows_removed": dup_count, "clean_rows": len(df)}
    return df, stats
