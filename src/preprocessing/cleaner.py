"""
CrimeSense – Data Preprocessing Module
=======================================
Cleans and transforms raw Chicago crime CSV records into a standardised
DataFrame ready for spatial clustering and risk classification.

Steps performed:
1. Load CSV with robust datetime parsing.
2. Drop rows with missing/invalid latitude or longitude.
3. Deduplicate on Case Number (cross-batch safe).
4. Normalise Primary Type labels to UPPER_SNAKE_CASE.
5. Derive temporal features: hour, month, day_of_week, is_night.
6. Validate user-submitted reports with keyword boundary matching.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name mapping: Chicago Open Data Portal → internal names
# ---------------------------------------------------------------------------
COLUMN_MAP = {
    "Case Number": "case_number",
    "Date": "date",
    "Primary Type": "primary_type",
    "Description": "description",
    "Location Description": "location_description",
    "Arrest": "arrest",
    "Domestic": "domestic",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Community Area": "community_area",
    "Year": "year",
}

# Known crime-domain keywords for user-submitted report validation
CRIME_KEYWORDS: set[str] = {
    "theft", "battery", "assault", "burglary", "robbery", "homicide",
    "narcotics", "arson", "kidnapping", "vandalism", "fraud", "shooting",
    "stalking", "trespass", "weapon", "deceptive practice", "criminal damage",
    "motor vehicle", "sex offense", "intimidation", "interference",
}


class CrimeDataCleaner:
    """
    Loads and cleans a Chicago crime CSV file.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw CSV file.
    drop_cols : list[str], optional
        Additional columns to drop after loading.
    """

    def __init__(self, filepath: str | Path, drop_cols: Optional[list] = None):
        self.filepath = Path(filepath)
        self.drop_cols = drop_cols or []
        self._raw: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "CrimeDataCleaner":
        """Load the CSV into self._raw."""
        logger.info("Loading crime data from %s", self.filepath)
        self._raw = pd.read_csv(
            self.filepath,
            low_memory=False,
            dtype={"Case Number": str, "Community Area": str},
        )
        logger.info("Loaded %d raw records, %d columns", len(self._raw), len(self._raw.columns))
        return self

    def clean(self) -> "CrimeDataCleaner":
        """
        Run the full cleaning pipeline and store the result in self.df.

        Returns self for method chaining.
        """
        if self._raw is None:
            raise RuntimeError("Call .load() before .clean()")

        df = self._raw.copy()

        # 1. Rename to internal names (keep unmapped columns as-is)
        df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})

        # 2. Drop configured extra columns
        df = df.drop(columns=[c for c in self.drop_cols if c in df.columns], errors="ignore")

        # 3. Parse datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 4. Drop rows without valid coordinates
        before = len(df)
        df = df.dropna(subset=["latitude", "longitude"])
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        df = df.dropna(subset=["latitude", "longitude"])
        # Chicago bounding box guard
        df = df[
            df["latitude"].between(41.6, 42.1)
            & df["longitude"].between(-87.95, -87.5)
        ]
        logger.info("Dropped %d rows with invalid coordinates", before - len(df))

        # 5. Deduplicate on case_number
        if "case_number" in df.columns:
            before_dedup = len(df)
            df = df.drop_duplicates(subset=["case_number"], keep="first")
            logger.info("Dropped %d duplicate case numbers", before_dedup - len(df))

        # 6. Normalise primary_type
        if "primary_type" in df.columns:
            df["primary_type"] = (
                df["primary_type"]
                .str.strip()
                .str.upper()
                .str.replace(r"\s+", "_", regex=True)
            )

        # 7. Boolean coercion
        for bool_col in ("arrest", "domestic"):
            if bool_col in df.columns:
                df[bool_col] = df[bool_col].map(
                    {"True": True, "False": False, True: True, False: False}
                ).astype("boolean")

        # 8. Derive temporal features
        df = self._add_temporal_features(df)

        # 9. Reset index
        df = df.reset_index(drop=True)
        self.df = df
        logger.info("Cleaning complete. Final shape: %s", df.shape)
        return self

    def get(self) -> pd.DataFrame:
        """Return the cleaned DataFrame."""
        if self.df is None:
            raise RuntimeError("Call .load().clean() first.")
        return self.df

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive hour, month, day_of_week and is_night from the date column."""
        if "date" not in df.columns:
            return df
        df = df.copy()
        df["hour"] = df["date"].dt.hour
        df["month"] = df["date"].dt.month
        df["day_of_week"] = df["date"].dt.dayofweek  # 0=Monday
        # Night: 22:00–05:59 inclusive
        df["is_night"] = df["hour"].apply(lambda h: int(h >= 22 or h < 6))
        return df

    @staticmethod
    def validate_user_report(report: dict) -> tuple[bool, str]:
        """
        Keyword boundary matching to check that a user-submitted crime report
        belongs to the crime-data domain.

        Parameters
        ----------
        report : dict
            Expected keys: 'primary_type', 'description' (optional),
            'latitude', 'longitude'.

        Returns
        -------
        (is_valid, reason) : tuple[bool, str]
        """
        errors: list[str] = []

        # Coordinate check
        try:
            lat = float(report.get("latitude", ""))
            lon = float(report.get("longitude", ""))
        except (TypeError, ValueError):
            errors.append("Missing or non-numeric latitude/longitude.")
            return False, "; ".join(errors)

        if not (41.6 <= lat <= 42.1 and -87.95 <= lon <= -87.5):
            errors.append("Coordinates outside Chicago bounding box.")

        # Crime-domain keyword check
        text = " ".join(
            str(report.get(k, ""))
            for k in ("primary_type", "description", "location_description")
        ).lower()

        matched = any(
            re.search(r"\b" + re.escape(kw) + r"\b", text)
            for kw in CRIME_KEYWORDS
        )
        if not matched:
            errors.append(
                "Report does not appear to describe a recognised crime type. "
                "Please include a valid Primary Type or description."
            )

        if errors:
            return False, "; ".join(errors)
        return True, "OK"


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def load_and_clean(filepath: str | Path, **kwargs) -> pd.DataFrame:
    """
    One-call convenience wrapper.

    Parameters
    ----------
    filepath : str or Path
    **kwargs : passed to CrimeDataCleaner.__init__

    Returns
    -------
    pd.DataFrame  Cleaned crime data.
    """
    return CrimeDataCleaner(filepath, **kwargs).load().clean().get()
