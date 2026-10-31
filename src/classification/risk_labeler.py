"""
CrimeSense – Grid-Based Risk Labeler
=====================================
Discretises Chicago into approximately 500 m grid cells and assigns
each cell a risk label (Low / Medium / High) using tercile cut-points
of per-cell incident counts.

Grid construction
-----------------
A 500 m grid cell in Chicago corresponds roughly to:
  Δlat ≈ 500 / 111_320  ≈ 0.004494°
  Δlon ≈ 500 / (111_320 × cos(41.85°))  ≈ 0.006014°

We bin latitude and longitude independently and assign each crime record
to the nearest cell centroid.

Risk labelling
--------------
1. Count incidents per cell.
2. Compute the 33rd and 67th percentile of the non-zero cell counts.
3. Assign:
   - Low    → count ≤ p33
   - Medium → p33 < count ≤ p67
   - High   → count > p67

This tercile rule is transparent and reproducible (see §III.C of the paper).

Leakage control
---------------
The incident_count feature is deliberately excluded from the feature set
passed to supervised classifiers, since it directly determines the label.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID_LAT_STEP = 0.004494   # ≈ 500 m in latitude degrees
GRID_LON_STEP = 0.006014   # ≈ 500 m in longitude degrees at Chicago latitude

RISK_LABELS = ["Low", "Medium", "High"]
RISK_LOW, RISK_MED, RISK_HIGH = RISK_LABELS


@dataclass
class GridConfig:
    lat_step: float = GRID_LAT_STEP
    lon_step: float = GRID_LON_STEP
    p_low: float = 33.0    # percentile for Low/Medium boundary
    p_high: float = 67.0   # percentile for Medium/High boundary


class RiskLabeler:
    """
    Assign Low / Medium / High risk labels to ~500 m grid cells.

    Parameters
    ----------
    config : GridConfig, optional
    """

    def __init__(self, config: Optional[GridConfig] = None):
        self.config = config or GridConfig()
        self.grid_df: Optional[pd.DataFrame] = None
        self._cut_low: Optional[float] = None
        self._cut_high: Optional[float] = None

    def fit(self, df: pd.DataFrame) -> "RiskLabeler":
        """
        Build the risk grid from cleaned crime records.

        Parameters
        ----------
        df : pd.DataFrame   Cleaned crime data.

        Returns
        -------
        self
        """
        cfg = self.config

        # 1. Assign each record to a grid cell
        df = df.copy()
        df["cell_lat"] = (df["latitude"] // cfg.lat_step) * cfg.lat_step
        df["cell_lon"] = (df["longitude"] // cfg.lon_step) * cfg.lon_step

        # 2. Aggregate features per cell
        agg = self._aggregate_cells(df)

        # 3. Compute tercile cut-points on non-zero cells
        counts = agg["incident_count"]
        self._cut_low = float(np.percentile(counts, cfg.p_low))
        self._cut_high = float(np.percentile(counts, cfg.p_high))
        logger.info(
            "Risk thresholds – Low≤%.0f, Medium≤%.0f, High>%.0f",
            self._cut_low, self._cut_high, self._cut_high,
        )

        # 4. Label cells
        agg["risk_label"] = agg["incident_count"].apply(self._label)
        agg["risk_code"] = agg["risk_label"].map({"Low": 0, "Medium": 1, "High": 2})

        # 5. Compute centroid of each cell
        agg["centroid_lat"] = agg["cell_lat"] + cfg.lat_step / 2
        agg["centroid_lon"] = agg["cell_lon"] + cfg.lon_step / 2

        self.grid_df = agg.reset_index(drop=True)
        logger.info(
            "Grid: %d cells | Low=%d, Medium=%d, High=%d",
            len(agg),
            (agg["risk_label"] == "Low").sum(),
            (agg["risk_label"] == "Medium").sum(),
            (agg["risk_label"] == "High").sum(),
        )
        return self

    def get_features_and_labels(self, include_incident_count: bool = False):
        """
        Return the feature matrix X and label vector y for supervised learning.

        Parameters
        ----------
        include_incident_count : bool
            If False (default, leakage-controlled), the incident_count column
            is excluded from X because it directly determines the risk label.

        Returns
        -------
        X : pd.DataFrame
        y : pd.Series   (str labels: Low / Medium / High)
        """
        if self.grid_df is None:
            raise RuntimeError("Call .fit() first.")

        feature_cols = [
            "crime_diversity",
            "night_fraction",
            "arrest_fraction",
            "domestic_fraction",
        ]
        if include_incident_count:
            feature_cols.insert(0, "incident_count")

        # Keep only columns that actually exist
        feature_cols = [c for c in feature_cols if c in self.grid_df.columns]
        X = self.grid_df[feature_cols].fillna(0.0)
        y = self.grid_df["risk_label"]
        return X, y

    def predict_cell(self, lat: float, lon: float) -> Optional[str]:
        """
        Look up the risk label for a given coordinate.

        Parameters
        ----------
        lat, lon : float   Decimal degrees.

        Returns
        -------
        str or None   'Low', 'Medium', 'High', or None if outside grid.
        """
        if self.grid_df is None:
            raise RuntimeError("Call .fit() first.")
        cfg = self.config
        cell_lat = (lat // cfg.lat_step) * cfg.lat_step
        cell_lon = (lon // cfg.lon_step) * cfg.lon_step
        match = self.grid_df[
            (self.grid_df["cell_lat"] == cell_lat)
            & (self.grid_df["cell_lon"] == cell_lon)
        ]
        if match.empty:
            return None
        return match.iloc[0]["risk_label"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _label(self, count: float) -> str:
        if count <= self._cut_low:
            return RISK_LOW
        if count <= self._cut_high:
            return RISK_MED
        return RISK_HIGH

    @staticmethod
    def _aggregate_cells(df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-cell aggregate features."""
        group_keys = ["cell_lat", "cell_lon"]
        agg = (
            df.groupby(group_keys)
            .agg(
                incident_count=("latitude", "count"),
                crime_diversity=("primary_type", "nunique") if "primary_type" in df.columns else ("latitude", "count"),
                night_fraction=("is_night", "mean") if "is_night" in df.columns else ("latitude", lambda x: 0.0),
                arrest_fraction=("arrest", "mean") if "arrest" in df.columns else ("latitude", lambda x: 0.0),
                domestic_fraction=("domestic", "mean") if "domestic" in df.columns else ("latitude", lambda x: 0.0),
            )
            .reset_index()
        )
        return agg
