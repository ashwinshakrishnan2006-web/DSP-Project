"""Grid-based risk labelling and classification (Low / Medium / High)."""
from __future__ import annotations
import numpy as np
import pandas as pd

KM_PER_DEG_LAT = 111.32
LABELS = ["Low", "Medium", "High"]


def assign_grid(df, cell_km: float = 0.5, lat_col="latitude", lon_col="longitude"):
    """Assign each record to a square grid cell of roughly cell_km size."""
    out = df.copy()
    lat0 = out[lat_col].mean()
    dlat = cell_km / KM_PER_DEG_LAT
    dlon = cell_km / (KM_PER_DEG_LAT * np.cos(np.radians(lat0)))
    out["cell_row"] = np.floor(out[lat_col] / dlat).astype(int)
    out["cell_col"] = np.floor(out[lon_col] / dlon).astype(int)
    out["cell_id"] = out["cell_row"].astype(str) + "_" + out["cell_col"].astype(str)
    return out, dlat, dlon


def build_cell_features(df, dlat, dlon, type_col="primary_type",
                        arrest_col="arrest", night_col="is_night",
                        lat_col="latitude", lon_col="longitude"):
    """Aggregate per-cell features; centroid is the cell's centre."""
    g = df.groupby("cell_id")
    feats = g.agg(
        incident_count=("cell_id", "size"),
        distinct_types=(type_col, "nunique"),
        night_fraction=(night_col, "mean"),
        arrest_fraction=(arrest_col, lambda s: s.astype(float).mean()),
        row=("cell_row", "first"),
        col=("cell_col", "first"),
    )
    feats["centroid_lat"] = (feats["row"] + 0.5) * dlat
    feats["centroid_lon"] = (feats["col"] + 0.5) * dlon
    return feats.drop(columns=["row", "col"])


def label_risk_terciles(counts: pd.Series) -> pd.Series:
    """Low/Medium/High from terciles of per-cell incident counts."""
    q1, q2 = counts.quantile([1 / 3, 2 / 3])
    def f(c):
        return "Low" if c <= q1 else ("Medium" if c <= q2 else "High")
    return counts.map(f)


LEAKAGE_SAFE_FEATURES = ["distinct_types", "night_fraction", "arrest_fraction"]


def train_risk_classifier(cells: pd.DataFrame, features=None, seed: int = 42):
    """Train a Random Forest on leakage-controlled features (no incident_count).

    Returns (model, metrics dict). Raises if incident_count is requested.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split

    features = features or LEAKAGE_SAFE_FEATURES
    if "incident_count" in features:
        raise ValueError("incident_count defines the labels; using it leaks the target.")
    X, y = cells[features], cells["risk"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    model = RandomForestClassifier(n_estimators=200, random_state=seed).fit(Xtr, ytr)
    pred = model.predict(Xte)
    return model, {
        "accuracy": accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro"),
        "test_size": len(yte),
    }
