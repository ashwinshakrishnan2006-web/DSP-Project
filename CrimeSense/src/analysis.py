from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.neighbors import BallTree

EARTH_RADIUS_KM = 6371.0088

def dbscan_hotspots(df: pd.DataFrame, radius_m: int = 500, min_samples: int = 8) -> pd.DataFrame:
    out = df.copy()
    if len(out) < min_samples:
        out["dbscan_cluster"] = -1
        return out
    coords = np.radians(out[["latitude", "longitude"]].to_numpy())
    labels = DBSCAN(eps=radius_m / 1000 / EARTH_RADIUS_KM, min_samples=min_samples,
                    metric="haversine", algorithm="ball_tree").fit_predict(coords)
    out["dbscan_cluster"] = labels
    return out

def kmeans_hotspots(df: pd.DataFrame, clusters: int = 5, random_state: int = 42) -> pd.DataFrame:
    out = df.copy()
    k = min(max(1, int(clusters)), len(out))
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    out["kmeans_cluster"] = model.fit_predict(out[["latitude", "longitude"]])
    return out

def nearby_events(df: pd.DataFrame, latitude: float, longitude: float, radius_km: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    coords = np.radians(df[["latitude", "longitude"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    query = np.radians([[latitude, longitude]])
    indices = tree.query_radius(query, r=max(0, radius_km) / EARTH_RADIUS_KM)[0]
    result = df.iloc[indices].copy()
    if len(result):
        result["distance_km"] = EARTH_RADIUS_KM * np.arccos(np.clip(
            np.sin(np.radians(latitude)) * np.sin(np.radians(result.latitude)) +
            np.cos(np.radians(latitude)) * np.cos(np.radians(result.latitude)) *
            np.cos(np.radians(result.longitude) - np.radians(longitude)), -1, 1))
        result = result.sort_values("distance_km")
    return result

def grid_density(df: pd.DataFrame, cell_size: float = 0.01) -> pd.DataFrame:
    """Aggregate events into approximate degree-sized cells for descriptive modelling."""
    if df.empty:
        return pd.DataFrame(columns=["grid_lat", "grid_lon", "event_count"])
    work = df.copy()
    work["grid_lat"] = np.floor(work.latitude / cell_size) * cell_size
    work["grid_lon"] = np.floor(work.longitude / cell_size) * cell_size
    return work.groupby(["grid_lat", "grid_lon"], as_index=False).size().rename(columns={"size": "event_count"})
