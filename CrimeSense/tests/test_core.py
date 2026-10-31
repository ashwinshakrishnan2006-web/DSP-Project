import pandas as pd
from src.data import clean_crime_data
from src.analysis import dbscan_hotspots, nearby_events, grid_density

def test_cleaning_drops_invalid_and_duplicates():
    raw = pd.DataFrame({"lat": [12.9, 12.9, 999], "lon": [79.1, 79.1, 79.1], "kind": ["Theft", "Theft", "X"]})
    clean, stats = clean_crime_data(raw, "lat", "lon", type_col="kind")
    assert len(clean) == 1
    assert stats["invalid_coordinate_rows"] == 1
    assert stats["duplicate_rows_removed"] == 1

def test_nearby_search_finds_point():
    data = pd.DataFrame({"latitude": [12.9, 15.0], "longitude": [79.1, 80.0]})
    assert len(nearby_events(data, 12.9, 79.1, 1.0)) == 1

def test_dbscan_and_grid_density():
    data = pd.DataFrame({"latitude": [12.9, 12.9001, 15.0], "longitude": [79.1, 79.1001, 80.0]})
    clustered = dbscan_hotspots(data, radius_m=300, min_samples=2)
    assert "dbscan_cluster" in clustered
    assert len(grid_density(data)) >= 2
