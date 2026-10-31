# Additional CrimeSense modules

Drop these files into the repo root (or your Python package folder) as separate commits.
Requires: `pandas`, `numpy`, `scikit-learn`. Run tests with `python -m unittest -v`.

| File | Purpose | Paper section |
|---|---|---|
| `temporal_analysis.py` | Hour / weekday / month counts, night flag, peak period | IV-B |
| `grid_risk.py` | ~500 m grid, per-cell features, tercile risk labels, leakage-controlled Random Forest | III-E, IV-D |
| `alert_engine.py` | Haversine distance, `check_alert` (Eq. 1), alert coverage vs radius | III-F, IV-E |
| `crime_statistics.py` | Top crime types, arrest rates, dataset summary | IV-A/B |
| `report_validator.py` | Coordinate, date and keyword (whole-word) checks before a report enters the admin queue | III-C, III-G |
| `test_crimesense_modules.py` | Unit tests for all of the above (synthetic data) | - |

## Expected columns
`date, latitude, longitude, primary_type, arrest` (names configurable via function arguments).

## Quick example
```python
import temporal_analysis as ta, grid_risk as gr, alert_engine as ae
df = ta.add_temporal_features(raw_df)
df, dlat, dlon = gr.assign_grid(df, cell_km=0.5)
cells = gr.build_cell_features(df, dlat, dlon)
cells["risk"] = gr.label_risk_terciles(cells["incident_count"])
model, metrics = gr.train_risk_classifier(cells)   # excludes incident_count
high = cells[cells.risk == "High"][["centroid_lat", "centroid_lon"]].values
print(ae.check_alert((41.88, -87.63), high, radius_km=1.0))
```

## Notes
- `train_risk_classifier` raises if `incident_count` is passed as a feature (it defines the labels).
- Grid centroids use a simple lat/lon grid; adequate for ~500 m cells at city scale.
- The 0.73 accuracy in the paper measures recovery of a density-derived label, not future-crime forecasting.
