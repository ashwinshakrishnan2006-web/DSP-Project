# CrimeSense

A data-driven crime analysis and community safety dashboard. It supports CSV upload, cleaning, exploratory analysis, DBSCAN/K-Means hotspot discovery, optional Random Forest/XGBoost risk classification, map visualisation, proximity alerts, and a locally stored report-review workflow.

> **Important:** CrimeSense is an educational decision-support prototype, not a crime-prediction authority or emergency service. Model outputs describe patterns in the uploaded historical dataset; they do not establish that a person or location will commit or experience a crime. Do not use the output for policing, profiling, or decisions about individuals. Validate data quality, bias, and model performance before any real-world use.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload a CSV containing latitude and longitude columns. CrimeSense attempts to recognise common names such as `latitude`/`longitude`, `lat`/`lon`, `x`/`y`, and `Primary Type`/`crime_type`/`category`.

## Data format

Minimum columns: latitude and longitude (decimal degrees). Recommended columns:

| Field | Purpose |
|---|---|
| `latitude`, `longitude` | Coordinates (required) |
| `date` | Event date/time |
| `crime_type` | Crime category / label |
| `description` | Optional description |
| `reported_at` | Report submission timestamp |

The column-mapping controls in the sidebar allow different source column names. Remove personal identifiers before upload. Do not upload victim names, addresses, phone numbers, or other sensitive personal data.

## Features

1. **Data preparation:** type conversion, coordinate validation, missing-value handling, duplicate removal, and date parsing.
2. **EDA:** category counts, monthly trend, and geographic distribution.
3. **Hotspots:** DBSCAN using haversine distance and K-Means clustering. DBSCAN noise is shown separately; cluster labels are not severity scores.
4. **Risk model:** Random Forest and (if installed) XGBoost classify higher/lower event-density grid cells using a spatially grouped holdout where possible. Results are descriptive and should be treated cautiously. The model is disabled when the data cannot support a valid evaluation.
5. **Map and alerts:** map layers and a configurable radius search around a location entered by the user. Alerts are proximity indicators only.
6. **Community reports:** local SQLite storage for submitted reports and a simple moderation view. This is a classroom demo, not a secure production reporting channel.

## Model target and evaluation

The model builds a grid-based target from event counts. Cells at or above the training-set median count are labelled `higher_density`; other cells are `lower_density`. This target is a relative density label, not a forecast of individual crimes. A stratified random holdout is used only when the sample size permits; otherwise the app reports that model evaluation is unavailable. The UI displays the split and metrics and avoids claiming results from an absent dataset.

## Database and moderation

Report submissions are stored in `crimesense_reports.sqlite3` on the machine running Streamlit. The moderation page uses a demo passphrase from `ADMIN_PASSWORD` environment variable (default: `change-me`). Change it before any shared demonstration. This local workflow is not suitable for public deployment without authentication, access controls, abuse prevention, encryption, and a privacy policy.

## Repository structure

```text
CrimeSense/
├── app.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── analysis.py
│   ├── modeling.py
│   └── reports.py
├── tests/test_core.py
├── data/                 # keep private datasets out of Git
├── requirements.txt
└── README.md
```

## Limitations

- Results depend on the quality, coverage, geocoding accuracy, and reporting practices of the uploaded data.
- Spatial clusters are sensitive to DBSCAN radius/minimum-points settings and coordinate precision.
- Grid-density labels are not causal explanations and are not individual-level predictions.
- A map pin or proximity alert is not a verified threat. Use local emergency services for urgent situations.
- The demo moderation store is local and is not shared across machines.

## Attribution and academic use

Record the actual contributors and cite/attribute any reused datasets, code, or prior project material according to the course rules. Do not present borrowed work as independently authored. Add your own documented changes and verify that your instructor permits reuse.
