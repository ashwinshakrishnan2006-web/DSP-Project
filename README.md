# CrimeSense 🔍

> **A Data-Driven Framework for Crime Hotspot Analysis and Proximity-Based Safety Alerting**

**CSI3004 – Data Science Programming | VIT Vellore**
Team: Ashwinsha P (24MID0226) · Mahithaa R (24MID0239)
Faculty In-charge: Dr. M. A. Berlin – SCOPE

---

## Overview

CrimeSense transforms historical municipal crime records into actionable, location-aware safety information through an integrated pipeline:

| Stage | Technique |
|---|---|
| Data Preprocessing | Deduplication, geocoding, temporal feature extraction |
| Hotspot Detection | DBSCAN (primary) + K-Means (comparison) |
| Risk Classification | Random Forest + XGBoost (Low / Medium / High) |
| Visualization | Interactive Folium web map |
| Proximity Alerting | Haversine distance to High-risk grid cells |
| User Reporting | Admin-verified crowd-sourced incident submission |

Experiments on **261,238 geocoded Chicago crime records (2023)** show:
- DBSCAN isolates **32 localized hotspots** with 2.5% noise  
- Leakage-controlled Random Forest attains **0.732 accuracy / 0.731 macro-F1**  
- 1 km proximity alert covers **30.1%** of simulated user locations

---

## Repository Structure

```
CrimeSense/
├── data/                          # Raw & processed datasets (gitignored for large files)
│   ├── sample_chicago_crimes.csv  # Small sample for testing
│   └── README.md
├── notebooks/                     # Exploratory Jupyter notebooks
│   ├── 01_eda.ipynb
│   └── 02_clustering_experiments.ipynb
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── cleaner.py             # Data cleaning & feature engineering
│   ├── clustering/
│   │   ├── __init__.py
│   │   ├── dbscan_hotspot.py      # DBSCAN hotspot detection
│   │   └── kmeans_hotspot.py      # K-Means comparison
│   ├── classification/
│   │   ├── __init__.py
│   │   ├── risk_labeler.py        # Grid-based tercile risk labeling
│   │   └── classifier.py          # Random Forest + XGBoost training & eval
│   ├── alerting/
│   │   ├── __init__.py
│   │   └── proximity_alert.py     # Haversine-based proximity alerting
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── map_renderer.py        # Folium interactive map generation
│   └── utils/
│       ├── __init__.py
│       └── haversine.py           # Haversine distance utility
├── tests/
│   ├── test_preprocessing.py
│   ├── test_clustering.py
│   ├── test_classification.py
│   └── test_alerting.py
├── pipeline.py                    # End-to-end pipeline runner
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the full pipeline (uses sample data if Chicago CSV not present)
python pipeline.py

# 4. Open the generated map
start output/crime_map.html    # Windows
```

---

## Data Source

Chicago Open Data Portal – **Crimes 2023**  
<https://data.cityofchicago.org/Public-Safety/Crimes-2023/xguy-4ndq>

Download the CSV and place it at `data/chicago_crimes_2023.csv`.  
A 500-row sample (`data/sample_chicago_crimes.csv`) is included for testing.

---

## Pipeline Stages (Part 1 – Data Science Core)

### 1. Preprocessing (`src/preprocessing/cleaner.py`)
- Drops rows with missing latitude/longitude  
- Deduplicates on `Case Number`  
- Normalises `Primary Type` labels  
- Derives `hour`, `month`, `is_night` temporal features  

### 2. Hotspot Detection
- **DBSCAN** (`src/clustering/dbscan_hotspot.py`): eps=0.3 km, min_samples=50 → 32 clusters, 2.5% noise  
- **K-Means** (`src/clustering/kmeans_hotspot.py`): k=32 for direct comparison  

### 3. Risk Classification (`src/classification/`)
- 500 m grid cells, features: `crime_diversity`, `night_fraction`, `arrest_fraction`  
- Labels: tercile cut-points of per-cell incident count → Low / Medium / High  
- Leakage-controlled: incident count excluded from feature set  
- Models: `RandomForestClassifier` and `XGBClassifier`  

### 4. Proximity Alerting (`src/alerting/proximity_alert.py`)
- Haversine distance from user location to every High-risk cell centroid  
- Alert if `min_distance ≤ radius` (default 1 km)  
- Coverage characterised at 0.5 km, 1 km, 2 km radii  

### 5. Interactive Map (`src/visualization/map_renderer.py`)
- Folium choropleth of risk levels (green / orange / red)  
- DBSCAN hotspot circles overlaid  
- Saved to `output/crime_map.html`  

---

## Part 2 – Web Application *(coming soon)*

The Flask/Django web app (citizen portal, admin portal, SOS system) will be added in Part 2.

---

## References

1. Pedregosa et al. – Scikit-learn (JMLR 2011)  
2. Chen & Guestrin – XGBoost (KDD 2016)  
3. Mohammed & Baiee – DBSCAN on Baltimore crimes  
4. Baqir et al. – HDBSCAN spatio-temporal hotspot detection  
