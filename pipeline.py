"""
CrimeSense – End-to-End Pipeline Runner
=========================================
Runs the complete data science pipeline:

  1. Load & clean data
  2. DBSCAN hotspot detection
  3. K-Means clustering (comparison)
  4. Grid-based risk labelling (tercile rule)
  5. Random Forest + XGBoost classification
  6. Proximity alert coverage analysis
  7. Interactive map generation

Usage
-----
  python pipeline.py                          # uses sample data
  python pipeline.py --data data/chicago_crimes_2023.csv
  python pipeline.py --data data/chicago_crimes_2023.csv --no-map
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.cleaner import load_and_clean
from src.clustering.dbscan_hotspot import detect_hotspots, DBSCANConfig
from src.clustering.kmeans_hotspot import kmeans_hotspots
from src.classification.risk_labeler import RiskLabeler
from src.classification.classifier import CrimeRiskClassifier
from src.alerting.proximity_alert import ProximityAlertEngine, simulate_coverage
from src.visualization.map_renderer import render_map

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("crimesense.pipeline")
# ---------------------------------------------------------------------------

SAMPLE_DATA_PATH = Path("data/sample_chicago_crimes.csv")
FULL_DATA_PATH = Path("data/chicago_crimes_2023.csv")


def run_pipeline(data_path: Path, render_html_map: bool = True) -> None:
    logger.info("=" * 60)
    logger.info("CrimeSense Pipeline – START")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1: Load & Clean
    # ------------------------------------------------------------------
    logger.info("[1/6] Loading and cleaning data: %s", data_path)
    df = load_and_clean(data_path)
    logger.info("Clean records: %d", len(df))

    # ------------------------------------------------------------------
    # Stage 2: DBSCAN Hotspot Detection
    # ------------------------------------------------------------------
    logger.info("[2/6] DBSCAN hotspot detection…")
    dbscan_cfg = DBSCANConfig(eps_km=0.3, min_samples=max(5, len(df) // 5000))
    dbscan_result = detect_hotspots(df, eps_km=dbscan_cfg.eps_km, min_samples=dbscan_cfg.min_samples)
    logger.info("Hotspots found: %d | Noise: %.1f%%",
                dbscan_result.n_clusters, dbscan_result.noise_fraction * 100)
    print(f"\n📍 DBSCAN: {dbscan_result.n_clusters} hotspot clusters detected")
    if not dbscan_result.hotspots.empty:
        print(dbscan_result.hotspots.head(5).to_string(index=False))

    # ------------------------------------------------------------------
    # Stage 3: K-Means (comparison)
    # ------------------------------------------------------------------
    logger.info("[3/6] K-Means clustering (k=%d)…", min(dbscan_result.n_clusters or 10, 32))
    k = max(2, dbscan_result.n_clusters or 10)
    km_result = kmeans_hotspots(df, k=k)
    logger.info("K-Means inertia: %.2f", km_result.inertia)
    print(f"\n📍 K-Means: {k} clusters | Inertia: {km_result.inertia:.2f}")

    # ------------------------------------------------------------------
    # Stage 4: Risk Labelling
    # ------------------------------------------------------------------
    logger.info("[4/6] Grid-based risk labelling…")
    labeler = RiskLabeler()
    labeler.fit(df)
    grid_df = labeler.grid_df
    print(f"\n🗺️  Risk Grid: {len(grid_df)} cells")
    print(grid_df["risk_label"].value_counts().to_string())

    # ------------------------------------------------------------------
    # Stage 5: Risk Classification (RF + XGB)
    # ------------------------------------------------------------------
    logger.info("[5/6] Training classifiers…")
    X, y = labeler.get_features_and_labels(include_incident_count=False)

    if len(X) < 10:
        logger.warning("Too few grid cells (%d) for meaningful classification. Skipping.", len(X))
    else:
        clf = CrimeRiskClassifier(models_dir="models")
        clf.prepare_data(X, y)
        clf.train_random_forest()
        try:
            clf.train_xgboost()
        except Exception as exc:
            logger.warning("XGBoost training failed: %s", exc)
        clf.print_results()
        clf.save_models()

    # ------------------------------------------------------------------
    # Stage 6: Proximity Alert Coverage
    # ------------------------------------------------------------------
    logger.info("[6/6] Simulating proximity alert coverage…")
    engine = ProximityAlertEngine(grid_df, radius_km=1.0)
    coverage_df = simulate_coverage(grid_df, n_simulated=5000)
    print("\n🔔 Proximity Alert Coverage:")
    print(coverage_df.to_string(index=False))

    # Example single-point check (Millennium Park)
    test_result = engine.check(41.8826, -87.6233)
    status = "⚠️  ALERT" if test_result.alert_triggered else "✅  SAFE"
    print(f"\nMillennium Park check: {status} "
          f"(nearest High-risk cell: {test_result.nearest_distance_km:.2f} km)")

    # ------------------------------------------------------------------
    # Stage 7 (optional): Interactive Map
    # ------------------------------------------------------------------
    if render_html_map:
        logger.info("[7/7] Rendering interactive map…")
        try:
            map_path = render_map(grid_df, dbscan_result.hotspots)
            print(f"\n🗺️  Map saved → {map_path}")
        except ImportError as exc:
            logger.warning("Map rendering skipped: %s", exc)

    logger.info("=" * 60)
    logger.info("CrimeSense Pipeline – COMPLETE")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CrimeSense end-to-end pipeline runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to the Chicago crime CSV. Uses sample data if omitted.",
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip interactive map generation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.data:
        data_path = args.data
    elif FULL_DATA_PATH.exists():
        data_path = FULL_DATA_PATH
    elif SAMPLE_DATA_PATH.exists():
        data_path = SAMPLE_DATA_PATH
    else:
        logger.error(
            "No data file found. Place 'chicago_crimes_2023.csv' in the data/ directory, "
            "or run:  python data/generate_sample.py"
        )
        sys.exit(1)

    run_pipeline(data_path, render_html_map=not args.no_map)
