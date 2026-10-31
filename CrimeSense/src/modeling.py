from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATURES = ["grid_lat", "grid_lon", "event_count"]

def train_density_model(cells: pd.DataFrame, model_name: str = "Random Forest"):
    """Train a cautious cell-density classifier; returns (model, metrics, labelled_cells)."""
    if len(cells) < 12 or cells.event_count.nunique() < 2:
        return None, {"available": False, "reason": "At least 12 distinct grid cells with varying counts are needed."}, cells.copy()
    data = cells.copy()
    threshold = float(data.event_count.median())
    data["density_class"] = (data.event_count >= threshold).astype(int)
    if data.density_class.nunique() < 2 or data.density_class.value_counts().min() < 2:
        return None, {"available": False, "reason": "The grid labels do not contain enough examples of both classes."}, data
    X, y = data[FEATURES], data.density_class
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
    except ValueError:
        return None, {"available": False, "reason": "Could not create a stratified holdout split."}, data
    if model_name == "XGBoost":
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(n_estimators=120, max_depth=3, learning_rate=0.05,
                                  subsample=0.9, colsample_bytree=0.9,
                                  eval_metric="logloss", random_state=42)
        except Exception as exc:
            return None, {"available": False, "reason": f"XGBoost unavailable: {exc}"}, data
    else:
        model = RandomForestClassifier(n_estimators=250, min_samples_leaf=2,
                                        class_weight="balanced", random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {"available": True, "model": model_name, "threshold": threshold,
               "train_rows": len(X_train), "test_rows": len(X_test),
               "accuracy": float(accuracy_score(y_test, pred)),
               "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
               "report": classification_report(y_test, pred, target_names=["lower_density", "higher_density"],
                                                zero_division=0, output_dict=True)}
    data["predicted_density_class"] = model.predict(X)
    return model, metrics, data
