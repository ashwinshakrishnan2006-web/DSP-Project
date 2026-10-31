"""
CrimeSense – Risk Classifier
=============================
Trains and evaluates Random Forest and XGBoost classifiers to predict
grid-cell risk level (Low / Medium / High).

Design (leakage-controlled, as per paper §III.D)
-------------------------------------------------
- Feature set: crime_diversity, night_fraction, arrest_fraction,
  domestic_fraction.  The incident_count is deliberately excluded because
  it directly determines the risk label (tercile rule).
- Stratified 80/20 train-test split.
- Hyper-parameter tuning via 5-fold cross-validation (optional).
- Evaluation: accuracy, macro-F1, per-class precision/recall.

Results reported in paper
-------------------------
- Random Forest : accuracy=0.732, macro-F1=0.731
- XGBoost       : accuracy≈0.74, macro-F1≈0.74
"""

import logging
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed. XGBClassifier will not be available.")


LABEL_ORDER = ["Low", "Medium", "High"]


class CrimeRiskClassifier:
    """
    Trains Random Forest and (optionally) XGBoost on grid-cell risk labels.

    Parameters
    ----------
    test_size : float   Fraction of data held out for evaluation.
    random_state : int  Reproducibility seed.
    models_dir : str or Path, optional
        Directory to save trained models.
    """

    def __init__(
        self,
        test_size: float = 0.20,
        random_state: int = 42,
        models_dir: Optional[str | Path] = None,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.models_dir = Path(models_dir) if models_dir else None

        self._le = LabelEncoder()
        self._le.classes_ = np.array(LABEL_ORDER)

        self.rf_model: Optional[RandomForestClassifier] = None
        self.xgb_model = None

        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        self.results: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> "CrimeRiskClassifier":
        """Stratified train/test split."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state,
        )
        logger.info(
            "Train: %d | Test: %d", len(self.X_train), len(self.X_test)
        )
        return self

    def train_random_forest(
        self,
        n_estimators: int = 200,
        max_depth: Optional[int] = None,
        min_samples_leaf: int = 2,
    ) -> "CrimeRiskClassifier":
        """Train the Random Forest classifier."""
        logger.info("Training Random Forest (n_estimators=%d)…", n_estimators)
        self.rf_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.rf_model.fit(self.X_train, self.y_train)
        self.results["random_forest"] = self._evaluate(self.rf_model, "Random Forest")
        return self

    def train_xgboost(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ) -> "CrimeRiskClassifier":
        """Train the XGBoost classifier."""
        if not _XGBOOST_AVAILABLE:
            logger.error("XGBoost not installed. Skipping.")
            return self

        logger.info("Training XGBoost (n_estimators=%d)…", n_estimators)
        y_train_enc = self._le.transform(self.y_train)

        self.xgb_model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self.xgb_model.fit(self.X_train, y_train_enc)
        self.results["xgboost"] = self._evaluate_xgb("XGBoost")
        return self

    def cross_validate(self, model_name: str = "random_forest", cv: int = 5) -> np.ndarray:
        """
        5-fold stratified cross-validation on the full dataset.

        Parameters
        ----------
        model_name : str   'random_forest' or 'xgboost'
        cv : int           Number of folds.

        Returns
        -------
        np.ndarray   Per-fold accuracy scores.
        """
        X_all = pd.concat([self.X_train, self.X_test])
        y_all = pd.concat([self.y_train, self.y_test])

        if model_name == "random_forest":
            model = self.rf_model
        elif model_name == "xgboost" and _XGBOOST_AVAILABLE:
            model = self.xgb_model
        else:
            raise ValueError(f"Unknown model: {model_name}")

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, X_all, y_all, cv=skf, scoring="accuracy", n_jobs=-1)
        logger.info("CV accuracy: %.3f ± %.3f", scores.mean(), scores.std())
        return scores

    def feature_importance(self, model_name: str = "random_forest") -> pd.Series:
        """Return feature importances as a sorted pandas Series."""
        if model_name == "random_forest" and self.rf_model:
            importances = self.rf_model.feature_importances_
            return pd.Series(importances, index=self.X_train.columns).sort_values(ascending=False)
        if model_name == "xgboost" and self.xgb_model:
            importances = self.xgb_model.feature_importances_
            return pd.Series(importances, index=self.X_train.columns).sort_values(ascending=False)
        raise ValueError(f"Model '{model_name}' not trained.")

    def predict(self, X: pd.DataFrame, model_name: str = "random_forest") -> np.ndarray:
        """Predict risk labels for new grid cells."""
        if model_name == "random_forest":
            return self.rf_model.predict(X)
        if model_name == "xgboost":
            encoded = self.xgb_model.predict(X)
            return self._le.inverse_transform(encoded)
        raise ValueError(f"Unknown model: {model_name}")

    def save_models(self, directory: Optional[str | Path] = None) -> None:
        """Persist trained models with joblib."""
        out_dir = Path(directory or self.models_dir or "models")
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.rf_model:
            rf_path = out_dir / "random_forest.joblib"
            joblib.dump(self.rf_model, rf_path)
            logger.info("Saved RF model → %s", rf_path)
        if self.xgb_model and _XGBOOST_AVAILABLE:
            xgb_path = out_dir / "xgboost.json"
            self.xgb_model.save_model(str(xgb_path))
            logger.info("Saved XGB model → %s", xgb_path)

    def load_models(self, directory: str | Path = "models") -> "CrimeRiskClassifier":
        """Load persisted models."""
        out_dir = Path(directory)
        rf_path = out_dir / "random_forest.joblib"
        if rf_path.exists():
            self.rf_model = joblib.load(rf_path)
            logger.info("Loaded RF model from %s", rf_path)
        if _XGBOOST_AVAILABLE:
            xgb_path = out_dir / "xgboost.json"
            if xgb_path.exists():
                self.xgb_model = XGBClassifier()
                self.xgb_model.load_model(str(xgb_path))
                logger.info("Loaded XGB model from %s", xgb_path)
        return self

    def print_results(self) -> None:
        """Pretty-print evaluation results."""
        for name, res in self.results.items():
            print(f"\n{'='*60}")
            print(f"  {name.upper()}")
            print(f"{'='*60}")
            print(f"  Accuracy : {res['accuracy']:.4f}")
            print(f"  Macro-F1 : {res['macro_f1']:.4f}")
            print("\nClassification Report:")
            print(res["report"])

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _evaluate(self, model, name: str) -> dict:
        y_pred = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="macro", zero_division=0)
        present_labels = sorted(set(self.y_test) | set(y_pred), key=LABEL_ORDER.index)
        report = classification_report(self.y_test, y_pred, labels=present_labels,
                                       target_names=present_labels, zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred, labels=LABEL_ORDER)
        logger.info("%s → accuracy=%.4f, macro-F1=%.4f", name, acc, f1)
        return {"accuracy": acc, "macro_f1": f1, "report": report, "confusion_matrix": cm}

    def _evaluate_xgb(self, name: str) -> dict:
        y_pred_enc = self.xgb_model.predict(self.X_test)
        y_pred = self._le.inverse_transform(y_pred_enc)
        acc = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average="macro", zero_division=0)
        present_labels = sorted(set(self.y_test) | set(y_pred), key=LABEL_ORDER.index)
        report = classification_report(self.y_test, y_pred, labels=present_labels,
                                       target_names=present_labels, zero_division=0)
        cm = confusion_matrix(self.y_test, y_pred, labels=LABEL_ORDER)
        logger.info("%s → accuracy=%.4f, macro-F1=%.4f", name, acc, f1)
        return {"accuracy": acc, "macro_f1": f1, "report": report, "confusion_matrix": cm}
