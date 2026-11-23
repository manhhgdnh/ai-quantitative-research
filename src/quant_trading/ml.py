"""Leakage-aware daily SPY probability forecasting: baseline and L2 logistic regression."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


DEFAULT_C_GRID = (0.01, 0.1, 1.0)


def make_baseline() -> DummyClassifier:
    """Predict the training fold's historical fraction of positive labels."""
    return DummyClassifier(strategy="prior")


def make_logistic(*, C: float = 0.1) -> Pipeline:
    """Fit scaler inside each training fold; never on future observations."""
    if C <= 0:
        raise ValueError("C must be positive")
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=C, solver="lbfgs", max_iter=3000,
            random_state=42,
        )),
    ])


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return only declared model features, never target-construction metadata."""
    required = set(FEATURE_COLUMNS) | {"label", "exit_date"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("Expected sorted, unique signal timestamps")
    X = frame.loc[:, list(FEATURE_COLUMNS)]
    y = frame["label"].astype(int)
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Non-finite model features")
    if not y.isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    return X, y


def probability_metrics(y: Sequence[int], p: Sequence[float]) -> dict[str, float]:
    """Evaluate probabilistic classification without imposing a trading threshold."""
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.asarray(p, dtype=float)
    if y_arr.shape != p_arr.shape or not np.isfinite(p_arr).all():
        raise ValueError("Invalid probability vector")
    if np.any((p_arr < 0) | (p_arr > 1)):
        raise ValueError("Probabilities must lie in [0, 1]")
    out = {
        "log_loss": float(log_loss(y_arr, p_arr, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y_arr, p_arr)),
        "roc_auc": float(roc_auc_score(y_arr, p_arr)) if len(np.unique(y_arr)) == 2 else float("nan"),
        "observed_positive_rate": float(y_arr.mean()),
        "mean_predicted_probability": float(p_arr.mean()),
    }
    return out


def cross_validate_models(
    train: pd.DataFrame,
    *,
    c_grid: Sequence[float] = DEFAULT_C_GRID,
    n_splits: int = 4,
    test_size: int = 252,
    gap: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """CV only on training era; ensure every train label matures before validation.

    Returns per-fold scores, candidate summary, and best Logistic C selected by
    mean CV log loss. The DummyClassifier serves as a comparison, not a tuned model.
    """
    if not c_grid or any(c <= 0 for c in c_grid):
        raise ValueError("c_grid must contain positive numbers")
    X, y = xy(train)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    results: list[dict] = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        fit_frame = train.iloc[train_idx]
        val_frame = train.iloc[val_idx]
        if not (fit_frame["exit_date"].max() < val_frame.index.min()):
            raise ValueError(f"Fold {fold}: immature training labels at validation boundary")
        X_fit, y_fit = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        if y_fit.nunique() < 2:
            raise ValueError(f"Fold {fold}: training data contains only one class")
        candidates = [("Constant Probability", float("nan"), make_baseline())]
        candidates.extend(("Logistic Regression", float(c), make_logistic(C=float(c))) for c in c_grid)
        for name, c, model in candidates:
            model.fit(X_fit, y_fit)
            p = model.predict_proba(X_val)[:, 1]
            results.append({
                "fold": fold,
                "model": name,
                "C": c,
                "train_end": fit_frame.index[-1],
                "validation_start": val_frame.index[0],
                "validation_end": val_frame.index[-1],
                **probability_metrics(y_val, p),
            })
    per_fold = pd.DataFrame(results)
    summary = (
        per_fold.groupby(["model", "C"], dropna=False)
        .agg(
            cv_log_loss=("log_loss", "mean"),
            cv_log_loss_std=("log_loss", "std"),
            cv_brier_score=("brier_score", "mean"),
            cv_roc_auc=("roc_auc", "mean"),
        )
        .reset_index()
    )
    logistic_rows = summary.loc[summary["model"] == "Logistic Regression"]
    if logistic_rows.empty:
        raise ValueError("No Logistic Regression candidates")
    # Tie-break with stronger regularization (smaller C).
    best = logistic_rows.sort_values(["cv_log_loss", "C"]).iloc[0]
    return per_fold, summary, float(best["C"])


def fit_and_evaluate(
    train: pd.DataFrame, validation: pd.DataFrame, *, best_C: float
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Train once on 2005-2018; evaluate once on 2019-2021, not the held-out test."""
    if train["exit_date"].max() >= validation.index.min():
        raise ValueError("Training label not mature before validation start")
    X_train, y_train = xy(train)
    X_val, y_val = xy(validation)
    models = {"Constant Probability": make_baseline(), "Logistic Regression": make_logistic(C=best_C)}
    outputs = pd.DataFrame({"actual_label": y_val}, index=validation.index)
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        outputs["p_baseline" if name == "Constant Probability" else "p_logistic"] = proba
        rows.append({"model": name, **probability_metrics(y_val, proba)})
    scores = pd.DataFrame(rows).sort_values("log_loss")
    return models, outputs, scores
