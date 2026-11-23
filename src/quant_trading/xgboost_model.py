"""Walk-forward XGBoost binary probability forecasting, without lookahead."""
from __future__ import annotations

from collections.abc import Mapping
from itertools import product

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from xgboost import XGBClassifier

from .ml import probability_metrics, xy

# Exactly eight configurations, fixed BEFORE inspecting validation performance.
DEFAULT_PARAM_GRID = {
    "max_depth": (2, 3),
    "min_child_weight": (5, 15),
    "n_estimators": (100, 300),
}


def make_xgb(**tuned_params: int) -> XGBClassifier:
    """Conservative XGB binary probability classifier; no feature scaling.

    All non-tuned hyperparameters remain identical across CV candidates.
    """
    defaults = {"max_depth": 2, "min_child_weight": 5, "n_estimators": 100}
    unknown = set(tuned_params) - set(defaults)
    if unknown:
        raise ValueError(f"Unexpected tuned parameters: {sorted(unknown)}")
    defaults.update(tuned_params)
    if any(int(v) <= 0 for v in defaults.values()):
        raise ValueError("Depth, min_child_weight and n_estimators must be positive")
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=1.0,
        reg_lambda=5.0,
        reg_alpha=0.0,
        gamma=0.0,
        tree_method="hist",
        random_state=42,
        n_jobs=4,
        **defaults,
    )


def tune_xgb(
    train: pd.DataFrame,
    *,
    param_grid: Mapping[str, tuple[int, ...] | list[int]] = DEFAULT_PARAM_GRID,
    n_splits: int = 4,
    test_size: int = 252,
    gap: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Tune only with train-era expanding-window CV and purged mature labels.

    Returns (fold_scores, summary, best_params).  Validation and held-out test
    are never read here. All candidates use exactly the same folds and labels.
    """
    if set(param_grid) != set(DEFAULT_PARAM_GRID):
        raise ValueError(f"Expected grid keys {sorted(DEFAULT_PARAM_GRID)}")
    grid = list(ParameterGrid(dict(param_grid)))
    if not grid:
        raise ValueError("Empty hyperparameter grid")
    X, y = xy(train)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    results: list[dict] = []

    for fold, (fit_idx, val_idx) in enumerate(splitter.split(X), start=1):
        fit_frame, val_frame = train.iloc[fit_idx], train.iloc[val_idx]
        if not (fit_frame["exit_date"].max() < val_frame.index.min()):
            raise ValueError(f"Fold {fold}: immature training labels at validation boundary")
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        if y_fit.nunique() < 2:
            raise ValueError(f"Fold {fold}: training data contains only one class")

        for params in grid:
            model = make_xgb(**params)
            model.fit(X_fit, y_fit)
            p = model.predict_proba(X_val)[:, 1]
            results.append({
                "fold": fold,
                **params,
                "train_start": fit_frame.index[0],
                "train_end": fit_frame.index[-1],
                "validation_start": val_frame.index[0],
                "validation_end": val_frame.index[-1],
                "validation_n": len(val_idx),
                **probability_metrics(y_val, p),
            })

    per_fold = pd.DataFrame(results)
    summary = (
        per_fold.groupby(list(DEFAULT_PARAM_GRID), as_index=False)
        .agg(
            cv_log_loss=("log_loss", "mean"),
            cv_log_loss_std=("log_loss", "std"),
            cv_brier_score=("brier_score", "mean"),
            cv_roc_auc=("roc_auc", "mean"),
        )
    )
    # Parsimonious deterministic tie-break; mean CV log loss always primary.
    best = summary.sort_values(
        ["cv_log_loss", "max_depth", "n_estimators", "min_child_weight"],
        ascending=[True, True, True, False],
    ).iloc[0]
    params = {key: int(best[key]) for key in DEFAULT_PARAM_GRID}
    return per_fold, summary, params


def fit_and_evaluate_xgb(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    best_params: Mapping[str, int],
) -> tuple[XGBClassifier, pd.DataFrame, pd.DataFrame]:
    """Refit selected model on all purged training rows, evaluate on validation."""
    X_train, y_train = xy(train)
    X_val, y_val = xy(validation)
    if train["exit_date"].max() >= validation.index.min():
        raise ValueError("Training labels not mature before validation start")
    if X_train.columns.tolist() != X_val.columns.tolist():
        raise ValueError("Train/validation feature mismatch")
    if y_train.nunique() < 2:
        raise ValueError("Training labels require both classes")
    model = make_xgb(**dict(best_params))
    model.fit(X_train, y_train)
    p_train = model.predict_proba(X_train)[:, 1]
    p_val = model.predict_proba(X_val)[:, 1]
    predictions = pd.DataFrame({"actual_label": y_val, "p_xgboost": p_val}, index=validation.index)
    train_metrics = {"split": "train", "model": "XGBoost", **probability_metrics(y_train, p_train)}
    validation_metrics = {"split": "validation", "model": "XGBoost", **probability_metrics(y_val, p_val)}
    return model, predictions, pd.DataFrame([train_metrics, validation_metrics])


def feature_gain(model: XGBClassifier, feature_names: list[str]) -> pd.DataFrame:
    """XGBoost gain importance; descriptive in-sample measure, not causality."""
    values = pd.Series(model.get_booster().get_score(importance_type="gain"), dtype=float)
    # When pandas features are passed to .fit, XGBoost names them explicitly.
    return (values.reindex(feature_names, fill_value=0.0)
            .rename_axis("feature").reset_index(name="gain")
            .sort_values("gain", ascending=False).reset_index(drop=True))
