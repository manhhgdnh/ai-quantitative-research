"""Paired probabilistic forecast comparison with time-series-aware uncertainty.

All diagnostics are descriptive/development-stage on validation predictions.
Bootstrap CIs assume sufficient weak dependence/stability; they do not adjust
for all earlier feature selection, model family decisions, or hyperparameter search.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .ml import probability_metrics

PROBABILITY_COLUMNS = {
    "Constant Probability": "p_baseline",
    "Logistic Regression": "p_logistic",
    "XGBoost": "p_xgboost",
}


def validate_predictions(pred: pd.DataFrame) -> None:
    required = {"actual_label", *PROBABILITY_COLUMNS.values()}
    if not required.issubset(pred.columns):
        raise ValueError(f"Missing prediction columns: {sorted(required - set(pred.columns))}")
    if not isinstance(pred.index, pd.DatetimeIndex) or not pred.index.is_unique or not pred.index.is_monotonic_increasing:
        raise ValueError("Predictions require sorted unique signal-date index")
    if pred.empty or pred["actual_label"].isna().any() or not pred["actual_label"].isin([0, 1]).all():
        raise ValueError("Invalid validation labels")
    for name in PROBABILITY_COLUMNS.values():
        p = pred[name].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid probabilities in {name}")


def observation_log_loss(y: np.ndarray | pd.Series, p: np.ndarray | pd.Series) -> np.ndarray:
    """Per-observation binary negative log likelihood (no aggregation)."""
    y_array = np.asarray(y, dtype=float)
    p_array = np.asarray(p, dtype=float)
    if y_array.ndim != 1 or p_array.shape != y_array.shape or y_array.size == 0:
        raise ValueError("Require equally sized, nonempty 1D inputs")
    if not np.isfinite(y_array).all() or not np.isin(y_array, [0, 1]).all():
        raise ValueError("Invalid binary labels")
    if not np.isfinite(p_array).all() or ((p_array < 0) | (p_array > 1)).any():
        raise ValueError("Invalid probabilities")
    p_array = np.clip(p_array, 1e-12, 1 - 1e-12)
    return -(y_array * np.log(p_array) + (1 - y_array) * np.log1p(-p_array))


def moving_block_bootstrap(
    loss_differences: np.ndarray | pd.Series,
    *,
    block_length: int = 10,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Moving-block percentile CI for mean of paired loss differences.

    Resample a *single* aligned loss-difference series in chronological blocks.
    The CI is exploratory and conditional on the fixed selected models.
    """
    d = np.asarray(loss_differences, dtype=float)
    if d.ndim != 1 or d.size < 4 or not np.isfinite(d).all():
        raise ValueError("Expected finite 1D loss-difference series of size >= 4")
    n = d.size
    if not isinstance(block_length, int) or not 1 <= block_length <= n // 2:
        raise ValueError("Block length must be integer from 1 to n//2")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 100:
        raise ValueError("At least 100 bootstrap replications required")
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    means = np.empty(n_bootstrap)
    offset = np.arange(block_length)
    for rep in range(n_bootstrap):
        start = rng.integers(0, n - block_length + 1, size=n_blocks)
        idx = (start[:, None] + offset[None, :]).reshape(-1)[:n]
        means[rep] = d[idx].mean()
    lo, hi = np.quantile(means, (0.025, 0.975))
    return {
        "mean_difference": float(d.mean()),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "bootstrap_means": means,
    }


def yearly_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    validate_predictions(pred)
    rows = []
    for year, frame in pred.groupby(pred.index.year):
        for name, column in PROBABILITY_COLUMNS.items():
            rows.append({
                "year": int(year), "model": name, "observations": len(frame),
                **probability_metrics(frame["actual_label"], frame[column]),
            })
    return pd.DataFrame(rows)


def volatility_regime_metrics(
    pred: pd.DataFrame,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature: str = "vol_21d",
) -> tuple[pd.DataFrame, float]:
    """Regime threshold from TRAIN 75th percentile; no future regime calibration."""
    validate_predictions(pred)
    if feature not in train.columns or feature not in validation.columns:
        raise ValueError(f"Missing {feature} for volatility regimes")
    if not pred.index.isin(validation.index).all():
        raise ValueError("Predictions and validation dataset dates are misaligned")
    threshold = float(train[feature].quantile(0.75))
    if not np.isfinite(threshold):
        raise ValueError("Invalid training-based regime threshold")
    values = validation.loc[pred.index, feature].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Invalid validation volatility feature")
    labels = np.where(values.to_numpy() > threshold, "High volatility", "Normal/low volatility")
    rows = []
    for regime in ("Normal/low volatility", "High volatility"):
        part = pred.loc[labels == regime]
        if part.empty:
            continue
        for name, col in PROBABILITY_COLUMNS.items():
            rows.append({
                "regime": regime, "model": name, "observations": len(part),
                **probability_metrics(part["actual_label"], part[col]),
            })
    return pd.DataFrame(rows), threshold


def rolling_log_loss(pred: pd.DataFrame, *, window: int = 63) -> pd.DataFrame:
    validate_predictions(pred)
    if not 2 <= window <= len(pred):
        raise ValueError("Invalid rolling window")
    result = pd.DataFrame(index=pred.index)
    for name, col in PROBABILITY_COLUMNS.items():
        losses = observation_log_loss(pred["actual_label"], pred[col])
        result[name] = pd.Series(losses, index=pred.index).rolling(window, min_periods=window).mean()
    return result


def calibration_table(pred: pd.DataFrame, *, n_bins: int = 5) -> pd.DataFrame:
    """Quantile reliability table; constant baseline appears as one bin."""
    validate_predictions(pred)
    if n_bins < 2:
        raise ValueError("Need at least two requested bins")
    rows = []
    for name, col in PROBABILITY_COLUMNS.items():
        frame = pred[["actual_label", col]].copy()
        if frame[col].nunique() == 1:
            labels = pd.Series(0, index=frame.index)
        else:
            # Integer category codes are stable for plotting; duplicate quantile
            # edges can reduce the realized number of nonempty bins.
            labels = pd.qcut(frame[col], q=n_bins, labels=False, duplicates="drop")
        frame["bin"] = labels
        grouped = frame.groupby("bin", observed=True).agg(
            observations=("actual_label", "size"),
            mean_probability=(col, "mean"),
            observed_frequency=("actual_label", "mean"),
        )
        for bin_index, values in grouped.iterrows():
            rows.append({
                "model": name, "bin": int(bin_index),
                "observations": int(values["observations"]),
                "mean_probability": float(values["mean_probability"]),
                "observed_frequency": float(values["observed_frequency"]),
            })
    return pd.DataFrame(rows)


def paired_losses(pred: pd.DataFrame) -> pd.DataFrame:
    """Log Loss improvement: baseline LR minus challenger, positive favors challenger."""
    validate_predictions(pred)
    y = pred["actual_label"].to_numpy()
    base = observation_log_loss(y, pred["p_baseline"])
    lr = observation_log_loss(y, pred["p_logistic"])
    xgb = observation_log_loss(y, pred["p_xgboost"])
    return pd.DataFrame({
        "log_loss_baseline": base,
        "log_loss_logistic": lr,
        "log_loss_xgboost": xgb,
        "lr_minus_xgboost": lr - xgb,
        "baseline_minus_logistic": base - lr,
        "baseline_minus_xgboost": base - xgb,
    }, index=pred.index)
