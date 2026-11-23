"""Shared PyTorch helpers. CPU-first, deterministic seeds, no future-data fitting."""
from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


def set_seed(seed: int) -> None:
    """Set CPU experiment seeds; cross-platform bitwise identity is not guaranteed."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def feature_scaler(fit_features: pd.DataFrame) -> StandardScaler:
    """Fit only on chronological training-fold observations."""
    if list(fit_features.columns) != list(FEATURE_COLUMNS):
        raise ValueError("Feature columns/order mismatch")
    if fit_features.empty or not np.isfinite(fit_features.to_numpy(dtype=float)).all():
        raise ValueError("Scaler training features must be nonempty and finite")
    return StandardScaler().fit(fit_features)


def chronological_folds(frame: pd.DataFrame, *, n_splits: int = 4,
                        test_size: int = 252, gap: int = 2):
    """Yield (fold_number, fit_idx, validation_idx), checking label maturity."""
    if gap < 2:
        raise ValueError("For this t+2 label, gap must be at least 2")
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(frame), start=1):
        exit_times = pd.to_datetime(frame["exit_date"].iloc[fit_idx])
        if exit_times.max() >= frame.index[val_idx[0]]:
            raise ValueError(f"Fold {fold}: immature labels cross validation start")
        yield fold, fit_idx, val_idx


def verify_train_validation(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    if not isinstance(train.index, pd.DatetimeIndex) or not isinstance(validation.index, pd.DatetimeIndex):
        raise ValueError("Signal dates must be a DatetimeIndex")
    for frame in (train, validation):
        if frame.empty or not frame.index.is_monotonic_increasing or not frame.index.is_unique:
            raise ValueError("Expected nonempty, ordered and unique signal dates")
    if train["exit_date"].max() >= validation.index.min():
        raise ValueError("Training label not mature before validation signal")
    if train.index.max() >= validation.index.min():
        raise ValueError("Train and validation signals overlap")


def verify_previous_predictions(prev: pd.DataFrame, validation: pd.DataFrame) -> None:
    if not prev.index.equals(validation.index):
        raise ValueError("Prediction index is not aligned with validation signals")
    if not np.array_equal(prev["actual_label"].to_numpy(), validation["label"].to_numpy()):
        raise ValueError("Previous label predictions not aligned")
    for column in ("p_baseline", "p_logistic", "p_xgboost"):
        if column not in prev:
            raise ValueError(f"Missing previous predictions: {column}")


def probability_table(validation: pd.DataFrame, forecasts: dict[str, Sequence[float]]) -> pd.DataFrame:
    from .ml import probability_metrics
    return pd.DataFrame([
        {"model": name, **probability_metrics(validation["label"], probabilities)}
        for name, probabilities in forecasts.items()
    ]).sort_values("log_loss").reset_index(drop=True)
