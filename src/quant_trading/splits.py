"""Temporal dataset partition with label-maturity purging."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


def make_supervised(features: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Join on original calendar BEFORE filtering warm-up and missing labels."""
    if not features.index.equals(targets.index):
        raise ValueError("Feature and target date indices must match exactly")
    joined = features.join(targets, validate="one_to_one")
    finite_features = np.isfinite(joined[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all(axis=1)
    observed_labels = joined["label"].notna() & joined["entry_date"].notna() & joined["exit_date"].notna()
    out = joined.loc[finite_features & observed_labels].copy()
    out["label"] = out["label"].astype("int64")
    if out.empty:
        raise ValueError("No complete supervised observations")
    if not (out["exit_date"] > out["entry_date"]).all():
        raise ValueError("Invalid entry/exit ordering")
    return out


def chronological_splits(supervised: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train 2005-18, validation 2019-21, untouched test 2022-Oct 2026.

    Purge rows whose label (known at exit open) crosses into the next split.
    """
    train = supervised.loc["2005-01-01":"2018-12-31"].copy()
    val = supervised.loc["2019-01-01":"2021-12-31"].copy()
    test = supervised.loc["2022-01-01":"2026-10-30"].copy()
    if any(x.empty for x in (train, val, test)):
        raise ValueError("Insufficient observations for predefined time partitions")
    val_start, test_start = val.index.min(), test.index.min()
    train = train.loc[train["exit_date"] < val_start].copy()
    val = val.loc[val["exit_date"] < test_start].copy()
    if train.empty or val.empty:
        raise ValueError("Empty partition after label-maturity purging")
    if not (train["exit_date"].max() < val_start and val["exit_date"].max() < test_start):
        raise AssertionError("Label leakage across partition boundaries")
    if not (train.index.max() < val.index.min() < test.index.min()):
        raise AssertionError("Split ordering is invalid")
    return {"train": train, "validation": val, "test": test}
