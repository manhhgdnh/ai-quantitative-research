"""Synthetic tests do not need a live Yahoo connection or market-data downloads."""

import numpy as np
import pandas as pd
import pytest
import exchange_calendars as xcals

from quant_trading.data import validate_spy
from quant_trading.features import FEATURE_COLUMNS, build_features, wilder_rsi
from quant_trading.targets import build_targets
from quant_trading.splits import chronological_splits, make_supervised


def sample_market_data():
    """Synthetic full XNYS daily series spanning all prescribed splits."""
    dates = xcals.get_calendar("XNYS").sessions_in_range("2016-01-04", "2024-12-31")
    k = np.arange(len(dates))
    close = 100.0 * np.exp(np.cumsum(0.0003 + 0.004 * np.sin(k / 8)))
    open_ = close * (1.0 + 0.0006 * np.sin(k / 3))
    high = np.maximum(close, open_) * 1.003
    low = np.minimum(close, open_) * 0.997
    adj = close * 0.85  # fixed synthetic adjustment factor
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Adj Close": adj,
        "Volume": 1_000_000 + 1_000 * (k % 15),
        "Dividends": np.zeros(len(k)), "Stock Splits": np.zeros(len(k)),
    }, index=dates.rename("Date"))


def test_wilder_rsi_constant_series_and_rising_series():
    index = pd.bdate_range("2020-01-01", periods=80)
    flat = pd.Series(100.0, index=index)
    rising = pd.Series(100.0 + np.arange(80), index=index)
    assert np.isnan(wilder_rsi(flat).iloc[13])
    assert wilder_rsi(flat).iloc[14] == pytest.approx(50)
    assert wilder_rsi(rising).iloc[14] == pytest.approx(100)


def test_features_are_causal_and_correct_shape():
    data = sample_market_data()
    features = build_features(data)
    assert features.columns.tolist() == list(FEATURE_COLUMNS)
    assert len(FEATURE_COLUMNS) == 17
    assert features.index.equals(data.index)
    assert features.iloc[:63].isna().any(axis=None)
    assert np.isfinite(features.iloc[100:].to_numpy(dtype=float)).all()
    cutoff = pd.Timestamp("2020-07-01")
    # Prefix invariance: adding future rows must not alter past features.
    truncated = build_features(data.loc[:cutoff])
    pd.testing.assert_frame_equal(features.loc[:cutoff], truncated, check_freq=False)
    # The future values can change, but past features cannot.
    altered = data.copy()
    altered.loc[altered.index > cutoff, "Adj Close"] *= 1.1
    altered.loc[altered.index > cutoff, "Volume"] *= 2.0
    pd.testing.assert_frame_equal(
        features.loc[:cutoff], build_features(altered).loc[:cutoff], check_freq=False
    )


def test_open_to_open_target_alignment_and_unknown_end_labels():
    dates = pd.bdate_range("2020-01-06", periods=5)
    prices = [100.0, 101.0, 104.0, 103.0, 106.0]
    data = pd.DataFrame({"Open": prices, "Close": prices, "Adj Close": prices}, index=dates)
    y = build_targets(data)
    assert y["label"].iloc[:3].tolist() == [1, 0, 1]
    assert y["label"].iloc[-2:].isna().all()
    assert y["future_return"].iloc[0] == pytest.approx(104 / 101 - 1)
    assert y["entry_date"].iloc[0] == dates[1]
    assert y["exit_date"].iloc[0] == dates[2]


def test_targets_depend_on_future_but_not_on_earlier_future_changes():
    data = sample_market_data()
    base = build_targets(data)
    modified = data.copy()
    k = 200
    modified.iloc[k, modified.columns.get_loc("Open")] *= 1.02
    # A changed future opening price must not change labels before t = k-2.
    changed = build_targets(modified)
    pd.testing.assert_series_equal(
        base["future_return"].iloc[:k-2], changed["future_return"].iloc[:k-2]
    )


def test_calendar_and_missing_session_detection():
    data = sample_market_data()
    validate_spy(data)
    missing = data.drop(index=data.index[100])
    with pytest.raises(ValueError, match="Missing NYSE sessions"):
        validate_spy(missing)


def test_training_split_purges_labels_not_yet_known():
    data = sample_market_data()
    supervised = make_supervised(build_features(data), build_targets(data))
    partitions = chronological_splits(supervised)
    train, val, test = (partitions[k] for k in ("train", "validation", "test"))
    assert train.index.max() < val.index.min() < test.index.min()
    assert train["exit_date"].max() < val.index.min()
    assert val["exit_date"].max() < test.index.min()
    assert len(train) > 50 and len(val) > 50 and len(test) > 50
    assert set(train["label"].unique()).issubset({0, 1})


def test_calendar_covers_historical_start_2005():
    """A default moving calendar horizon must not reject our 2005 SPY rows."""
    calendar = xcals.get_calendar("XNYS", start="2005-01-01", end="2006-01-31")
    dates = calendar.sessions_in_range("2005-01-03", "2005-07-01")
    n = len(dates)
    data = pd.DataFrame({
        "Open": np.full(n, 100.0),
        "High": np.full(n, 101.0),
        "Low": np.full(n, 99.0),
        "Close": np.full(n, 100.0),
        "Adj Close": np.full(n, 100.0),
        "Volume": np.full(n, 1000.0),
    }, index=dates.rename("Date"))
    validate_spy(data, check_calendar=True)
