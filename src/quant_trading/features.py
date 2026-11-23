"""Causal, end-of-day financial features (all information available after close t)."""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "ret_1d", "ret_lag_1", "ret_lag_2", "ret_lag_5",
    "momentum_5d", "momentum_21d",
    "vol_21d", "vol_63d", "vol_ratio",
    "sma_gap_20", "ema_gap_12", "ema_gap_26",
    "rsi_14", "macd_norm", "macd_hist_norm",
    "bb_position", "relative_volume_20",
)


def wilder_rsi(price: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI with a period-long initial simple-average seed."""
    if period < 1:
        raise ValueError("RSI period must be positive")
    if price.isna().any() or (price <= 0).any():
        raise ValueError("RSI requires nonmissing, strictly positive prices")

    change = price.diff()
    gains = change.clip(lower=0).to_numpy(dtype=float)
    losses = (-change.clip(upper=0)).to_numpy(dtype=float)
    out = np.full(len(price), np.nan)
    if len(price) <= period:
        return pd.Series(out, index=price.index, name="rsi_14")
    avg_gain = float(np.mean(gains[1:period + 1]))
    avg_loss = float(np.mean(losses[1:period + 1]))
    for i in range(period, len(price)):
        if i > period:
            avg_gain = ((period - 1) * avg_gain + gains[i]) / period
            avg_loss = ((period - 1) * avg_loss + losses[i]) / period
        denom = avg_gain + avg_loss
        out[i] = 50.0 if denom == 0 else 100.0 * avg_gain / denom
    return pd.Series(out, index=price.index, name="rsi_14")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate 17 features without target, global scaling, or negative shifts."""
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Require sorted unique trading sessions")
    price = df["Adj Close"].astype(float)
    volume = df["Volume"].astype(float)
    if not np.isfinite(price.to_numpy()).all() or (price <= 0).any():
        raise ValueError("Invalid adjusted close")
    if not np.isfinite(volume.to_numpy()).all() or (volume < 0).any():
        raise ValueError("Invalid volume")

    X = pd.DataFrame(index=df.index)
    r = np.log(price / price.shift(1))
    X["ret_1d"] = r
    for lag in (1, 2, 5):
        X[f"ret_lag_{lag}"] = r.shift(lag)
    for window in (5, 21):
        X[f"momentum_{window}d"] = np.log(price / price.shift(window))

    vol21 = r.rolling(21, min_periods=21).std(ddof=1)
    vol63 = r.rolling(63, min_periods=63).std(ddof=1)
    X["vol_21d"] = vol21 * np.sqrt(252)
    X["vol_63d"] = vol63 * np.sqrt(252)
    X["vol_ratio"] = vol21 / vol63.replace(0, np.nan)

    sma20 = price.rolling(20, min_periods=20).mean()
    X["sma_gap_20"] = price / sma20 - 1
    ema12 = price.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = price.ewm(span=26, adjust=False, min_periods=26).mean()
    X["ema_gap_12"] = price / ema12 - 1
    X["ema_gap_26"] = price / ema26 - 1

    X["rsi_14"] = wilder_rsi(price)
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    X["macd_norm"] = macd / price
    X["macd_hist_norm"] = (macd - signal) / price
    price_sd20 = price.rolling(20, min_periods=20).std(ddof=0)
    X["bb_position"] = (price - sma20) / (2 * price_sd20.replace(0, np.nan))
    X["relative_volume_20"] = volume / volume.rolling(20, min_periods=20).mean().replace(0, np.nan)

    assert list(X.columns) == list(FEATURE_COLUMNS)
    return X
