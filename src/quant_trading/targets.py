"""Next-open-to-next-open target; adjustment factor has EOD/hindsight limitations."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """At close t: enter open t+1, exit open t+2; target uses an EOD-adjusted proxy.

    This is NOT a point-in-time observation of an adjusted opening price:
    Adj Close / Close for session t+2 is based on EOD data and can be revised
    subsequently. Conservative train/validation purging requires exit_date <
    first validation SIGNAL date. Backtests need raw opens and dividend events.
    """
    for col in ("Open", "Close", "Adj Close"):
        if col not in df:
            raise ValueError(f"Missing column: {col}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Expected datetime-indexed market data")
    if not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Require sorted unique market dates")
    prices = df[["Open", "Close", "Adj Close"]].astype(float)
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("Invalid prices")

    adjusted_open = prices["Open"] * prices["Adj Close"] / prices["Close"]
    dates = pd.Series(df.index, index=df.index)
    entry = adjusted_open.shift(-1)
    exit_ = adjusted_open.shift(-2)
    future_return = exit_ / entry - 1
    y = (future_return > 0).astype("Int64").where(future_return.notna())

    return pd.DataFrame({
        "entry_date": dates.shift(-1),
        "exit_date": dates.shift(-2),   # exit session; proxy adjustment may require that session EOD
        "future_return": future_return,
        "label": y,
    }, index=df.index).rename_axis("signal_date")
