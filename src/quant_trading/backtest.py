"""Research-only SPY Long/Cash backtest on adjusted-open RETURN proxies.

Positions are decided after Close(t), executed at Open(t+1), and carried
until Open(t+2). The adjusted-open return is retrospective total-return
proxy, NOT actual executable entry/exit prices. No held-out test is read.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_COLUMNS = {
    "Constant Probability": "p_baseline",
    "Logistic Regression": "p_logistic",
    "XGBoost": "p_xgboost",
    "MLP": "p_mlp",
    "LSTM": "p_lstm",
}


def validate_backtest_frame(frame: pd.DataFrame) -> None:
    """Verify contiguous, chronologically valid one-session trade intervals."""
    needed = {"entry_date", "exit_date", "future_return", "label"}
    if not needed.issubset(frame.columns):
        raise ValueError(f"Missing backtest columns: {sorted(needed - set(frame.columns))}")
    if frame.empty or not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Expected nonempty DatetimeIndex of signal dates")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("Require sorted unique signal dates")
    dates = pd.Series(frame.index, index=frame.index)
    entry = pd.to_datetime(frame["entry_date"])
    exit_ = pd.to_datetime(frame["exit_date"])
    if entry.isna().any() or exit_.isna().any():
        raise ValueError("Null entry/exit dates")
    if not ((dates < entry) & (entry < exit_)).all():
        raise ValueError("Signal < entry < exit chronology violated")
    # Every next interval starts at the previous interval's exit. A gap in
    # signals makes raw per-period compounding/annualization invalid.
    if len(frame) > 1 and not np.array_equal(
        exit_.iloc[:-1].to_numpy(), entry.iloc[1:].to_numpy()
    ):
        raise ValueError("Nonconsecutive holding periods; require continuous validation data")
    r = frame["future_return"].to_numpy(dtype=float)
    labels = frame["label"].to_numpy(dtype=float)
    if not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Invalid SPY period returns")
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid binary labels")
    if not np.array_equal(labels.astype(int), (r > 0).astype(int)):
        raise ValueError("Future return and binary label mismatch")


def align_validation_predictions(validation: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Require exact matching signal dates/labels; no silent index joins."""
    validate_backtest_frame(validation)
    if (
        not isinstance(predictions.index, pd.DatetimeIndex)
        or predictions.index.has_duplicates
        or not predictions.index.equals(validation.index)
        or "actual_label" not in predictions
    ):
        raise ValueError("Predictions require exact validation signal-date alignment")
    if not np.array_equal(
        predictions["actual_label"].to_numpy(dtype=float),
        validation["label"].to_numpy(dtype=float),
    ):
        raise ValueError("Validation labels and predictions disagree")
    for col in MODEL_COLUMNS.values():
        if col not in predictions:
            raise ValueError(f"Missing {col}: run all Phase 1–3 model scripts first")
        p = predictions[col].to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError(f"Invalid probabilities in {col}")
    combined = validation[["entry_date", "exit_date", "future_return", "label"]].copy()
    for col in MODEL_COLUMNS.values():
        combined[col] = predictions[col].to_numpy(dtype=float)
    return combined


def run_backtest(
    frame: pd.DataFrame,
    probabilities: pd.Series | None = None,
    *,
    threshold: float = 0.55,
    cost_bps: float = 5.0,
    initial_capital: float = 10_000.0,
) -> pd.DataFrame:
    """Full-capital 0/1 long/cash research accounting, cash return = 0.

    Cost convention: proportional trading fee paid at each session's entry
    open, before holding-period market return. For continuously long
    positions no additional daily charge. Portfolio is not liquidated at
    final exit open (mark-to-market only). Initial position is 100% cash.
    """
    validate_backtest_frame(frame)
    if not np.isfinite([threshold, cost_bps, initial_capital]).all():
        raise ValueError("Parameters must be finite")
    if not 0 <= threshold <= 1 or not 0 <= cost_bps < 10_000 or initial_capital <= 0:
        raise ValueError("Invalid threshold/cost/capital")
    n = len(frame)
    if probabilities is None:
        position = np.ones(n, dtype=np.int8)
        p = np.full(n, np.nan)
    else:
        if not isinstance(probabilities, pd.Series) or not probabilities.index.equals(frame.index):
            raise ValueError("Prediction Series must have exact signal-date alignment")
        p = probabilities.to_numpy(dtype=float)
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError("Probabilities must be finite values in [0,1]")
        position = (p > threshold).astype(np.int8)
    previous_position = np.r_[0, position[:-1]]
    turnover = np.abs(position.astype(int) - previous_position).astype(int)
    cost_fraction = (float(cost_bps) / 10000) * turnover
    market_return = frame["future_return"].to_numpy(dtype=float)
    gross_return = position * market_return
    net_return = (1 - cost_fraction) * (1 + gross_return) - 1
    if (net_return <= -1).any():
        raise ValueError("Portfolio equity cannot remain strictly positive")
    equity = float(initial_capital) * np.cumprod(1 + net_return)
    if not np.isfinite(equity).all() or (equity <= 0).any():
        raise ValueError("Overflow or nonpositive equity")
    return pd.DataFrame({
        "signal_date": frame.index.to_numpy(),
        "entry_date": pd.to_datetime(frame["entry_date"]).to_numpy(),
        "probability": p,
        "position": position,
        "turnover": turnover,
        "cost_fraction": cost_fraction,
        "market_return": market_return,
        "gross_return": gross_return,
        "net_return": net_return,
        "equity": equity,
    }, index=pd.DatetimeIndex(pd.to_datetime(frame["exit_date"]), name="exit_date"))
