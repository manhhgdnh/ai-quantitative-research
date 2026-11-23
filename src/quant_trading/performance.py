"""Return and risk metrics for the fixed Phase-4 research backtest."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _validate_backtest(backtest: pd.DataFrame, initial_capital: float) -> None:
    needed = {"entry_date", "net_return", "equity", "position", "turnover", "cost_fraction"}
    if backtest.empty or not needed.issubset(backtest.columns):
        raise ValueError("Empty backtest or missing performance columns")
    if (not isinstance(backtest.index, pd.DatetimeIndex)
            or not backtest.index.is_monotonic_increasing
            or backtest.index.has_duplicates):
        raise ValueError("Require sorted unique exit-date index")
    if not np.isfinite(initial_capital) or initial_capital <= 0:
        raise ValueError("Initial capital must be strictly positive")
    for col in ("net_return", "equity", "position", "turnover", "cost_fraction"):
        if not np.isfinite(backtest[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite {col}")
    r = backtest["net_return"].to_numpy(dtype=float)
    equity = backtest["equity"].to_numpy(dtype=float)
    if (r <= -1).any() or (equity <= 0).any():
        raise ValueError("Nonpositive equity / return <= -100%")
    derived = initial_capital * np.cumprod(1 + r)
    if not np.allclose(derived, equity, rtol=1e-8, atol=1e-5):
        raise ValueError("Equity does not match compounded net returns")
    if not np.isin(backtest["position"], [0,1]).all():
        raise ValueError("Only binary long/cash positions are supported")
    if (backtest["turnover"] < 0).any() or (backtest["cost_fraction"] < 0).any():
        raise ValueError("Invalid turnover or costs")
    entry = pd.to_datetime(backtest["entry_date"])
    if entry.isna().any() or not (entry.to_numpy() < backtest.index.to_numpy()).all():
        raise ValueError("Invalid entry/exit dates")


def drawdown_series(backtest: pd.DataFrame, initial_capital: float = 10_000.0) -> pd.Series:
    """Drawdown from running peak including initial capital at t0."""
    _validate_backtest(backtest, initial_capital)
    equity = backtest["equity"].to_numpy(dtype=float)
    peaks = np.maximum.accumulate(np.r_[initial_capital, equity])[1:]
    return pd.Series(equity / peaks - 1, index=backtest.index, name="drawdown")


def compute_performance(
    backtest: pd.DataFrame, *, initial_capital: float = 10_000.0,
    risk_free_annual: float = 0.0,
) -> dict:
    """Conventional daily Sharpe/volatility; cash return is 0 in engine.

    Nonzero risk-free rate in the Sharpe subtracts a reference from daily
    portfolio returns; it does NOT add cash yield to the simulated equity.
    """
    _validate_backtest(backtest, initial_capital)
    if not np.isfinite(risk_free_annual) or risk_free_annual <= -1:
        raise ValueError("Invalid annual risk-free rate")
    r = backtest["net_return"].to_numpy(dtype=float)
    eq = backtest["equity"].to_numpy(dtype=float)
    first_entry = pd.Timestamp(backtest["entry_date"].iloc[0])
    last_exit = backtest.index[-1]
    years = (last_exit - first_entry).total_seconds() / (365.25 * 86400)
    if years <= 0:
        raise ValueError("Invalid annualization duration")
    cumulative = eq[-1] / initial_capital - 1
    cagr = (eq[-1] / initial_capital) ** (1 / years) - 1
    volatility = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else float("nan")
    daily_rf = (1 + risk_free_annual) ** (1 / TRADING_DAYS) - 1
    excess = r - daily_rf
    e_std = float(np.std(excess, ddof=1)) if len(excess) > 1 else float("nan")
    sharpe = float(np.sqrt(TRADING_DAYS) * np.mean(excess) / e_std) if e_std > 0 else float("nan")
    dd = drawdown_series(backtest, initial_capital=initial_capital)
    mdd = float(-dd.min())
    calmar = float(cagr / mdd) if mdd > 0 else float("nan")
    before = np.r_[initial_capital, eq[:-1]]
    cost_dollars = float(np.dot(before, backtest["cost_fraction"].to_numpy(dtype=float)))
    return {
        "cumulative_return": float(cumulative),
        "cagr": float(cagr),
        "annualized_volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": mdd,
        "calmar_ratio": calmar,
        "market_exposure": float(backtest["position"].mean()),
        "one_sided_trades": int(backtest["turnover"].sum()),
        "total_turnover": float(backtest["turnover"].sum()),
        "estimated_cost_dollars": cost_dollars,
        "observations": int(len(backtest)),
        "first_entry": str(first_entry.date()),
        "last_exit": str(last_exit.date()),
    }
