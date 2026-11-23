"""Phase 4 strategy development on 2019–2021 only; not final OOS evidence."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .performance import compute_performance
from .robustness import moving_block_bootstrap

DEFAULT_THRESHOLDS = (0.50, 0.55, 0.60)
DEFAULT_COST_SCENARIOS_BPS = (0.0, 5.0, 10.0)
DEFAULT_BLOCK_LENGTHS = (5, 10, 20)


def threshold_sweep(
    frame: pd.DataFrame, probabilities: pd.Series, *,
    thresholds=DEFAULT_THRESHOLDS, cost_bps: float=5.0,
    initial_capital: float=10_000.0,
) -> pd.DataFrame:
    """Development-only exploration. A test-period Sharpe must NOT be optimized."""
    rows = []
    for threshold in thresholds:
        bt = run_backtest(frame, probabilities, threshold=float(threshold),
                          cost_bps=cost_bps, initial_capital=initial_capital)
        rows.append({"threshold": float(threshold), **compute_performance(bt, initial_capital=initial_capital)})
    out = pd.DataFrame(rows)
    if out.empty or out.threshold.duplicated().any():
        raise ValueError("Threshold list must be nonempty and unique")
    return out


def choose_threshold(
    sweep: pd.DataFrame, *, min_exposure: float=0.10,
    max_exposure: float=0.95, fallback_threshold: float=0.50,
) -> dict:
    """Pick eligible max Sharpe, else fixed, explicitly FLAGGED fallback.

    Fallback is not described as a CV/Sharpe-optimized successful strategy.
    """
    if not 0 <= min_exposure <= max_exposure <= 1:
        raise ValueError("Invalid exposure constraints")
    if sweep.empty or not {"threshold", "market_exposure", "sharpe_ratio", "total_turnover"}.issubset(sweep):
        raise ValueError("Missing threshold sweep results")
    eligible = sweep.loc[
        sweep["market_exposure"].between(min_exposure, max_exposure)
        & np.isfinite(sweep["sharpe_ratio"])
    ].copy()
    if eligible.empty:
        if not np.isclose(sweep["threshold"].to_numpy(dtype=float), fallback_threshold).any():
            raise ValueError("No eligible strategy and specified fixed fallback not in grid")
        return {
            "threshold": float(fallback_threshold),
            "eligibility_met": False,
            "selection_status": "fixed_threshold_fallback_no_eligible_candidate",
            "selection_warning": "NO eligible threshold. Fallback is a predeclared diagnostic rule, not a successful optimized strategy.",
        }
    best = eligible.sort_values(
        ["sharpe_ratio", "total_turnover", "threshold"],
        ascending=[False, True, True], kind="mergesort",
    ).iloc[0]
    return {
        "threshold": float(best["threshold"]),
        "eligibility_met": True,
        "selection_status": "development_max_sharpe_subject_to_exposure",
        "selection_warning": "Optimized on 2019–2021; reported development Sharpe is selection-biased. NOT independent evidence.",
    }


def volatility_regime_analysis(
    strategy: pd.DataFrame, benchmark: pd.DataFrame,
    train: pd.DataFrame, validation: pd.DataFrame,
    *, feature: str = "vol_21d", quantile: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Analyze a FULL continuous backtest by signal-time volatility regimes.

    Never reset positions or recompute trade costs within regime subsets.
    Do not annualize scattered regime observations as a continuous portfolio.
    """
    if not 0 < quantile < 1:
        raise ValueError("Regime quantile must lie in (0,1)")
    if not strategy.index.equals(benchmark.index) or not strategy["signal_date"].equals(benchmark["signal_date"]):
        raise ValueError("Strategy/benchmark dates are not exactly aligned")
    if feature not in train or feature not in validation:
        raise ValueError("Volatility feature missing")
    threshold = float(train[feature].quantile(quantile))
    if not np.isfinite(threshold):
        raise ValueError("Invalid training volatility quantile")
    signal_dates = pd.DatetimeIndex(strategy["signal_date"], name="signal_date")
    if not signal_dates.isin(validation.index).all():
        raise ValueError("Some backtest signals missing from development validation")
    vol = validation.loc[signal_dates, feature].to_numpy(dtype=float)
    if not np.isfinite(vol).all():
        raise ValueError("Non-finite signal-time volatility features")
    detailed = pd.DataFrame({
        "signal_date": signal_dates,
        "exit_date": strategy.index,
        "vol_21d": vol,
        "regime": np.where(vol > threshold, "High volatility", "Normal/low volatility"),
        "strategy_return": strategy["net_return"].to_numpy(dtype=float),
        "benchmark_return": benchmark["net_return"].to_numpy(dtype=float),
        "position": strategy["position"].to_numpy(dtype=int),
        "turnover": strategy["turnover"].to_numpy(dtype=int),
    })
    detailed["excess_return"] = detailed["strategy_return"] - detailed["benchmark_return"]
    summary = detailed.groupby("regime", as_index=False).agg(
        observations=("excess_return", "size"),
        mean_strategy_return=("strategy_return", "mean"),
        mean_benchmark_return=("benchmark_return", "mean"),
        mean_excess_return=("excess_return", "mean"),
        market_exposure=("position", "mean"),
        one_sided_trades=("turnover", "sum"),
    )
    return detailed, summary, threshold


def cost_sensitivity(
    frame: pd.DataFrame, probabilities: pd.Series, *,
    threshold: float, costs=DEFAULT_COST_SCENARIOS_BPS,
    initial_capital: float=10_000.0,
) -> pd.DataFrame:
    rows = []
    for cost in costs:
        bt = run_backtest(frame, probabilities, threshold=threshold,
                          cost_bps=float(cost), initial_capital=initial_capital)
        rows.append({"cost_bps": float(cost), **compute_performance(bt, initial_capital=initial_capital)})
    if not rows:
        raise ValueError("Must supply at least one cost scenario")
    return pd.DataFrame(rows)


def bootstrap_paired_excess_returns(
    strategy: pd.DataFrame, benchmark: pd.DataFrame, *,
    block_lengths=DEFAULT_BLOCK_LENGTHS,
    n_bootstrap: int=2000, seed: int=42,
) -> pd.DataFrame:
    """Exploratory block-bootstrap CI for mean STRATEGY−BENCHMARK return.

    Conditional on already selected models/thresholds; does not account
    for multiple testing, regimes, model selection or future shifts.
    """
    if not strategy.index.equals(benchmark.index):
        raise ValueError("Strategy and benchmark exits must match")
    d = strategy["net_return"].to_numpy(dtype=float) - benchmark["net_return"].to_numpy(dtype=float)
    rows = []
    for length in block_lengths:
        result = moving_block_bootstrap(
            d, block_length=int(length), n_bootstrap=n_bootstrap, seed=seed,
        )
        rows.append({
            "block_length": int(length),
            "observations": int(len(d)),
            "mean_daily_excess_return": result["mean_difference"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
        })
    if not rows:
        raise ValueError("Must specify at least one bootstrap block length")
    return pd.DataFrame(rows)
