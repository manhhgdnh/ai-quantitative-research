"""Phase 4.4 CLI: threshold, regime, cost and paired block-bootstrap checks.

Run only on the 2019–2021 development validation period. Prior model-family
comparisons make all reported CIs exploratory, not independently confirmatory.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import MODEL_COLUMNS, align_validation_predictions, run_backtest
from .pipeline import write_json
from .performance import compute_performance
from .strategy_research import (
    DEFAULT_THRESHOLDS, bootstrap_paired_excess_returns, choose_threshold,
    cost_sensitivity, threshold_sweep, volatility_regime_analysis,
)


def run(root: Path, *, model: str="XGBoost", thresholds=DEFAULT_THRESHOLDS,
        cost_bps: float=5.0, min_exposure: float=0.10, max_exposure: float=0.95,
        block_lengths=(5, 10, 20), n_bootstrap: int=2000, seed: int=42,
        initial_capital: float=10_000.0) -> dict:
    if model not in MODEL_COLUMNS:
        raise ValueError(f"Unknown model {model}; options: {list(MODEL_COLUMNS)}")
    root = Path(root).resolve()
    processed = root / "data" / "processed"
    report = root / "reports" / "phase4_4"
    train = pd.read_parquet(processed / "spy_train.parquet")
    validation = pd.read_parquet(processed / "spy_validation.parquet")
    pred = pd.read_csv(root / "reports" / "phase3_2" / "validation_predictions.csv",
                       parse_dates=["signal_date"]).set_index("signal_date")
    data = align_validation_predictions(validation, pred)
    prob = data[MODEL_COLUMNS[model]]
    sweep = threshold_sweep(data, prob, thresholds=thresholds, cost_bps=cost_bps,
                            initial_capital=initial_capital)
    selection = choose_threshold(sweep, min_exposure=min_exposure,
                                 max_exposure=max_exposure, fallback_threshold=0.50)
    threshold = selection["threshold"]
    selected_bt = run_backtest(data, prob, threshold=threshold, cost_bps=cost_bps,
                                initial_capital=initial_capital)
    benchmark_bt = run_backtest(data, cost_bps=cost_bps, initial_capital=initial_capital)
    regimes, regime_summary, vol_cutoff = volatility_regime_analysis(
        selected_bt, benchmark_bt, train, validation)
    costs = cost_sensitivity(data, prob, threshold=threshold, costs=(0, 5, 10),
                             initial_capital=initial_capital)
    blocks = bootstrap_paired_excess_returns(
        selected_bt, benchmark_bt, block_lengths=block_lengths,
        n_bootstrap=n_bootstrap, seed=seed)
    yearly = regimes.assign(year=pd.DatetimeIndex(regimes["signal_date"]).year).groupby(
        "year", as_index=False,
    ).agg(observations=("excess_return", "size"),
          mean_daily_excess_return=("excess_return", "mean"),
          market_exposure=("position", "mean"))
    report.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(report / "threshold_sweep.csv", index=False)
    selected_bt.to_csv(report / "selected_strategy_backtest.csv", index_label="exit_date")
    benchmark_bt.to_csv(report / "benchmark_backtest.csv", index_label="exit_date")
    regimes.to_csv(report / "regime_observations.csv", index=False)
    regime_summary.to_csv(report / "regime_metrics.csv", index=False)
    costs.to_csv(report / "cost_sensitivity.csv", index=False)
    blocks.to_csv(report / "block_bootstrap.csv", index=False)
    yearly.to_csv(report / "yearly_excess_returns.csv", index=False)
    metrics = pd.DataFrame([
        {"strategy": f"{model} Long/Cash", **compute_performance(selected_bt, initial_capital=initial_capital)},
        {"strategy": "Buy & Hold", **compute_performance(benchmark_bt, initial_capital=initial_capital)},
    ])
    metrics.to_csv(report / "selected_vs_benchmark_metrics.csv", index=False)
    config = {
        "phase": "4.4", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "development_interval": "2019–2021", "test_interval": "2022–2026-10-30",
        "model_name": model, "model_prediction_column": MODEL_COLUMNS[model],
        "selected_threshold": threshold, "threshold_grid": [float(v) for v in thresholds],
        "selection_metric": "max development net Sharpe within exposure constraints",
        "minimum_exposure": min_exposure, "maximum_exposure": max_exposure,
        "eligibility_met": selection["eligibility_met"],
        "selection_status": selection["selection_status"],
        "selection_warning": selection["selection_warning"],
        "cost_bps_per_side": cost_bps, "cost_scenarios_bps_per_side": [0, 5, 10],
        "training_volatility_75pct_cutoff": vol_cutoff,
        "bootstrap_lengths": [int(x) for x in block_lengths],
        "bootstrap_replicates": n_bootstrap, "bootstrap_seed": seed,
        "initial_capital": initial_capital, "cash_return": 0.0,
        "test_used_for_selection": False, "test_data_opened": False,
        "independence_warning": "Exploratory block bootstrap does NOT adjust for model, threshold or benchmark search or market regime shifts.",
        "price_warning": "Retrospective Yahoo adjusted-open return proxy, NOT point-in-time order fills.",
        "final_test_requires": "Predeclare model, scalers, refit protocol, threshold and benchmark; use executable prices/dividend model for deployment claims.",
    }
    write_json(report / "strategy_config.json", config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4.4 development diagnostics, test held out")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--model", choices=list(MODEL_COLUMNS), default="XGBoost")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--bootstrap", type=int, default=2000, help="Replications, minimum 100")
    args = parser.parse_args()
    config = run(args.root, model=args.model, cost_bps=args.cost_bps, n_bootstrap=args.bootstrap)
    print("Phase 4.4 development diagnostics completed.")
    print("Model:", config["model_name"], "threshold:", config["selected_threshold"])
    print("Selection:", config["selection_status"], "(not independent evidence)")
    print("Reports: reports/phase4_4. Test set not opened.")


if __name__ == "__main__":
    main()
