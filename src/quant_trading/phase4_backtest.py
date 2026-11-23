"""Phase 4.1–4.3 CLI: fixed-threshold development backtests for five models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .backtest import MODEL_COLUMNS, align_validation_predictions, run_backtest
from .performance import compute_performance, drawdown_series
from .pipeline import write_json


def run(root: Path, *, threshold: float=0.55, cost_bps: float=5.0,
        initial_capital: float=10_000.0) -> dict:
    root = Path(root).resolve()
    processed = root / "data" / "processed"
    prior = root / "reports" / "phase3_2" / "validation_predictions.csv"
    report = root / "reports" / "phase4_2"
    # Only 2019-2021 development validation data; no 2022+ test reads.
    val = pd.read_parquet(processed / "spy_validation.parquet")
    pred = pd.read_csv(prior, parse_dates=["signal_date"]).set_index("signal_date")
    data = align_validation_predictions(val, pred)
    experiments = {"Buy & Hold": run_backtest(data, cost_bps=cost_bps,
                                                initial_capital=initial_capital)}
    for model, col in MODEL_COLUMNS.items():
        experiments[model] = run_backtest(
            data, data[col], threshold=threshold, cost_bps=cost_bps,
            initial_capital=initial_capital,
        )
    report.mkdir(parents=True, exist_ok=True)
    trades = report / "backtests"
    trades.mkdir(exist_ok=True)
    equity = pd.DataFrame(index=next(iter(experiments.values())).index)
    drawdowns = pd.DataFrame(index=equity.index)
    metrics = []
    for model, bt in experiments.items():
        slug = model.lower().replace(" ", "_").replace("&", "and")
        bt.to_csv(trades / f"{slug}.csv", index_label="exit_date")
        equity[model] = bt["equity"]
        drawdowns[model] = drawdown_series(bt, initial_capital=initial_capital)
        metrics.append({"strategy": model, **compute_performance(bt, initial_capital=initial_capital)})
    equity.to_csv(report / "equity_curves.csv", index_label="exit_date")
    drawdowns.to_csv(report / "drawdown_curves.csv", index_label="exit_date")
    pd.DataFrame(metrics).to_csv(report / "performance_metrics.csv", index=False)
    config = {
        "phase": "4.1–4.3", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_period": "2019–2021 model-development validation only",
        "threshold": float(threshold), "cost_bps_per_side": float(cost_bps),
        "initial_capital": float(initial_capital),
        "models": list(MODEL_COLUMNS), "benchmark": "Buy & Hold SPY",
        "position_type": "0/1 long/cash; initial cash; no leverage or shorts",
        "cash_return": 0.0, "terminal_liquidation_fee_included": False,
        "execution": "Close(t) signal; Open(t+1) entry; Open(t+2) exit/rebalance",
        "return_source": "Phase 1 adjusted-open total-return proxy, not executable fills",
        "test_used": False,
        "warning": "Development backtests, not independent profitability evidence; Yahoo adjusted data revised retrospectively.",
    }
    write_json(report / "config.json", config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: research backtest on 2019-2021 development validation")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    config = run(args.root, threshold=args.threshold, cost_bps=args.cost_bps)
    print("Phase 4 backtesting completed (development ONLY). Report: reports/phase4_2")
    print("Test set not opened. Research adjusted-open returns are NOT executable fills.")


if __name__ == "__main__":
    main()
