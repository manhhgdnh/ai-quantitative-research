"""Phase 2.3: temporal stability, calibration, gain diagnostics and block bootstrap."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLUMNS
from .ml import make_logistic, xy
from .pipeline import write_json
from .robustness import (
    calibration_table, moving_block_bootstrap, paired_losses,
    rolling_log_loss, validate_predictions, volatility_regime_metrics,
    yearly_metrics,
)


def coefficient_stability(
    train: pd.DataFrame, *, C: float, n_splits: int = 4,
    test_size: int = 252, gap: int = 2,
) -> pd.DataFrame:
    """Fit standardized Logistic Regression independently on each CV training fold."""
    X, y = xy(train)
    splitter = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    rows = []
    for fold, (fit_idx, val_idx) in enumerate(splitter.split(X), start=1):
        if train["exit_date"].iloc[fit_idx].max() >= train.index[val_idx[0]]:
            raise ValueError("Immature training label during coefficient stability check")
        model = make_logistic(C=C)
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        beta = model.named_steps["classifier"].coef_[0]
        for feature, coefficient in zip(FEATURE_COLUMNS, beta):
            rows.append({"fold": fold, "feature": feature, "coefficient_scaled": float(coefficient)})
    return pd.DataFrame(rows)


def run(root: Path, *, n_bootstrap: int = 2000, block_lengths=(5, 10, 20)) -> dict:
    root = Path(root).resolve()
    report = root / "reports" / "phase2_3"
    report.mkdir(parents=True, exist_ok=True)
    data_dir = root / "data" / "processed"
    prev = root / "reports" / "phase2_2"
    baseline = root / "reports" / "phase2_1"

    # Strictly no final test read; no fitting on validation labels.
    pred = pd.read_csv(
        prev / "validation_predictions.csv", parse_dates=["signal_date"],
    ).set_index("signal_date").sort_index()
    validate_predictions(pred)
    train = pd.read_parquet(data_dir / "spy_train.parquet")
    validation = pd.read_parquet(data_dir / "spy_validation.parquet")
    if not pred.index.equals(validation.index):
        raise ValueError("Prediction dates do not match validation dates")
    if not pred["actual_label"].equals(validation["label"]):
        raise ValueError("Prediction labels differ from validation labels")

    yearly = yearly_metrics(pred)
    regime, threshold = volatility_regime_metrics(pred, train, validation)
    rolling = rolling_log_loss(pred, window=63)
    calibration = calibration_table(pred, n_bins=5)
    losses = paired_losses(pred)
    comparison = []
    for column in ("lr_minus_xgboost", "baseline_minus_logistic", "baseline_minus_xgboost"):
        for length in block_lengths:
            bs = moving_block_bootstrap(losses[column], block_length=int(length),
                                       n_bootstrap=n_bootstrap, seed=42)
            comparison.append({
                "comparison": column,
                "block_length": int(length),
                "n_bootstrap": n_bootstrap,
                "mean_difference": bs["mean_difference"],
                "ci_lower": bs["ci_lower"],
                "ci_upper": bs["ci_upper"],
            })

    cfg = json.loads((baseline / "config.json").read_text(encoding="utf-8"))
    coefficients = coefficient_stability(
        train, C=float(cfg["chosen_C"]), n_splits=int(cfg["cv_n_splits"]),
        test_size=int(cfg["cv_test_size"]), gap=int(cfg["cv_gap"]),
    )
    correlations = train[list(FEATURE_COLUMNS)].corr()

    yearly.to_csv(report / "yearly_metrics.csv", index=False)
    regime.to_csv(report / "regime_metrics.csv", index=False)
    rolling.to_csv(report / "rolling_log_loss.csv", index_label="signal_date")
    calibration.to_csv(report / "calibration_bins.csv", index=False)
    losses.to_csv(report / "paired_losses.csv", index_label="signal_date")
    pd.DataFrame(comparison).to_csv(report / "block_bootstrap.csv", index=False)
    coefficients.to_csv(report / "coefficient_stability.csv", index=False)
    correlations.to_csv(report / "train_feature_correlations.csv")
    config = {
        "phase": "2.3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "block_lengths": list(block_lengths),
        "bootstrap_replicates": n_bootstrap,
        "block_bootstrap_ci_type": "percentile, two-sided, 95%",
        "volatility_regime_threshold_from_train_q75": threshold,
        "test_used": False,
        "interpretation": "Exploratory validation diagnostics, not independent confirmatory inference",
        "limitations": [
            "Fixed-model bootstrap does not adjust for model and feature search",
            "Block bootstrap assumes adequate stability and dependence conditions",
            "Changes made after studying 2019-2021 are model-development choices",
        ],
    }
    write_json(report / "config.json", config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2.3 validation robustness analysis")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    config = run(args.root, n_bootstrap=args.bootstrap)
    print("Phase 2.3 complete: block lengths", config["block_lengths"])
    print("See reports/phase2_3/. Held-out test was not read.")


if __name__ == "__main__":
    main()
