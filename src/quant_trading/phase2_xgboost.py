"""Execute Phase 2.2 from Phase 1 processed data and Phase 2.1 reports."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from .features import FEATURE_COLUMNS
from .xgboost_model import (
    DEFAULT_PARAM_GRID, feature_gain, fit_and_evaluate_xgb, tune_xgb,
)
from .pipeline import write_json


def run(
    root: Path, *, param_grid=DEFAULT_PARAM_GRID,
    n_splits: int = 4, test_size: int = 252, gap: int = 2,
) -> dict:
    root = Path(root).resolve()
    processed = root / "data" / "processed"
    prev = root / "reports" / "phase2_1"
    report = root / "reports" / "phase2_2"
    report.mkdir(parents=True, exist_ok=True)

    # Do not even open the held-out test dataset.
    train = pd.read_parquet(processed / "spy_train.parquet")
    validation = pd.read_parquet(processed / "spy_validation.parquet")
    baseline_preds = pd.read_csv(
        prev / "validation_predictions.csv", parse_dates=["signal_date"],
    ).set_index("signal_date")
    if not baseline_preds.index.equals(validation.index):
        raise ValueError("Phase 2.1 predictions are not aligned with Phase 2.2 validation dates")
    if not baseline_preds.index.is_unique:
        raise ValueError("Duplicate predictions")
    if not baseline_preds["actual_label"].equals(validation["label"]):
        raise ValueError("Phase 2.1 and 2.2 label mismatch")
    if not {"p_baseline", "p_logistic"}.issubset(baseline_preds.columns):
        raise ValueError("Run Phase 2.1 before Phase 2.2")

    folds, summary, best_params = tune_xgb(
        train, param_grid=param_grid, n_splits=n_splits,
        test_size=test_size, gap=gap,
    )
    model, xgb_preds, split_metrics = fit_and_evaluate_xgb(
        train, validation, best_params=best_params,
    )
    from .ml import probability_metrics
    combined = baseline_preds.copy()
    combined["p_xgboost"] = xgb_preds["p_xgboost"]
    val_scores = pd.DataFrame([
        {"model": name, **probability_metrics(combined["actual_label"], combined[col])}
        for name, col in (
            ("Constant Probability", "p_baseline"),
            ("Logistic Regression", "p_logistic"),
            ("XGBoost", "p_xgboost"),
        )
    ]).sort_values("log_loss").reset_index(drop=True)

    folds.to_csv(report / "cv_folds.csv", index=False)
    summary.sort_values("cv_log_loss").to_csv(report / "cv_summary.csv", index=False)
    val_scores.to_csv(report / "validation_metrics.csv", index=False)
    split_metrics.to_csv(report / "train_vs_validation_metrics.csv", index=False)
    combined.to_csv(report / "validation_predictions.csv", index_label="signal_date")
    feature_gain(model, list(FEATURE_COLUMNS)).to_csv(report / "feature_gain.csv", index=False)
    model.save_model(str(report / "xgboost_tuned.json"))
    write_json(report / "best_params.json", best_params)
    config = {
        "phase": "2.2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "xgboost_version": version("xgboost"),
        "param_grid": {k: list(v) for k, v in param_grid.items()},
        "best_params": best_params,
        "cv_splits": n_splits,
        "cv_test_size": test_size,
        "cv_gap": gap,
        "selection_metric": "mean CV log_loss (minimize)",
        "validation_is_model_development_data": True,
        "test_used": False,
        "note": "No early stopping; selected n_estimators inside training-era CV.",
    }
    write_json(report / "config.json", config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2.2 XGBoost training-era CV")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = run(args.root)
    print("Phase 2.2 complete: ", config["best_params"])
    print("See reports/phase2_2/. Held-out test was not read.")


if __name__ == "__main__":
    main()
