"""Phase 3.2 CLI: LSTM sequence CV with purged labels and feature-only history."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from .features import FEATURE_COLUMNS
from .lstm_model import (
    DEFAULT_HIDDEN_SIZES, DEFAULT_SEEDS, DEFAULT_SEQUENCE_LENGTHS,
    cross_validate_lstm, fit_and_evaluate_lstm,
)
from .neural_utils import probability_table, verify_previous_predictions
from .pipeline import write_json


def run(
    root: Path, *, sequence_lengths=DEFAULT_SEQUENCE_LENGTHS,
    hidden_sizes=DEFAULT_HIDDEN_SIZES, seeds=DEFAULT_SEEDS,
    n_splits: int = 4, test_size: int = 252, gap: int = 2,
    epochs: int = 60, batch_size: int = 64, refit_seed: int = 42,
) -> dict:
    root = Path(root).resolve()
    processed = root / "data" / "processed"
    report = root / "reports" / "phase3_2"
    # Do not load test features *beyond* the last validation signal: feature
    # history is sliced before any sequence construction.
    train = pd.read_parquet(processed / "spy_train.parquet")
    validation = pd.read_parquet(processed / "spy_validation.parquet")
    all_features = pd.read_parquet(processed / "spy_features_through_validation.parquet")
    if all_features.index.max() > validation.index.max():
        raise ValueError("Feature history unexpectedly includes post-validation dates")
    history = all_features.loc[:validation.index.max(), list(FEATURE_COLUMNS)].copy()
    previous = pd.read_csv(
        root / "reports" / "phase3_1" / "validation_predictions.csv",
        parse_dates=["signal_date"],
    ).set_index("signal_date")
    verify_previous_predictions(previous, validation)
    if "p_mlp" not in previous:
        raise ValueError("Run Phase 3.1 before Phase 3.2")
    folds, summary, selected, common_train = cross_validate_lstm(
        train, history, sequence_lengths=sequence_lengths,
        hidden_sizes=hidden_sizes, seeds=seeds,
        n_splits=n_splits, test_size=test_size, gap=gap,
        epochs=epochs, batch_size=batch_size,
    )
    model, scaler, losses, pred, split_metrics = fit_and_evaluate_lstm(
        train, validation, history, best=selected, seed=refit_seed,
        epochs=epochs, batch_size=batch_size, candidate_train=common_train,
    )
    combined = previous.copy()
    combined["p_lstm"] = pred["p_lstm"]
    scores = probability_table(validation, {
        "Constant Probability": combined["p_baseline"],
        "Logistic Regression": combined["p_logistic"],
        "XGBoost": combined["p_xgboost"],
        "MLP": combined["p_mlp"],
        "LSTM": combined["p_lstm"],
    })
    report.mkdir(parents=True, exist_ok=True)
    folds.to_csv(report / "cv_folds_seeds.csv", index=False)
    summary.sort_values("cv_log_loss").to_csv(report / "cv_summary.csv", index=False)
    scores.to_csv(report / "validation_metrics.csv", index=False)
    split_metrics.to_csv(report / "train_vs_validation_metrics.csv", index=False)
    combined.to_csv(report / "validation_predictions.csv", index_label="signal_date")
    pd.DataFrame({"epoch": range(1, epochs + 1), "train_loss": losses}).to_csv(
        report / "training_losses.csv", index=False)
    torch.save(model.state_dict(), report / "lstm_state_dict.pt")
    joblib.dump(scaler, report / "lstm_scaler.joblib")
    config = {
        "phase": "3.2", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": version("torch"),
        "sequence_lengths": list(sequence_lengths), "hidden_sizes": list(hidden_sizes),
        "seeds": list(seeds), "selected": selected,
        "common_training_observations": len(common_train),
        "final_refit_seed": refit_seed, "epochs": epochs,
        "batch_size": batch_size, "learning_rate": 0.001,
        "weight_decay": 0.01, "gradient_clipping_norm": 1.0,
        "feature_columns": list(FEATURE_COLUMNS), "cv_n_splits": n_splits,
        "cv_test_size": test_size, "cv_gap": gap,
        "selection_metric": "training-era mean CV log_loss over folds and seeds",
        "test_used": False, "validation_is_model_development_data": True,
        "note": "Feature-only historical context across purged boundaries; same signal folds for all sequence lengths; fixed epochs",
    }
    write_json(report / "config.json", config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3.2 PyTorch LSTM CV")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--quick", action="store_true", help="Smoke test only: 2 folds, 2 epochs, one seed; NOT final research")
    args = parser.parse_args()
    options = dict(n_splits=2, test_size=60, epochs=2, seeds=(42,)) if args.quick else {}
    config = run(args.root, **options)
    print("Phase 3.2 complete. Selected:", config["selected"])
    print("Held-out test not read. Reports in reports/phase3_2/.")


if __name__ == "__main__":
    main()
