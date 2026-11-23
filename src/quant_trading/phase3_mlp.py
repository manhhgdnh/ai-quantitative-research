"""Phase 3.1 CLI: training-era MLP CV, refit and development validation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd
import torch

from .ml import probability_metrics
from .mlp_model import (
    DEFAULT_ARCHITECTURES, DEFAULT_SEEDS, DEFAULT_WEIGHT_DECAYS,
    cross_validate_mlp, fit_and_evaluate_mlp,
)
from .neural_utils import probability_table, verify_previous_predictions
from .pipeline import write_json


def run(
    root: Path, *, architectures=DEFAULT_ARCHITECTURES,
    weight_decays=DEFAULT_WEIGHT_DECAYS, seeds=DEFAULT_SEEDS,
    n_splits: int = 4, test_size: int = 252, gap: int = 2,
    epochs: int = 60, batch_size: int = 64, refit_seed: int = 42,
) -> dict:
    root = Path(root).resolve()
    processed = root / "data" / "processed"
    report = root / "reports" / "phase3_1"
    prev_path = root / "reports" / "phase2_2" / "validation_predictions.csv"
    # Intentionally no spy_test.parquet anywhere in the Phase 3 code path.
    train = pd.read_parquet(processed / "spy_train.parquet")
    validation = pd.read_parquet(processed / "spy_validation.parquet")
    previous = pd.read_csv(prev_path, parse_dates=["signal_date"]).set_index("signal_date")
    verify_previous_predictions(previous, validation)
    folds, summary, selected = cross_validate_mlp(
        train, architectures=architectures, weight_decays=weight_decays, seeds=seeds,
        n_splits=n_splits, test_size=test_size, gap=gap,
        epochs=epochs, batch_size=batch_size,
    )
    model, scaler, train_losses, pred, split_metrics = fit_and_evaluate_mlp(
        train, validation, selected=selected, seed=refit_seed,
        epochs=epochs, batch_size=batch_size,
    )
    combined = previous.copy()
    combined["p_mlp"] = pred["p_mlp"]
    scores = probability_table(validation, {
        "Constant Probability": combined["p_baseline"],
        "Logistic Regression": combined["p_logistic"],
        "XGBoost": combined["p_xgboost"],
        "MLP": combined["p_mlp"],
    })
    report.mkdir(parents=True, exist_ok=True)
    folds.to_csv(report / "cv_folds_seeds.csv", index=False)
    summary.sort_values("cv_log_loss").to_csv(report / "cv_summary.csv", index=False)
    scores.to_csv(report / "validation_metrics.csv", index=False)
    split_metrics.to_csv(report / "train_vs_validation_metrics.csv", index=False)
    combined.to_csv(report / "validation_predictions.csv", index_label="signal_date")
    pd.DataFrame({"epoch": range(1, epochs + 1), "train_loss": train_losses}).to_csv(
        report / "training_losses.csv", index=False)
    torch.save(model.state_dict(), report / "mlp_state_dict.pt")
    joblib.dump(scaler, report / "mlp_scaler.joblib")
    config = {
        "phase": "3.1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": version("torch"), "architectures": {k: list(v) for k, v in architectures.items()},
        "weight_decays": list(weight_decays), "seeds": list(seeds),
        "selected": selected, "final_refit_seed": refit_seed,
        "epochs": epochs, "batch_size": batch_size,
        "learning_rate": 0.001, "cv_n_splits": n_splits,
        "cv_test_size": test_size, "cv_gap": gap,
        "selection_metric": "training-era mean CV log_loss over folds and seeds",
        "test_used": False,
        "validation_is_model_development_data": True,
        "note": "Fixed epochs; no validation-based early stopping. Initializers and mini-batches use fixed seeds.",
    }
    write_json(report / "config.json", config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 3.1 PyTorch MLP CV")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--quick", action="store_true", help="Smoke test only: 2 folds, 2 epochs, one seed; NOT final research")
    args = parser.parse_args()
    options = dict(n_splits=2, test_size=60, epochs=2, seeds=(42,)) if args.quick else {}
    config = run(args.root, **options)
    print("Phase 3.1 complete. Selected:", config["selected"])
    print("Held-out test not read. Reports in reports/phase3_1/.")


if __name__ == "__main__":
    main()
