"""Run Phase 2.1 from an already-generated Phase 1 snapshot; do not read the test split."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS
from .ml import DEFAULT_C_GRID, cross_validate_models, fit_and_evaluate


def run(root: Path, *, c_grid=DEFAULT_C_GRID, n_splits=4, test_size=252, gap=2) -> dict:
    root = Path(root).resolve()
    processed = root / "data" / "processed"
    report = root / "reports" / "phase2_1"
    report.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(processed / "spy_train.parquet")
    validation = pd.read_parquet(processed / "spy_validation.parquet")
    per_fold, cv_summary, best_C = cross_validate_models(
        train, c_grid=c_grid, n_splits=n_splits, test_size=test_size, gap=gap
    )
    models, predictions, validation_scores = fit_and_evaluate(train, validation, best_C=best_C)

    per_fold.to_csv(report / "cv_folds.csv", index=False)
    cv_summary.to_csv(report / "cv_summary.csv", index=False)
    validation_scores.to_csv(report / "validation_metrics.csv", index=False)
    predictions.to_csv(report / "validation_predictions.csv", index_label="signal_date")

    model = models["Logistic Regression"]
    coef = pd.DataFrame({"feature": FEATURE_COLUMNS, "coefficient_scaled": model.named_steps["classifier"].coef_[0]})
    coef.reindex(coef["coefficient_scaled"].abs().sort_values(ascending=False).index).to_csv(
        report / "logistic_coefficients.csv", index=False
    )
    for name, fitted in models.items():
        target = "logistic_regression.joblib" if name == "Logistic Regression" else "constant_baseline.joblib"
        joblib.dump(fitted, report / target)
    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "2.1",
        "train_dates": [str(train.index.min().date()), str(train.index.max().date())],
        "validation_dates": [str(validation.index.min().date()), str(validation.index.max().date())],
        "test_used": False,
        "feature_columns": list(FEATURE_COLUMNS),
        "model": "L2 Logistic Regression with training-fold StandardScaler",
        "candidate_C": list(c_grid),
        "chosen_C": best_C,
        "cv_n_splits": n_splits,
        "cv_test_size": test_size,
        "cv_gap": gap,
        "selection_metric": "mean CV log_loss (minimize)",
        "validation_note": "Validation is model comparison data; final test remains untouched",
        "data_limitation": "Adjusted-price features and adjusted-open target proxy may be retrospectively revised",
    }
    (report / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2.1 baseline and Logistic Regression")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    config = run(args.root)
    print("Phase 2.1 complete. Selected Logistic C =", config["chosen_C"])
    print("See reports/phase2_1/. Final test was not read.")


if __name__ == "__main__":
    main()
