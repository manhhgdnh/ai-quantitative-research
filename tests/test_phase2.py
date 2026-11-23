"""Phase 2.1 tests: no exchange network or market data required."""

import numpy as np
import pandas as pd
import pytest

from quant_trading.features import FEATURE_COLUMNS
from quant_trading.ml import cross_validate_models, fit_and_evaluate, make_logistic, probability_metrics


def synthetic_supervised(n: int = 850) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = pd.bdate_range("2010-01-04", periods=n)
    X = rng.normal(0, 1, (n, len(FEATURE_COLUMNS)))
    latent = 0.30 * X[:, 0] - 0.15 * X[:, 2]
    probability = 1 / (1 + np.exp(-latent))
    labels = rng.binomial(1, probability)
    data = pd.DataFrame(X, index=dates, columns=FEATURE_COLUMNS)
    data["label"] = labels
    data["exit_date"] = pd.Series(dates, index=dates).shift(-2)
    data["entry_date"] = pd.Series(dates, index=dates).shift(-1)
    return data.iloc[:-2].copy()


def test_probabilistic_metrics_are_valid():
    results = probability_metrics([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7])
    assert results["roc_auc"] == pytest.approx(1)
    assert results["brier_score"] == pytest.approx((.04 + .04 + .09 + .09) / 4)
    assert 0 < results["log_loss"] < 1


def test_cv_and_logistic_model_selection():
    train = synthetic_supervised().iloc[:650].copy()
    folds, summary, chosen = cross_validate_models(
        train, c_grid=(0.01, .1, 1.0), n_splits=3, test_size=100, gap=2
    )
    assert len(folds) == 3 * 4  # dummy + 3 logistic candidates, per fold
    assert 3 <= len(summary) <= 4
    assert chosen in (0.01, .1, 1.0)
    assert (folds["train_end"] < folds["validation_start"]).all()
    assert np.isfinite(folds["log_loss"]).all()


def test_fold_must_have_mature_labels():
    train = synthetic_supervised().iloc[:650].copy()
    with pytest.raises(ValueError, match="immature training labels"):
        cross_validate_models(train, n_splits=3, test_size=100, gap=0)


def test_validation_scaling_and_data_isolation():
    data = synthetic_supervised()
    train = data.iloc[:650].copy()
    validation = data.iloc[652:750].copy()
    models, predictions, scores = fit_and_evaluate(train, validation, best_C=0.1)
    assert len(predictions) == len(validation)
    assert predictions[["p_baseline", "p_logistic"]].ge(0).all().all()
    assert predictions[["p_baseline", "p_logistic"]].le(1).all().all()
    assert len(scores) == 2
    scaler = models["Logistic Regression"].named_steps["scaler"]
    expected_mean = train[list(FEATURE_COLUMNS)].mean().to_numpy()
    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-12)

    modified_validation = validation.copy()
    modified_validation.loc[:, list(FEATURE_COLUMNS)] += 100
    modified_validation.loc[:, "label"] = 1 - modified_validation["label"]
    new_models, _, _ = fit_and_evaluate(train, modified_validation, best_C=0.1)
    np.testing.assert_allclose(
        new_models["Logistic Regression"].named_steps["scaler"].mean_,
        expected_mean,
    )


def test_runner_does_not_read_heldout_test(tmp_path, monkeypatch):
    from quant_trading.phase2_baselines import run

    data = synthetic_supervised()
    training = data.iloc[:650].copy()
    validation = data.iloc[652:750].copy()
    accessed = []

    def fake_read_parquet(path, **kwargs):
        accessed.append(str(path))
        if str(path).endswith("spy_train.parquet"):
            return training
        if str(path).endswith("spy_validation.parquet"):
            return validation
        raise AssertionError("The final test must not be accessed")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    config = run(tmp_path, c_grid=(.01, .1), n_splits=2, test_size=100, gap=2)
    assert config["test_used"] is False
    assert len(accessed) == 2
    assert not any("spy_test" in f for f in accessed)
    report = tmp_path / "reports" / "phase2_1"
    assert (report / "validation_metrics.csv").exists()
    assert (report / "logistic_regression.joblib").exists()
