"""Phase 2.2/2.3 synthetic offline tests: no downloads or real SPY observations."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import log_loss

from quant_trading.features import FEATURE_COLUMNS
from quant_trading.ml import probability_metrics
from quant_trading.xgboost_model import tune_xgb, fit_and_evaluate_xgb, make_xgb
from quant_trading.robustness import (
    observation_log_loss, moving_block_bootstrap, calibration_table,
    yearly_metrics, volatility_regime_metrics, paired_losses, rolling_log_loss,
)


def synthetic_frame(n=680, seed=305):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    features = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    p = 1 / (1 + np.exp(-(0.25 * features[:, 0] - .18 * features[:, 5])))
    frame = pd.DataFrame(features, index=dates, columns=list(FEATURE_COLUMNS))
    frame["label"] = rng.binomial(1, p)
    frame["entry_date"] = pd.Series(dates, index=dates).shift(-1)
    frame["exit_date"] = pd.Series(dates, index=dates).shift(-2)
    frame["future_return"] = np.where(frame["label"] == 1, 0.005, -0.005)
    return frame.iloc[:-2].copy()


def small_grid():
    return {"max_depth": [2], "min_child_weight": [5], "n_estimators": [12, 20]}


def make_predictions(val):
    rng = np.random.default_rng(128)
    p = pd.DataFrame(index=val.index)
    p["actual_label"] = val["label"]
    p["p_baseline"] = 0.51
    p["p_logistic"] = np.clip(0.5 + rng.normal(0, 0.04, len(val)), 0.01, 0.99)
    p["p_xgboost"] = np.clip(0.5 + rng.normal(0, 0.06, len(val)), 0.01, 0.99)
    return p


def test_xgb_walk_forward_and_refit():
    data = synthetic_frame()
    train, validation = data.iloc[:450], data.iloc[452:]
    folds, summary, best = tune_xgb(
        train, param_grid=small_grid(), n_splits=2, test_size=70, gap=2,
    )
    assert len(folds) == 4
    assert len(summary) == 2
    assert best["n_estimators"] in (12, 20)
    assert best["max_depth"] == 2
    assert (folds["train_end"] < folds["validation_start"]).all()
    model, pred, split_metrics = fit_and_evaluate_xgb(train, validation, best_params=best)
    assert len(pred) == len(validation)
    assert pred.index.equals(validation.index)
    assert pred["p_xgboost"].between(0, 1).all()
    assert model.n_features_in_ == 17
    assert set(split_metrics["split"]) == {"train", "validation"}


def test_xgb_cv_detects_label_leakage():
    frame = synthetic_frame(560).iloc[:440]
    with pytest.raises(ValueError, match="immature training labels"):
        tune_xgb(frame, param_grid=small_grid(), n_splits=2, test_size=70, gap=0)


def test_xgb_rejects_unexpected_hyperparameter():
    with pytest.raises(ValueError, match="Unexpected"):
        make_xgb(foo=3)


def test_pointwise_loss_and_bootstrap():
    y = [0, 1, 0, 1]
    p = [0.1, 0.8, 0.2, 0.75]
    losses = observation_log_loss(y, p)
    assert losses.mean() == pytest.approx(log_loss(y, p))
    with pytest.raises(ValueError, match="Invalid"):
        observation_log_loss(y, [0.1, 0.8, np.nan, .75])
    rng = np.random.default_rng(1)
    differences = rng.normal(.01, .02, 160)
    first = moving_block_bootstrap(differences, block_length=8, n_bootstrap=250, seed=42)
    second = moving_block_bootstrap(differences, block_length=8, n_bootstrap=250, seed=42)
    np.testing.assert_array_equal(first["bootstrap_means"], second["bootstrap_means"])
    assert first["ci_lower"] < first["ci_upper"]
    assert first["mean_difference"] == pytest.approx(differences.mean())
    with pytest.raises(ValueError, match="Block length"):
        moving_block_bootstrap(differences, block_length=100, n_bootstrap=250)


def test_diagnostics_and_train_only_vol_threshold():
    frame = synthetic_frame()
    train, val = frame.iloc[:450], frame.iloc[452:]
    pred = make_predictions(val)
    cal = calibration_table(pred)
    assert len(cal.query("model == 'Constant Probability'")) == 1
    assert (cal["observations"] > 0).all()
    yearly = yearly_metrics(pred)
    assert set(yearly["model"]) == {"Constant Probability", "Logistic Regression", "XGBoost"}
    regime, threshold = volatility_regime_metrics(pred, train, val)
    assert threshold == pytest.approx(train["vol_21d"].quantile(.75))
    assert (regime["observations"] > 0).all()
    rolling = rolling_log_loss(pred, window=20)
    assert rolling.iloc[19].notna().all()
    assert rolling.iloc[18].isna().all()
    losses = paired_losses(pred)
    np.testing.assert_allclose(losses["lr_minus_xgboost"],
                               losses["log_loss_logistic"] - losses["log_loss_xgboost"])


def test_full_cli_chain_skips_heldout_test(tmp_path, monkeypatch):
    from quant_trading.phase2_baselines import run as run_logistic
    from quant_trading.phase2_xgboost import run as run_xgb
    from quant_trading.phase2_robustness import run as run_robustness
    frame = synthetic_frame()
    train, validation = frame.iloc[:450].copy(), frame.iloc[452:].copy()
    reads = []

    def fake_read_parquet(path, **kwargs):
        name = str(path)
        reads.append(name)
        if name.endswith("spy_train.parquet"):
            return train.copy()
        if name.endswith("spy_validation.parquet"):
            return validation.copy()
        raise AssertionError("Held-out spy_test.parquet must NOT be read")

    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)
    cfg_lr = run_logistic(tmp_path, c_grid=(0.01, 0.1), n_splits=2, test_size=70, gap=2)
    cfg_xgb = run_xgb(tmp_path, param_grid=small_grid(), n_splits=2, test_size=70, gap=2)
    cfg_robust = run_robustness(tmp_path, n_bootstrap=150, block_lengths=(5, 10))
    assert cfg_lr["test_used"] is False
    assert cfg_xgb["test_used"] is False
    assert cfg_robust["test_used"] is False
    assert not any("spy_test" in name for name in reads)
    root = tmp_path / "reports"
    for phase, items in {
        "phase2_1": ["validation_predictions.csv", "logistic_regression.joblib"],
        "phase2_2": ["cv_folds.csv", "best_params.json", "xgboost_tuned.json", "validation_predictions.csv"],
        "phase2_3": ["block_bootstrap.csv", "calibration_bins.csv", "yearly_metrics.csv", "coefficient_stability.csv"],
    }.items():
        for item in items:
            assert (root / phase / item).is_file(), (phase, item)
    scores = pd.read_csv(root / "phase2_2" / "validation_metrics.csv")
    assert len(scores) == 3
    assert np.isfinite(scores["log_loss"]).all()
