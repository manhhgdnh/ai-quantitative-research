"""Offline Phase 3 tests: temporal integrity, tensor shapes, CV and CLI integration."""
import numpy as np
import pandas as pd
import pytest
import torch

from quant_trading.features import FEATURE_COLUMNS
from quant_trading.mlp_model import MLP, fit_mlp, predict_mlp, cross_validate_mlp
from quant_trading.lstm_model import (
    LSTMClassifier, build_sequences, cross_validate_lstm,
    eligible_training_signals, fit_and_evaluate_lstm,
)
from quant_trading.neural_utils import chronological_folds, feature_scaler

# Small matrix workloads in CI: prevent thread oversubscription.
torch.set_num_threads(2)


def synthetic_frame(n=490, seed=53):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=n)
    values = rng.normal(size=(n, len(FEATURE_COLUMNS)))
    features = pd.DataFrame(values, index=dates, columns=FEATURE_COLUMNS)
    probability = 1/(1+np.exp(-(0.25*values[:, 0] - .15*values[:, 1])))
    labels = rng.binomial(1, probability)
    frame = features.copy()
    frame["label"] = labels
    frame["entry_date"] = pd.Series(dates, index=dates).shift(-1)
    frame["exit_date"] = pd.Series(dates, index=dates).shift(-2)
    frame["future_return"] = np.where(labels == 1, .01, -.01)
    frame = frame.iloc[:-2].copy()
    return frame, features


def test_network_shapes_and_gradients():
    torch.manual_seed(2)
    mlp = MLP(input_dim=17, hidden=(32, 16))
    assert sum(p.numel() for p in mlp.parameters()) == 1121
    out = mlp(torch.zeros(4, 17))
    assert tuple(out.shape) == (4, 1)
    out.square().mean().backward()
    lstm = LSTMClassifier(input_dim=17, hidden_size=16)
    assert sum(p.numel() for p in lstm.parameters()) == 2257
    output = lstm(torch.randn(4, 20, 17))
    assert tuple(output.shape) == (4, 1)
    output.square().mean().backward()


def test_mlp_cv_and_training_only_scaler():
    frame, _ = synthetic_frame()
    train = frame.iloc[:320]
    folds, summary, chosen = cross_validate_mlp(
        train, architectures={"small": (16, 8)}, weight_decays=(.01, .001),
        seeds=(11, 42), n_splits=2, test_size=40, gap=2, epochs=2,
    )
    assert len(folds) == 8 and len(summary) == 2
    assert chosen["hidden"] == [16, 8]
    assert (folds.train_end < folds.validation_start).all()
    X = train.loc[:, list(FEATURE_COLUMNS)]
    y = train.label
    model, scaler, losses = fit_mlp(X, y, hidden=(16, 8), epochs=2)
    np.testing.assert_allclose(scaler.mean_, X.mean(axis=0))
    p = predict_mlp(model, scaler, X.iloc[:5])
    assert len(p) == 5 and ((p >= 0) & (p <= 1)).all()
    assert len(losses) == 2
    with pytest.raises(ValueError, match="names/order"):
        predict_mlp(model, scaler, X.iloc[:5, ::-1])


def test_sequence_causality_and_window_context():
    frame, features = synthetic_frame()
    train, validation = frame.iloc[:320], frame.iloc[322:]
    candidate = eligible_training_signals(train, features, max_sequence_length=40)
    assert candidate.index.equals(train.index[39:])
    scaler = feature_scaler(train.loc[:, list(FEATURE_COLUMNS)])
    windows = build_sequences(features, validation.index[:3], scaler, sequence_length=40)
    assert windows.shape == (3, 40, 17)
    last_context = scaler.transform(features.loc[[validation.index[0]], list(FEATURE_COLUMNS)])[0]
    np.testing.assert_allclose(windows[0,-1], last_context, rtol=1e-5)
    future_modified = features.copy()
    future_modified.loc[validation.index[3]:, "ret_1d"] += 999
    altered = build_sequences(future_modified, validation.index[:3], scaler, sequence_length=40)
    np.testing.assert_array_equal(windows, altered)
    # Purged gap sessions must remain available as feature context.
    first_validation_position = features.index.get_loc(validation.index[0])
    assert features.index[first_validation_position - 1] not in train.index
    assert np.isfinite(windows).all()
    corrupted = features.copy()
    corrupted.loc[validation.index[0], "ret_1d"] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        build_sequences(corrupted, validation.index[:3], scaler, sequence_length=40)


def test_lstm_cv_shared_dates_and_label_purge():
    frame, features = synthetic_frame()
    train = frame.iloc[:350]
    folds, summary, best, eligible = cross_validate_lstm(
        train, features, sequence_lengths=(10, 20, 40), hidden_sizes=(8,),
        seeds=(42,), n_splits=2, test_size=40, gap=2, epochs=1,
    )
    assert len(folds) == 6 and len(summary) == 3
    assert best["sequence_length"] in (10, 20, 40)
    for fold in (1, 2):
        assert folds.loc[folds.fold == fold, "validation_start"].nunique() == 1
    assert (folds.train_end < folds.validation_start).all()
    assert eligible.index.equals(train.index[39:])
    with pytest.raises(ValueError, match="gap"):
        list(chronological_folds(train, n_splits=2, test_size=40, gap=0))


def test_lstm_fit_eval_dates():
    frame, features = synthetic_frame()
    train, val = frame.iloc[:320], frame.iloc[322:]
    cand = eligible_training_signals(train, features, max_sequence_length=40)
    model, scaler, losses, predictions, metrics = fit_and_evaluate_lstm(
        train, val, features, best={"sequence_length": 20, "hidden_size": 8},
        epochs=2, candidate_train=cand,
    )
    assert len(losses) == 2 and len(predictions) == len(val)
    assert predictions.index.equals(val.index)
    assert predictions.p_lstm.between(0, 1).all()
    assert set(metrics["split"]) == {"train", "validation"}
    with pytest.raises(ValueError, match="mature"):
        bad_train = train.copy()
        bad_train.loc[bad_train.index[-1], "exit_date"] = val.index[0]
        fit_and_evaluate_lstm(bad_train, val, features, best={"sequence_length":20, "hidden_size":8}, epochs=1)


def test_phase3_cli_integration_never_reads_test(tmp_path, monkeypatch):
    from quant_trading.phase3_mlp import run as run_mlp
    from quant_trading.phase3_lstm import run as run_lstm

    frame, features = synthetic_frame()
    train, val = frame.iloc[:320], frame.iloc[322:]
    baseline = pd.DataFrame({
        "actual_label": val.label,
        "p_baseline": .51,
        "p_logistic": .50,
        "p_xgboost": .52,
    }, index=val.index)
    prev = tmp_path / "reports" / "phase2_2"
    prev.mkdir(parents=True)
    baseline.to_csv(prev / "validation_predictions.csv", index_label="signal_date")
    calls = []
    def fake_read(path, **kwargs):
        calls.append(str(path))
        if str(path).endswith("spy_train.parquet"):
            return train.copy()
        if str(path).endswith("spy_validation.parquet"):
            return val.copy()
        if str(path).endswith("spy_features_through_validation.parquet"):
            return features.loc[:val.index.max()].copy()
        raise AssertionError("Forbidden dataset read: " + str(path))

    monkeypatch.setattr(pd, "read_parquet", fake_read)
    cfg_mlp = run_mlp(tmp_path, architectures={"small": (16,8)}, weight_decays=(.01,),
                      seeds=(42,), n_splits=2, test_size=35, epochs=1)
    cfg_lstm = run_lstm(tmp_path, sequence_lengths=(10, 20), hidden_sizes=(8,),
                        seeds=(42,), n_splits=2, test_size=35, epochs=1)
    assert cfg_mlp["test_used"] is False and cfg_lstm["test_used"] is False
    assert not any("spy_test" in c for c in calls)
    result = pd.read_csv(tmp_path / "reports" / "phase3_2" / "validation_predictions.csv")
    assert len(result) == len(val)
    assert {"p_baseline", "p_logistic", "p_xgboost", "p_mlp", "p_lstm"}.issubset(result.columns)
    assert (tmp_path / "reports" / "phase3_2" / "lstm_state_dict.pt").exists()
