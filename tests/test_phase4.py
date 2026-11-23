"""Offline Phase 4 tests; no live SPY download or held-out test access."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quant_trading.backtest import (
    MODEL_COLUMNS, align_validation_predictions, run_backtest,
    validate_backtest_frame,
)
from quant_trading.performance import compute_performance, drawdown_series
from quant_trading.strategy_research import (
    bootstrap_paired_excess_returns, choose_threshold, cost_sensitivity,
    threshold_sweep, volatility_regime_analysis,
)


def example_frame():
    signals = pd.to_datetime(["2019-01-02", "2019-01-03", "2019-01-04"])
    return pd.DataFrame({
        "entry_date": pd.to_datetime(["2019-01-03", "2019-01-04", "2019-01-07"]),
        "exit_date": pd.to_datetime(["2019-01-04", "2019-01-07", "2019-01-08"]),
        "future_return": [.01, -.02, .03],
        "label": [1, 0, 1],
    }, index=pd.DatetimeIndex(signals, name="signal_date"))


def synthetic_frame(n=90):
    all_dates = pd.bdate_range("2019-01-02", periods=n + 2)
    rng = np.random.default_rng(4)
    r = rng.normal(.0003, .015, n)
    frame = pd.DataFrame({
        "entry_date": all_dates[1:-1],
        "exit_date": all_dates[2:],
        "future_return": r,
        "label": (r > 0).astype(int),
        "vol_21d": .15 + .03 * np.cos(np.arange(n)/7),
    }, index=pd.DatetimeIndex(all_dates[:-2], name="signal_date"))
    p = pd.Series(np.clip(.52 + .09 * np.sin(np.arange(n)/4), .01, .99), index=frame.index)
    return frame, p


def test_three_period_execution_exact_cost_and_equity():
    frame = example_frame()
    p = pd.Series([.60, .60, .40], index=frame.index)
    bt = run_backtest(frame, p, threshold=.55, cost_bps=5)
    assert bt.index.equals(pd.DatetimeIndex(pd.to_datetime(frame.exit_date), name="exit_date"))
    assert list(bt.position) == [1, 1, 0]
    assert list(bt.turnover) == [1, 0, 1]
    np.testing.assert_allclose(bt.net_return.to_numpy(), [.009495, -.02, -.0005])
    np.testing.assert_allclose(bt.equity.iloc[-1], 9888.1044745)
    scores = compute_performance(bt)
    np.testing.assert_allclose(scores["cumulative_return"], -.01118955255)
    assert scores["one_sided_trades"] == 2
    np.testing.assert_allclose(scores["estimated_cost_dollars"], 5 + (10000*1.009495*.98)*.0005)
    dd = drawdown_series(bt)
    assert np.isclose(scores["max_drawdown"], -dd.min())
    assert scores["max_drawdown"] > 0


def test_buy_hold_pays_one_initial_fee_without_daily_roundtrips():
    frame = example_frame()
    bh = run_backtest(frame, cost_bps=5)
    assert list(bh.position) == [1, 1, 1]
    assert list(bh.turnover) == [1, 0, 0]
    assert bh.cost_fraction.iloc[0] == .0005
    assert bh.cost_fraction.iloc[1] == 0
    assert bh.cost_fraction.iloc[2] == 0
    np.testing.assert_allclose(bh.equity.iloc[-1], 10000*.9995*1.01*.98*1.03)


def test_first_day_drawdown_includes_initial_capital():
    frame = example_frame()
    p = pd.Series([.99, .01, .01], index=frame.index)
    frame.loc[frame.index[0], "future_return"] = -.04
    frame.loc[frame.index[0], "label"] = 0
    bt = run_backtest(frame, p, threshold=.5, cost_bps=0)
    assert drawdown_series(bt).iloc[0] == pytest.approx(-.04)


def test_bad_chronology_missing_days_labels_and_prediction_alignment():
    frame, p = synthetic_frame()
    validate_backtest_frame(frame)
    with pytest.raises(ValueError, match="Prediction Series"):
        run_backtest(frame, p.iloc[::-1])
    bad = frame.copy()
    bad.loc[bad.index[5], "exit_date"] = bad.loc[bad.index[5], "entry_date"]
    with pytest.raises(ValueError, match="chronology"):
        validate_backtest_frame(bad)
    bad = frame.drop(index=frame.index[8])
    with pytest.raises(ValueError, match="Nonconsecutive"):
        validate_backtest_frame(bad)
    bad = frame.copy()
    bad.loc[bad.index[0], "label"] = 1 - int(bad.iloc[0].label)
    with pytest.raises(ValueError, match="mismatch"):
        validate_backtest_frame(bad)
    bad = frame.copy()
    bad.loc[bad.index[0], "future_return"] = np.nan
    with pytest.raises(ValueError, match="Invalid SPY"):
        validate_backtest_frame(bad)
    with pytest.raises(ValueError, match="Invalid threshold"):
        run_backtest(frame, p, threshold=1.1)


def test_alignment_requires_all_five_exact_valid_probabilities():
    frame, p = synthetic_frame()
    predictions = pd.DataFrame({
        "actual_label": frame.label,
        **{col: p for col in MODEL_COLUMNS.values()},
    }, index=frame.index)
    aligned = align_validation_predictions(frame, predictions)
    assert aligned.index.equals(frame.index)
    assert len(aligned) == len(frame)
    bad = predictions.copy()
    bad.loc[bad.index[3], "p_mlp"] = -2
    with pytest.raises(ValueError, match="probabilities"):
        align_validation_predictions(frame, bad)
    with pytest.raises(ValueError, match="alignment"):
        align_validation_predictions(frame, predictions.iloc[::-1])


def test_threshold_selection_fallback_is_flagged_not_fake_success():
    frame, p = synthetic_frame()
    grid = threshold_sweep(frame, p)
    assert len(grid) == 3
    best = choose_threshold(grid, min_exposure=0, max_exposure=1)
    assert best["eligibility_met"]
    impossible = choose_threshold(grid, min_exposure=.99, max_exposure=1.)
    assert impossible["eligibility_met"] is False
    assert impossible["threshold"] == .5
    assert "fallback" in impossible["selection_status"]


def test_regime_and_cost_sensitivity_and_paired_bootstrap():
    frame, p = synthetic_frame()
    train = frame.iloc[:30].copy()
    strategy = run_backtest(frame, p, threshold=.55)
    benchmark = run_backtest(frame)
    detailed, summary, cutoff = volatility_regime_analysis(strategy, benchmark, train, frame)
    assert len(detailed) == len(frame)
    assert np.isclose(cutoff, train.vol_21d.quantile(.75))
    assert summary.observations.sum() == len(frame)
    np.testing.assert_allclose(detailed.excess_return,
                               strategy.net_return.to_numpy() - benchmark.net_return.to_numpy())
    costs = cost_sensitivity(frame, p, threshold=.55)
    assert list(costs.cost_bps) == [0, 5, 10]
    assert costs.cumulative_return.is_monotonic_decreasing
    boots = bootstrap_paired_excess_returns(strategy, benchmark, n_bootstrap=100)
    assert len(boots) == 3
    assert (boots.ci_lower <= boots.ci_upper).all()
    assert np.isfinite(boots.mean_daily_excess_return).all()
    assert strategy.turnover.sum() == detailed.turnover.sum()


def test_phase4_cli_integration_never_opens_test(tmp_path, monkeypatch):
    from quant_trading.phase4_backtest import run as run_backtest_cli
    from quant_trading.phase4_robustness import run as run_robustness_cli
    val, p = synthetic_frame()
    train = val.iloc[:30].copy()
    predictions = pd.DataFrame({
        "actual_label": val.label,
        **{col: (p + .015*i).clip(0, 1) for i, col in enumerate(MODEL_COLUMNS.values())},
    }, index=val.index)
    prior = tmp_path / "reports" / "phase3_2"
    prior.mkdir(parents=True)
    predictions.to_csv(prior / "validation_predictions.csv", index_label="signal_date")
    calls = []
    def fake_read(path, **kwargs):
        calls.append(str(path))
        if str(path).endswith("spy_validation.parquet"):
            return val.copy()
        if str(path).endswith("spy_train.parquet"):
            return train.copy()
        raise AssertionError("Forbidden dataset read: " + str(path))
    monkeypatch.setattr(pd, "read_parquet", fake_read)
    bt_cfg = run_backtest_cli(tmp_path)
    rob_cfg = run_robustness_cli(tmp_path, n_bootstrap=100, block_lengths=(5, 10, 20))
    assert bt_cfg["test_used"] is False
    assert rob_cfg["test_data_opened"] is False
    assert not any("spy_test" in call for call in calls)
    assert len(pd.read_csv(tmp_path / "reports/phase4_2/performance_metrics.csv")) == 6
    assert (tmp_path / "reports/phase4_2/equity_curves.csv").exists()
    assert (tmp_path / "reports/phase4_4/selected_strategy_backtest.csv").exists()
    assert (tmp_path / "reports/phase4_4/block_bootstrap.csv").exists()
    loaded = json.loads((tmp_path / "reports/phase4_4/strategy_config.json").read_text())
    assert loaded["test_used_for_selection"] is False
