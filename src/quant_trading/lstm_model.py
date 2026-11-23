"""Sequence-to-one LSTM on strictly past/current features; mature-label CV."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import FEATURE_COLUMNS
from .ml import probability_metrics, xy
from .neural_utils import chronological_folds, feature_scaler, set_seed, verify_train_validation

DEFAULT_SEQUENCE_LENGTHS = (10, 20, 40)
DEFAULT_HIDDEN_SIZES = (8, 16)
DEFAULT_SEEDS = (11, 42, 73)


class LSTMClassifier(nn.Module):
    """One-layer, one-direction LSTM; returns raw logit for each full sequence."""
    def __init__(self, input_dim: int = 17, hidden_size: int = 16):
        super().__init__()
        if input_dim < 1 or hidden_size < 1:
            raise ValueError("Invalid LSTM dimensions")
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected [batch, sequence, features]")
        _, (hidden, _) = self.lstm(x)
        return self.output(hidden[-1])


def eligible_training_signals(
    train: pd.DataFrame, feature_history: pd.DataFrame, *, max_sequence_length: int = 40,
) -> pd.DataFrame:
    """Keep SAME train signal dates for every candidate sequence length.

    Eligibility is based solely on historical feature availability, never labels
    in validation. Raw feature history includes feature-only purged gap sessions.
    """
    _validate_history(feature_history)
    if max_sequence_length < 1:
        raise ValueError("Invalid maximum sequence length")
    positions = feature_history.index.get_indexer(train.index)
    if (positions < 0).any():
        raise ValueError("Training signals missing from feature history")
    finite_rows = np.isfinite(feature_history.to_numpy(dtype=float)).all(axis=1)
    n = len(feature_history)
    cumulative_bad = np.concatenate(([0], np.cumsum(~finite_rows)))
    eligible = np.array([
        (pos >= max_sequence_length - 1 and
         cumulative_bad[pos + 1] == cumulative_bad[pos + 1 - max_sequence_length])
        for pos in positions
    ], dtype=bool)
    result = train.iloc[np.flatnonzero(eligible)].copy()
    if len(result) < 10:
        raise ValueError("Too few eligible training signals for LSTM")
    return result


def _validate_history(history: pd.DataFrame) -> None:
    if not isinstance(history.index, pd.DatetimeIndex) or history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("Feature history must have sorted unique dates")
    if list(history.columns) != list(FEATURE_COLUMNS):
        raise ValueError("Feature history names or order mismatch")


def build_sequences(
    history: pd.DataFrame, signal_dates: pd.DatetimeIndex | Sequence,
    scaler: StandardScaler, *, sequence_length: int,
) -> np.ndarray:
    """At signal time t, use only [t-L+1, ..., t] with no future features."""
    _validate_history(history)
    dates = pd.DatetimeIndex(signal_dates)
    if dates.empty or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("Sequence signals must be ordered, unique, nonempty")
    if sequence_length < 1:
        raise ValueError("Sequence length must be positive")
    positions = history.index.get_indexer(dates)
    if np.any(positions < sequence_length - 1):
        raise ValueError("Insufficient past or missing signal in feature history")
    # Restrict transform to history up through last signal. A fitted train-only
    # scaler may transform current/earlier validation features, never fit on them.
    values = history.iloc[:positions[-1] + 1].to_numpy(dtype=float)
    # Warm-up NaNs before the first valid window are permitted; every actual
    # model input window must be finite (checked immediately below).
    windows = np.stack([values[i - sequence_length + 1:i + 1] for i in positions])
    if not np.isfinite(windows).all():
        raise ValueError("Non-finite historical sequence features")
    if list(scaler.feature_names_in_) != list(FEATURE_COLUMNS):
        raise ValueError("Scaler feature names/order mismatch")
    flattened = windows.reshape(-1, len(FEATURE_COLUMNS))
    transformed = scaler.transform(pd.DataFrame(flattened, columns=list(FEATURE_COLUMNS)))
    return transformed.reshape(len(dates), sequence_length, len(FEATURE_COLUMNS)).astype(np.float32)


def fit_lstm(
    X_sequences: np.ndarray, labels: pd.Series | np.ndarray, *,
    hidden_size: int = 16, seed: int = 42, epochs: int = 60,
    batch_size: int = 64, learning_rate: float = .001,
    weight_decay: float = .01, max_grad_norm: float = 1.0,
) -> tuple[LSTMClassifier, list[float]]:
    """Train a fresh LSTM; no outer-validation based early stopping."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0 or max_grad_norm <= 0:
        raise ValueError("Invalid LSTM training settings")
    samples = np.asarray(X_sequences, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32).reshape(-1, 1)
    if samples.ndim != 3 or len(samples) != len(y) or len(samples) < 2:
        raise ValueError("Invalid LSTM input/label dimensions")
    if not np.isfinite(samples).all() or not np.isin(y, [0, 1]).all() or np.unique(y).size != 2:
        raise ValueError("LSTM needs finite features and both binary classes")
    set_seed(seed)
    dataset = TensorDataset(torch.from_numpy(samples), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = LSTMClassifier(input_dim=samples.shape[-1], hidden_size=hidden_size)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            total += float(loss.item()) * len(batch_x)
        losses.append(total / len(dataset))
    return model, losses


def predict_lstm(model: LSTMClassifier, X_sequences: np.ndarray,
                 *, batch_size: int = 512) -> np.ndarray:
    """Batch inference to avoid large sequence-memory allocations."""
    if batch_size < 1:
        raise ValueError("Invalid inference batch size")
    samples = np.asarray(X_sequences, dtype=np.float32)
    if samples.ndim != 3:
        raise ValueError("Expected three-dimensional sequence input")
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(samples), batch_size):
            x = torch.from_numpy(samples[start:start + batch_size])
            output.append(torch.sigmoid(model(x)).squeeze(1).numpy())
    return np.concatenate(output) if output else np.empty(0, dtype=float)


def cross_validate_lstm(
    train: pd.DataFrame, history: pd.DataFrame, *,
    sequence_lengths: Sequence[int] = DEFAULT_SEQUENCE_LENGTHS,
    hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_splits: int = 4, test_size: int = 252, gap: int = 2,
    epochs: int = 60, batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """CV same eligible signal dates for ALL sequence lengths & hidden sizes."""
    if not sequence_lengths or not hidden_sizes or not seeds:
        raise ValueError("Empty LSTM CV search space")
    candidate_train = eligible_training_signals(
        train, history, max_sequence_length=max(sequence_lengths))
    _, _ = xy(candidate_train)
    records = []
    # One scaler per fold, not one per seed or future validation period.
    for fold, fit_idx, val_idx in chronological_folds(
        candidate_train, n_splits=n_splits, test_size=test_size, gap=gap
    ):
        fold_train = candidate_train.iloc[fit_idx]
        fold_val = candidate_train.iloc[val_idx]
        X_fit, y_fit = xy(fold_train)
        _, y_val = xy(fold_val)
        scaler = feature_scaler(X_fit)
        for length in sequence_lengths:
            fit_sequences = build_sequences(history, fold_train.index, scaler, sequence_length=int(length))
            val_sequences = build_sequences(history, fold_val.index, scaler, sequence_length=int(length))
            for hidden in hidden_sizes:
                for seed in seeds:
                    model, _ = fit_lstm(fit_sequences, y_fit, hidden_size=int(hidden),
                                        seed=int(seed), epochs=epochs, batch_size=batch_size)
                    proba = predict_lstm(model, val_sequences)
                    records.append({
                        "fold": fold, "sequence_length": int(length), "hidden_size": int(hidden),
                        "seed": int(seed), "train_end": fold_train.index[-1],
                        "validation_start": fold_val.index[0],
                        **probability_metrics(y_val, proba),
                    })
    folds = pd.DataFrame(records)
    summary = folds.groupby(["sequence_length", "hidden_size"], as_index=False).agg(
        cv_log_loss=("log_loss", "mean"), cv_log_loss_std=("log_loss", "std"),
        cv_brier_score=("brier_score", "mean"), cv_roc_auc=("roc_auc", "mean"),
    )
    winner = summary.sort_values(["cv_log_loss", "hidden_size", "sequence_length"]).iloc[0]
    best = {"sequence_length": int(winner["sequence_length"]),
            "hidden_size": int(winner["hidden_size"])}
    return folds, summary, best, candidate_train


def fit_and_evaluate_lstm(
    train: pd.DataFrame, validation: pd.DataFrame, history: pd.DataFrame,
    *, best: dict, seed: int = 42, epochs: int = 60, batch_size: int = 64,
    candidate_train: pd.DataFrame | None = None,
):
    """Refit on same common eligible train signals; evaluate ALL validation dates."""
    verify_train_validation(train, validation)
    _validate_history(history)
    if candidate_train is None:
        candidate_train = eligible_training_signals(train, history,
            max_sequence_length=max(DEFAULT_SEQUENCE_LENGTHS))
    if not candidate_train.index.isin(train.index).all():
        raise ValueError("Candidate train includes non-training signals")
    X_fit, y_fit = xy(candidate_train)
    _, y_val = xy(validation)
    scaler = feature_scaler(X_fit)
    fit_sequences = build_sequences(history, candidate_train.index, scaler,
                                    sequence_length=int(best["sequence_length"]))
    val_sequences = build_sequences(history, validation.index, scaler,
                                    sequence_length=int(best["sequence_length"]))
    model, losses = fit_lstm(fit_sequences, y_fit, hidden_size=int(best["hidden_size"]),
                             seed=seed, epochs=epochs, batch_size=batch_size)
    p_val = predict_lstm(model, val_sequences)
    p_train = predict_lstm(model, fit_sequences)
    predictions = pd.DataFrame({"actual_label": y_val, "p_lstm": p_val}, index=validation.index)
    scores = pd.DataFrame([
        {"split": "train", "model": "LSTM", **probability_metrics(y_fit, p_train)},
        {"split": "validation", "model": "LSTM", **probability_metrics(y_val, p_val)},
    ])
    return model, scaler, losses, predictions, scores
