"""Small tabular MLP: training-era chronological CV and multi-seed evaluation."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .ml import probability_metrics, xy
from .neural_utils import chronological_folds, feature_scaler, set_seed, verify_train_validation

DEFAULT_ARCHITECTURES = {"small": (16, 8), "medium": (32, 16)}
DEFAULT_WEIGHT_DECAYS = (0.001, 0.01)
DEFAULT_SEEDS = (11, 42, 73)


class MLP(nn.Module):
    """Output a single raw logit per observation (NO sigmoid in forward)."""
    def __init__(self, input_dim: int = 17, hidden: tuple[int, int] = (32, 16)):
        super().__init__()
        if input_dim < 1 or len(hidden) != 2 or min(hidden) < 1:
            raise ValueError("Invalid MLP architecture")
        h1, h2 = hidden
        self.network = nn.Sequential(
            nn.Linear(input_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def fit_mlp(
    X_train: pd.DataFrame, y_train: pd.Series, *,
    hidden: tuple[int, int] = (32, 16), weight_decay: float = 0.01,
    seed: int = 42, epochs: int = 60, batch_size: int = 64, learning_rate: float = 0.001,
) -> tuple[MLP, StandardScaler, list[float]]:
    """Fit one fresh MLP and its fold-private scaler; no validation-based stopping."""
    if epochs < 1 or batch_size < 1 or learning_rate <= 0 or weight_decay < 0:
        raise ValueError("Invalid optimizer/training configuration")
    if len(y_train) != len(X_train) or set(y_train.unique()) != {0, 1}:
        raise ValueError("Training observations must be aligned and contain both classes")
    set_seed(seed)
    scaler = feature_scaler(X_train)
    scaled = scaler.transform(X_train).astype(np.float32)
    labels = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
    dataset = TensorDataset(torch.from_numpy(scaled), torch.from_numpy(labels))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = MLP(input_dim=X_train.shape[1], hidden=hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        for features, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(features)
        history.append(total / len(dataset))
    return model, scaler, history


def predict_mlp(model: MLP, scaler: StandardScaler, X: pd.DataFrame) -> np.ndarray:
    if list(X.columns) != list(scaler.feature_names_in_):
        raise ValueError("MLP inference feature names/order differ from training")
    inputs = torch.from_numpy(scaler.transform(X).astype(np.float32))
    model.eval()
    with torch.inference_mode():
        probabilities = torch.sigmoid(model(inputs)).squeeze(1)
    return probabilities.numpy()


def cross_validate_mlp(
    train: pd.DataFrame, *,
    architectures: dict[str, tuple[int, int]] = DEFAULT_ARCHITECTURES,
    weight_decays: Sequence[float] = DEFAULT_WEIGHT_DECAYS,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    n_splits: int = 4, test_size: int = 252, gap: int = 2,
    epochs: int = 60, batch_size: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Tune architecture + weight decay by mean chronological CV Log Loss."""
    if not architectures or not weight_decays or not seeds:
        raise ValueError("Empty MLP CV search space")
    X, y = xy(train)
    records = []
    for fold, fit_idx, val_idx in chronological_folds(train, n_splits=n_splits,
                                                       test_size=test_size, gap=gap):
        X_fit, y_fit = X.iloc[fit_idx], y.iloc[fit_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        for name, hidden in architectures.items():
            for wd in weight_decays:
                for seed in seeds:
                    model, scaler, _ = fit_mlp(X_fit, y_fit, hidden=hidden,
                                               weight_decay=float(wd), seed=int(seed),
                                               epochs=epochs, batch_size=batch_size)
                    pred = predict_mlp(model, scaler, X_val)
                    records.append({
                        "fold": fold, "architecture": name, "hidden_1": hidden[0],
                        "hidden_2": hidden[1], "weight_decay": float(wd), "seed": int(seed),
                        "train_end": train.index[fit_idx[-1]],
                        "validation_start": train.index[val_idx[0]],
                        **probability_metrics(y_val, pred),
                    })
    folds = pd.DataFrame(records)
    summary = folds.groupby(["architecture", "hidden_1", "hidden_2", "weight_decay"], as_index=False).agg(
        cv_log_loss=("log_loss", "mean"), cv_log_loss_std=("log_loss", "std"),
        cv_brier_score=("brier_score", "mean"), cv_roc_auc=("roc_auc", "mean"),
    )
    winner = summary.sort_values(["cv_log_loss", "hidden_1", "hidden_2", "weight_decay"]).iloc[0]
    selected = {"architecture": str(winner["architecture"]),
                "hidden": [int(winner["hidden_1"]), int(winner["hidden_2"])],
                "weight_decay": float(winner["weight_decay"])}
    return folds, summary, selected


def fit_and_evaluate_mlp(train: pd.DataFrame, validation: pd.DataFrame, *,
                         selected: dict, seed: int = 42, epochs: int = 60,
                         batch_size: int = 64):
    verify_train_validation(train, validation)
    X_fit, y_fit = xy(train)
    X_val, y_val = xy(validation)
    model, scaler, losses = fit_mlp(
        X_fit, y_fit, hidden=tuple(selected["hidden"]),
        weight_decay=float(selected["weight_decay"]), seed=seed,
        epochs=epochs, batch_size=batch_size,
    )
    predictions = pd.DataFrame({"actual_label": y_val,
                                "p_mlp": predict_mlp(model, scaler, X_val)}, index=validation.index)
    train_pred = predict_mlp(model, scaler, X_fit)
    split_metrics = pd.DataFrame([
        {"split": "train", "model": "MLP", **probability_metrics(y_fit, train_pred)},
        {"split": "validation", "model": "MLP", **probability_metrics(y_val, predictions["p_mlp"])},
    ])
    return model, scaler, losses, predictions, split_metrics
