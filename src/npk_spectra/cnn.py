"""Compact 1D CNN with local spectral inductive bias for joint N/P/K regression."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RepeatedKFold, train_test_split

from .attention import resolve_device, seed_everything
from .benchmark import regression_metrics
from .config import DEFAULT_TARGETS, MAX_BUDGET, RANDOM_STATE
from .dataset import SpectralDataset
from .preprocessing import SNVTransformer


@dataclass(frozen=True, slots=True)
class CNNConfig:
    channels: tuple[int, ...] = (16, 32, 64, 64)
    kernels: tuple[int, ...] = (9, 7, 5, 5)
    pooled_length: int = 4
    hidden_size: int = 64
    dropout: float = 0.15
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 100
    patience: int = 15
    validation_fraction: float = 0.15


class TinySpectralCNN(nn.Module):
    """Learn local absorption motifs before joint N/P/K regression."""

    def __init__(self, n_targets: int = 3, config: CNNConfig | None = None):
        super().__init__()
        self.config = config or CNNConfig()
        blocks: list[nn.Module] = []
        input_channels = 1
        for output_channels, kernel in zip(self.config.channels, self.config.kernels, strict=True):
            blocks.extend(
                (
                    nn.Conv1d(input_channels, output_channels, kernel_size=kernel, stride=2, padding=kernel // 2),
                    nn.BatchNorm1d(output_channels),
                    nn.GELU(),
                )
            )
            input_channels = output_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(self.config.pooled_length)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.config.dropout),
            nn.Linear(input_channels * self.config.pooled_length, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_size, n_targets),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        features = self.features(spectra.unsqueeze(1))
        remainder = features.shape[-1] % self.config.pooled_length
        if remainder:
            features = F.pad(features, (0, self.config.pooled_length - remainder), mode="replicate")
        return self.head(self.pool(features))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(slots=True)
class TrainedCNN:
    model: TinySpectralCNN
    y_mean: np.ndarray
    y_std: np.ndarray
    best_epoch: int
    validation_loss: float
    device: str

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        values = SNVTransformer().fit_transform(X).astype(np.float32)
        loader = DataLoader(torch.from_numpy(values), batch_size=batch_size, shuffle=False)
        output: list[np.ndarray] = []
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.inference_mode():
            for batch in loader:
                output.append(self.model(batch.to(device)).cpu().numpy())
        standardized = np.vstack(output)
        return np.expm1(standardized * self.y_std + self.y_mean)


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle, generator=generator)


def train_cnn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: CNNConfig | None = None,
    device: str = "auto",
    seed: int = RANDOM_STATE,
) -> TrainedCNN:
    config = config or CNNConfig()
    seed_everything(seed)
    selected_device = resolve_device(device)
    X_snv = SNVTransformer().fit_transform(X).astype(np.float32)
    y_log = np.log1p(np.asarray(y, dtype=np.float32))
    y_mean = y_log.mean(axis=0)
    y_std = np.maximum(y_log.std(axis=0), np.finfo(np.float32).eps)
    y_scaled = (y_log - y_mean) / y_std
    train_indices, validation_indices = train_test_split(
        np.arange(len(X_snv)),
        test_size=config.validation_fraction,
        random_state=seed,
    )
    train_loader = _loader(X_snv[train_indices], y_scaled[train_indices], config.batch_size, True)
    validation_loader = _loader(X_snv[validation_indices], y_scaled[validation_indices], config.batch_size, False)
    model = TinySpectralCNN(n_targets=y.shape[1], config=config).to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loss_function = nn.MSELoss()
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(batch_X.to(selected_device)), batch_y.to(selected_device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        losses: list[float] = []
        with torch.inference_mode():
            for batch_X, batch_y in validation_loader:
                loss = loss_function(model(batch_X.to(selected_device)), batch_y.to(selected_device))
                losses.append(float(loss.cpu()))
        validation_loss = float(np.mean(losses))
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Training tiny 1D CNN tidak menghasilkan checkpoint valid.")
    model.load_state_dict(best_state)
    model.to(selected_device)
    return TrainedCNN(model, y_mean, y_std, best_epoch, best_loss, str(selected_device))


def benchmark_cnn(
    dataset: SpectralDataset,
    budget: int,
    *,
    config: CNNConfig | None = None,
    outer_repeats: int = 1,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or CNNConfig()
    subset = dataset.subset(budget)
    X = subset.spectra
    y = subset.metadata[[target.code for target in DEFAULT_TARGETS]].to_numpy(dtype=float)
    outer = RepeatedKFold(n_splits=5, n_repeats=outer_repeats, random_state=RANDOM_STATE + budget)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for split_index, (train_index, test_index) in enumerate(outer.split(X), start=1):
        repeat = (split_index - 1) // 5 + 1
        fold = (split_index - 1) % 5 + 1
        timer = perf_counter()
        trained = train_cnn(
            X[train_index],
            y[train_index],
            config=config,
            device=device,
            seed=RANDOM_STATE + budget + split_index,
        )
        fit_seconds = perf_counter() - timer
        predicted = trained.predict(X[test_index], batch_size=config.batch_size * 2)
        model_bytes = sum(value.numel() * value.element_size() for value in trained.model.state_dict().values())
        for target_index, target in enumerate(DEFAULT_TARGETS):
            baseline = np.full(test_index.size, np.median(y[train_index, target_index]), dtype=float)
            model_metrics = regression_metrics(y[test_index, target_index], predicted[:, target_index])
            baseline_metrics = regression_metrics(y[test_index, target_index], baseline)
            metric_rows.append(
                {
                    "family": "Tiny 1D CNN",
                    "target": target.code,
                    "budget": budget,
                    "repeat": repeat,
                    "fold": fold,
                    "model": "Tiny 1D CNN",
                    "preprocessing": "SNV + local convolutions",
                    "best_epoch": trained.best_epoch,
                    "validation_loss": trained.validation_loss,
                    "parameters": trained.model.parameter_count,
                    "device": trained.device,
                    "fit_seconds": fit_seconds,
                    "model_bytes": model_bytes,
                    **model_metrics,
                    "baseline_rmse": baseline_metrics["rmse"],
                    "rmse_improvement_pct": 100
                    * (baseline_metrics["rmse"] - model_metrics["rmse"])
                    / baseline_metrics["rmse"],
                }
            )
            for position, observed, estimate, baseline_value in zip(
                test_index,
                y[test_index, target_index],
                predicted[:, target_index],
                baseline,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "family": "Tiny 1D CNN",
                        "target": target.code,
                        "budget": budget,
                        "repeat": repeat,
                        "fold": fold,
                        "sample_id": subset.metadata.iloc[position]["sample_id"],
                        "observed": observed,
                        "predicted": estimate,
                        "baseline_predicted": baseline_value,
                        "residual": observed - estimate,
                        "model": "Tiny 1D CNN",
                        "preprocessing": "SNV + local convolutions",
                    }
                )
        print(
            f"Tiny 1D CNN n={budget} repeat={repeat} fold={fold} epoch={trained.best_epoch} "
            f"device={trained.device} fit={fit_seconds:.2f}s",
            flush=True,
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def fit_final_cnn(
    dataset: SpectralDataset,
    artifact_dir: Path,
    *,
    config: CNNConfig | None = None,
    device: str = "auto",
) -> TrainedCNN:
    config = config or CNNConfig()
    full = dataset.subset(MAX_BUDGET)
    y = full.metadata[[target.code for target in DEFAULT_TARGETS]].to_numpy(dtype=float)
    trained = train_cnn(full.spectra, y, config=config, device=device)
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in trained.model.state_dict().items()},
        "config": asdict(config),
        "target_codes": [target.code for target in DEFAULT_TARGETS],
        "y_mean": trained.y_mean.tolist(),
        "y_std": trained.y_std.tolist(),
        "best_epoch": trained.best_epoch,
        "validation_loss": trained.validation_loss,
        "parameter_count": trained.model.parameter_count,
    }
    torch.save(checkpoint, artifact_dir / "cnn_model.pt")
    return trained
