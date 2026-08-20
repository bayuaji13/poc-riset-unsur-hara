"""Tiny patch-transformer benchmark for joint N/P/K spectral regression."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RepeatedKFold, train_test_split

from .benchmark import regression_metrics, summarize_metrics
from .config import BUDGETS, DEFAULT_TARGETS, MAX_BUDGET, RANDOM_STATE
from .dataset import SpectralDataset, load_processed_dataset
from .preprocessing import SNVTransformer


@dataclass(frozen=True, slots=True)
class AttentionConfig:
    patch_size: int = 16
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.15
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    max_epochs: int = 150
    patience: int = 18
    validation_fraction: float = 0.15


class TinySpectralTransformer(nn.Module):
    """Patch spectra, learn wavelength interactions, and jointly regress N/P/K."""

    def __init__(self, input_length: int = 1701, n_targets: int = 3, config: AttentionConfig | None = None):
        super().__init__()
        self.config = config or AttentionConfig()
        self.input_length = input_length
        self.n_patches = int(np.ceil(input_length / self.config.patch_size))
        self.patch_embedding = nn.Conv1d(
            1,
            self.config.d_model,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        self.position = nn.Parameter(torch.zeros(1, self.n_patches + 1, self.config.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=self.config.d_model,
            nhead=self.config.n_heads,
            dim_feedforward=self.config.dim_feedforward,
            dropout=self.config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(self.config.d_model)
        self.head = nn.Linear(self.config.d_model, n_targets)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        spectra = spectra.unsqueeze(1)
        padding = self.n_patches * self.config.patch_size - spectra.shape[-1]
        if padding:
            spectra = F.pad(spectra, (0, padding), mode="replicate")
        tokens = self.patch_embedding(spectra).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = tokens + self.position[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass(slots=True)
class TrainedAttention:
    model: TinySpectralTransformer
    y_mean: np.ndarray
    y_std: np.ndarray
    best_epoch: int
    validation_loss: float
    device: str

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        values = SNVTransformer().fit_transform(X).astype(np.float32)
        loader = DataLoader(torch.from_numpy(values), batch_size=batch_size, shuffle=False)
        predictions: list[np.ndarray] = []
        self.model.eval()
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            for batch in loader:
                predictions.append(self.model(batch.to(device)).cpu().numpy())
        standardized = np.vstack(predictions)
        transformed = standardized * self.y_std + self.y_mean
        return np.expm1(transformed)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle, generator=generator)


def train_attention(
    X: np.ndarray,
    y: np.ndarray,
    *,
    config: AttentionConfig | None = None,
    device: str = "auto",
    seed: int = RANDOM_STATE,
) -> TrainedAttention:
    config = config or AttentionConfig()
    seed_everything(seed)
    selected_device = resolve_device(device)
    X_snv = SNVTransformer().fit_transform(X).astype(np.float32)
    y_log = np.log1p(np.asarray(y, dtype=np.float32))
    y_mean = y_log.mean(axis=0)
    y_std = np.maximum(y_log.std(axis=0), np.finfo(np.float32).eps)
    y_scaled = (y_log - y_mean) / y_std
    indices = np.arange(len(X_snv))
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=config.validation_fraction,
        random_state=seed,
    )
    train_loader = _loader(X_snv[train_indices], y_scaled[train_indices], config.batch_size, True)
    validation_loader = _loader(X_snv[validation_indices], y_scaled[validation_indices], config.batch_size, False)
    model = TinySpectralTransformer(input_length=X.shape[1], n_targets=y.shape[1], config=config).to(selected_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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
                validation_loss = loss_function(model(batch_X.to(selected_device)), batch_y.to(selected_device))
                losses.append(float(validation_loss.cpu()))
        mean_loss = float(np.mean(losses))
        if mean_loss < best_loss - 1e-5:
            best_loss = mean_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("Training self-attention tidak menghasilkan checkpoint valid.")
    model.load_state_dict(best_state)
    model.to(selected_device)
    return TrainedAttention(model, y_mean, y_std, best_epoch, best_loss, str(selected_device))


def benchmark_attention(
    dataset: SpectralDataset,
    budget: int,
    *,
    config: AttentionConfig | None = None,
    outer_repeats: int = 2,
    device: str = "auto",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = config or AttentionConfig()
    subset = dataset.subset(budget)
    X = subset.spectra
    target_codes = [target.code for target in DEFAULT_TARGETS]
    y = subset.metadata[target_codes].to_numpy(dtype=float)
    outer = RepeatedKFold(n_splits=5, n_repeats=outer_repeats, random_state=RANDOM_STATE + budget)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for split_index, (train_index, test_index) in enumerate(outer.split(X), start=1):
        repeat = (split_index - 1) // 5 + 1
        fold = (split_index - 1) % 5 + 1
        trained = train_attention(
            X[train_index],
            y[train_index],
            config=config,
            device=device,
            seed=RANDOM_STATE + budget + split_index,
        )
        predicted = trained.predict(X[test_index], batch_size=config.batch_size * 2)
        for target_index, target in enumerate(DEFAULT_TARGETS):
            baseline = np.full(test_index.size, np.median(y[train_index, target_index]), dtype=float)
            model_metrics = regression_metrics(y[test_index, target_index], predicted[:, target_index])
            baseline_metrics = regression_metrics(y[test_index, target_index], baseline)
            metric_rows.append(
                {
                    "target": target.code,
                    "budget": budget,
                    "repeat": repeat,
                    "fold": fold,
                    "model": "Patch transformer",
                    "preprocessing": "SNV + patches(16)",
                    "best_epoch": trained.best_epoch,
                    "validation_loss": trained.validation_loss,
                    "parameters": trained.model.parameter_count,
                    "device": trained.device,
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
                        "target": target.code,
                        "budget": budget,
                        "repeat": repeat,
                        "fold": fold,
                        "sample_id": subset.metadata.iloc[position]["sample_id"],
                        "observed": observed,
                        "predicted": estimate,
                        "baseline_predicted": baseline_value,
                        "residual": observed - estimate,
                        "model": "Patch transformer",
                        "preprocessing": "SNV + patches(16)",
                    }
                )
        print(
            f"Attention n={budget} repeat={repeat} fold={fold} epoch={trained.best_epoch} device={trained.device}",
            flush=True,
        )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def fit_final_attention(
    dataset: SpectralDataset,
    artifact_dir: Path,
    *,
    config: AttentionConfig,
    device: str,
) -> TrainedAttention:
    full = dataset.subset(MAX_BUDGET)
    y = full.metadata[[target.code for target in DEFAULT_TARGETS]].to_numpy(dtype=float)
    trained = train_attention(full.spectra, y, config=config, device=device, seed=RANDOM_STATE)
    checkpoint = {
        "state_dict": {name: value.detach().cpu() for name, value in trained.model.state_dict().items()},
        "config": asdict(config),
        "input_length": full.n_features,
        "target_codes": [target.code for target in DEFAULT_TARGETS],
        "y_mean": trained.y_mean,
        "y_std": trained.y_std,
        "best_epoch": trained.best_epoch,
        "validation_loss": trained.validation_loss,
        "parameter_count": trained.model.parameter_count,
    }
    torch.save(checkpoint, artifact_dir / "attention_model.pt")
    return trained


def run_attention_benchmark(
    project_root: Path,
    *,
    budgets: tuple[int, ...] = BUDGETS,
    outer_repeats: int = 2,
    device: str = "auto",
    config: AttentionConfig | None = None,
    fit_final: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or AttentionConfig()
    started = perf_counter()
    dataset = load_processed_dataset(project_root)
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for budget in budgets:
        predictions, metrics = benchmark_attention(
            dataset,
            budget,
            config=config,
            outer_repeats=outer_repeats,
            device=device,
        )
        prediction_frames.append(predictions)
        metric_frames.append(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = summarize_metrics(metrics)
    artifact_dir = project_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(artifact_dir / "attention_predictions.csv", index=False)
    metrics.to_csv(artifact_dir / "attention_fold_metrics.csv", index=False)
    summary.to_csv(artifact_dir / "attention_summary.csv", index=False)
    final_details: dict[str, object] = {}
    if fit_final:
        trained = fit_final_attention(dataset, artifact_dir, config=config, device=device)
        final_details = {
            "final_best_epoch": trained.best_epoch,
            "final_validation_loss": trained.validation_loss,
            "parameter_count": trained.model.parameter_count,
            "device": trained.device,
        }
    manifest = {
        "budgets": list(budgets),
        "outer_folds": 5,
        "outer_repeats": outer_repeats,
        "random_state": RANDOM_STATE,
        "config": asdict(config),
        "elapsed_seconds": perf_counter() - started,
        **final_details,
    }
    (artifact_dir / "attention_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return predictions, metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cpu", "cuda"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()
    config = AttentionConfig(max_epochs=args.epochs)
    _, _, summary = run_attention_benchmark(
        args.project_root.resolve(),
        outer_repeats=args.repeats,
        device=args.device,
        config=config,
        fit_final=not args.skip_final,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
