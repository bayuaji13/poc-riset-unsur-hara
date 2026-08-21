"""Leave-one-county-out Local-first versus OSSL-first transfer benchmark."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .attention import resolve_device, seed_everything
from .benchmark import regression_metrics
from .config import DEFAULT_TARGETS, RANDOM_STATE
from .dataset import SpectralDataset, load_dataset_directory
from .hybrid import HybridConfig, HybridSpectralRegressor
from .local import load_local_dataset
from .preprocessing import SNVTransformer


@dataclass(frozen=True, slots=True)
class TransferConfig:
    ossl_subset_size: int = 5000
    pretrain_epochs: int = 80
    finetune_epochs: int = 100
    patience: int = 15
    batch_size: int = 64
    pretrain_lr: float = 1e-3
    local_lr: float = 1e-3
    weight_decay: float = 1e-3
    validation_fraction: float = 0.15


@dataclass(slots=True)
class TrainedTransfer:
    model: HybridSpectralRegressor
    y_mean: np.ndarray
    y_std: np.ndarray
    best_epoch: int
    validation_loss: float
    device: str

    def predict(self, X: np.ndarray) -> np.ndarray:
        values = SNVTransformer().fit_transform(X).astype(np.float32)
        device = next(self.model.parameters()).device
        self.model.eval()
        with torch.inference_mode():
            result = self.model(torch.from_numpy(values).to(device)).cpu().numpy()
        return np.expm1(result * self.y_std + self.y_mean)


def select_nearest_ossl(ossl: SpectralDataset, local_train_spectra: np.ndarray, count: int) -> tuple[SpectralDataset, np.ndarray]:
    """Nearest OSSL records in row-wise SNV spectral space; target values are never used."""
    if ossl.n_samples < count:
        raise ValueError(f"Pool OSSL hanya {ossl.n_samples}; diperlukan sedikitnya {count} untuk transfer benchmark.")
    source = SNVTransformer().fit_transform(ossl.spectra)
    query = SNVTransformer().fit_transform(local_train_spectra)
    distances = np.empty(len(source), dtype=float)
    chunk = 512
    for start in range(0, len(source), chunk):
        values = source[start : start + chunk]
        distances[start : start + len(values)] = np.sqrt(((values[:, None, :] - query[None, :, :]) ** 2).mean(axis=2)).min(axis=1)
    indices = np.argsort(distances, kind="stable")[:count]
    metadata = ossl.metadata.iloc[indices].copy().reset_index(drop=True)
    metadata["similarity_distance"] = distances[indices]
    return SpectralDataset(metadata, ossl.spectra[indices], ossl.grid.copy()), distances[indices]


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=shuffle, generator=torch.Generator().manual_seed(seed))


def train_transfer_model(X: np.ndarray, y: np.ndarray, *, config: TransferConfig, max_epochs: int, learning_rate: float, seed: int, device: str, model: HybridSpectralRegressor | None = None) -> TrainedTransfer:
    seed_everything(seed)
    selected_device = resolve_device(device)
    X_snv = SNVTransformer().fit_transform(X).astype(np.float32)
    y_log = np.log1p(np.asarray(y, dtype=np.float32))
    y_mean, y_std = y_log.mean(axis=0), np.maximum(y_log.std(axis=0), np.finfo(np.float32).eps)
    scaled = (y_log - y_mean) / y_std
    if len(X) < 8:
        train_index, validation_index = np.arange(len(X)), np.arange(len(X))
    else:
        train_index, validation_index = train_test_split(np.arange(len(X)), test_size=config.validation_fraction, random_state=seed)
    trained_model = model or HybridSpectralRegressor(n_targets=y.shape[1])
    trained_model = trained_model.to(selected_device)
    optimizer = torch.optim.AdamW((p for p in trained_model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=config.weight_decay)
    loss_fn = nn.MSELoss()
    best_loss, best_epoch, stale, best_state = float("inf"), 0, 0, None
    train_loader = _loader(X_snv[train_index], scaled[train_index], config.batch_size, True, seed)
    validation_loader = _loader(X_snv[validation_index], scaled[validation_index], config.batch_size, False, seed)
    for epoch in range(1, max_epochs + 1):
        trained_model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss_fn(trained_model(batch_X.to(selected_device)), batch_y.to(selected_device)).backward()
            nn.utils.clip_grad_norm_(trained_model.parameters(), 1.0)
            optimizer.step()
        trained_model.eval()
        with torch.inference_mode():
            losses = []
            for batch_X, batch_y in validation_loader:
                estimate = trained_model(batch_X.to(selected_device))
                losses.append(float(loss_fn(estimate, batch_y.to(selected_device)).cpu()))
            loss = float(np.mean(losses))
        if loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in trained_model.state_dict().items()}
        else:
            stale += 1
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("Training transfer tidak menghasilkan checkpoint.")
    trained_model.load_state_dict(best_state)
    return TrainedTransfer(trained_model.to(selected_device), y_mean, y_std, best_epoch, best_loss, str(selected_device))


def _metric_rows(variant: str, fold: int, test_meta: pd.DataFrame, y_train: np.ndarray, y_test: np.ndarray, predicted: np.ndarray, details: dict[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    predictions, metrics = [], []
    for index, target in enumerate(DEFAULT_TARGETS):
        baseline = np.full(len(y_test), np.median(y_train[:, index]))
        values, baseline_values = regression_metrics(y_test[:, index], predicted[:, index]), regression_metrics(y_test[:, index], baseline)
        bias = float(np.mean(predicted[:, index] - y_test[:, index]))
        metrics.append({"variant": variant, "fold": fold, "held_out_county": test_meta["group_id"].iloc[0], "target": target.code, **details, **values, "bias": bias, "baseline_rmse": baseline_values["rmse"], "rmse_improvement_pct": 100 * (baseline_values["rmse"] - values["rmse"]) / baseline_values["rmse"]})
        predictions.extend({"variant": variant, "fold": fold, "held_out_county": test_meta["group_id"].iloc[pos], "target": target.code, "sample_id": test_meta.iloc[pos]["sample_id"], "observed": observed, "predicted": estimate, "baseline_predicted": baseline[pos], "residual": observed-estimate} for pos, (observed, estimate) in enumerate(zip(y_test[:, index], predicted[:, index], strict=True)))
    return predictions, metrics


def summarize_transfer(metrics: pd.DataFrame) -> pd.DataFrame:
    numeric = ["mae", "rmse", "r2", "rpiq", "bias", "baseline_rmse", "rmse_improvement_pct"]
    rows = []
    for (variant, target), group in metrics.groupby(["variant", "target"], sort=True):
        row: dict[str, object] = {"variant": variant, "target": target, "n_folds": len(group)}
        row.update({f"{name}_median": float(group[name].median()) for name in numeric})
        rows.append(row)
    return pd.DataFrame(rows)


def run_transfer_benchmark(project_root: Path, *, workbook: Path, spectra_root: Path, ossl: SpectralDataset | None = None, config: TransferConfig | None = None, device: str = "auto") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or TransferConfig()
    local, report = load_local_dataset(workbook, spectra_root)
    ossl = ossl or load_dataset_directory(project_root / "data" / "ossl_transfer")
    targets = [target.code for target in DEFAULT_TARGETS]
    y = local.metadata[targets].to_numpy(dtype=float)
    groups = local.metadata["group_id"].to_numpy()
    all_predictions: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []
    selections: dict[str, object] = {}
    started = perf_counter()
    for fold, (train_index, test_index) in enumerate(LeaveOneGroupOut().split(local.spectra, y, groups), start=1):
        X_train, X_test, y_train, y_test = local.spectra[train_index], local.spectra[test_index], y[train_index], y[test_index]
        selected, distances = select_nearest_ossl(ossl, X_train, config.ossl_subset_size)
        selections[str(fold)] = {"held_out_county": str(groups[test_index][0]), "source_ids": selected.metadata["sample_id"].tolist(), "distance_min": float(distances.min()), "distance_median": float(np.median(distances)), "distance_max": float(distances.max())}
        pretrained = train_transfer_model(selected.spectra, selected.metadata[targets].to_numpy(float), config=config, max_epochs=config.pretrain_epochs, learning_rate=config.pretrain_lr, seed=RANDOM_STATE + fold, device=device)
        local_first = train_transfer_model(X_train, y_train, config=config, max_epochs=config.finetune_epochs, learning_rate=config.local_lr, seed=RANDOM_STATE + 100 + fold, device=device)
        zero_shot = pretrained.predict(X_test)
        adapted_model = deepcopy(pretrained.model).cpu()
        adapted_model.head = nn.Sequential(nn.Dropout(adapted_model.config.dropout), nn.Linear(adapted_model.config.channels[-1], len(targets)))
        adapted_model.freeze_encoder()
        fine_tuned = train_transfer_model(X_train, y_train, config=config, max_epochs=config.finetune_epochs, learning_rate=config.local_lr, seed=RANDOM_STATE + 200 + fold, device=device, model=adapted_model)
        for variant, fitted, estimate in (("local_first", local_first, local_first.predict(X_test)), ("ossl_zero_shot", pretrained, zero_shot), ("ossl_first_head_finetuned", fine_tuned, fine_tuned.predict(X_test))):
            prediction_rows, metric_rows = _metric_rows(variant, fold, local.metadata.iloc[test_index].reset_index(drop=True), y_train, y_test, estimate, {"best_epoch": fitted.best_epoch, "validation_loss": fitted.validation_loss, "parameters": fitted.model.parameter_count, "ossl_subset_size": selected.n_samples})
            all_predictions.extend(prediction_rows); all_metrics.extend(metric_rows)
    predictions, metrics = pd.DataFrame(all_predictions), pd.DataFrame(all_metrics)
    summary = summarize_transfer(metrics)
    artifacts = project_root / "artifacts"; artifacts.mkdir(exist_ok=True)
    predictions.to_csv(artifacts / "transfer_predictions.csv", index=False); metrics.to_csv(artifacts / "transfer_fold_metrics.csv", index=False); summary.to_csv(artifacts / "transfer_summary.csv", index=False)
    (artifacts / "transfer_manifest.json").write_text(json.dumps({"local_workbook": str(workbook), "spectra_root": str(spectra_root), "matched_local_spectra": report.matched, "label_only_rows": report.label_only, "ossl_pool_size": ossl.n_samples, "config": asdict(config), "selections": selections, "elapsed_seconds": perf_counter()-started, "provisional_lab_method_warning": "Local P/K methods are unknown; OSSL Olsen P and NH4OAc K are provisional."}, indent=2), encoding="utf-8")
    return predictions, metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--workbook", type=Path, default=Path("NPK_Filled_Soil_Data_v2.xlsx"))
    parser.add_argument("--spectra-root", type=Path, default=Path("../unsur-hara/sample-bu-Yayuk-0303/Bu Yayuk-0303"))
    parser.add_argument("--ossl-size", type=int, default=5000)
    parser.add_argument("--ossl-root", type=Path, default=Path("data/ossl_transfer"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    config = TransferConfig(ossl_subset_size=args.ossl_size, pretrain_epochs=args.epochs)
    ossl = load_dataset_directory(args.project_root.resolve() / args.ossl_root)
    _, _, summary = run_transfer_benchmark(args.project_root.resolve(), workbook=args.workbook, spectra_root=args.spectra_root, ossl=ossl, config=config, device=args.device)
    print(summary.to_string(index=False))
