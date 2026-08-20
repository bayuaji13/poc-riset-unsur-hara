"""Small-data nonlinear model zoo for MIR-to-NPK regression."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from time import perf_counter
import warnings

from cubist import Cubist
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold, RepeatedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR

from .benchmark import regression_metrics
from .config import BUDGETS, DEFAULT_TARGETS, MAX_BUDGET, RANDOM_STATE
from .dataset import SpectralDataset, load_processed_dataset
from .preprocessing import SNVTransformer


MODEL_FAMILIES = (
    "PLS–RBF-SVR",
    "PLS–Cubist",
    "PLS–Extra Trees",
    "PCA–Gaussian Process",
    "PLS–Cascade Forest",
    "Tiny 1D CNN",
)

MODEL_SLUGS = {
    "PLS–RBF-SVR": "svr",
    "PLS–Cubist": "cubist",
    "PLS–Extra Trees": "extra_trees",
    "PCA–Gaussian Process": "gpr",
    "PLS–Cascade Forest": "cascade_forest",
}


class PLSScoreTransformer(TransformerMixin, BaseEstimator):
    """Use supervised PLS x-scores as compact, leakage-safe spectral features."""

    def __init__(self, n_components: int = 10):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y: np.ndarray):
        components = min(self.n_components, X.shape[1], X.shape[0] - 1)
        self.model_ = PLSRegression(n_components=components, scale=False, max_iter=1000)
        self.model_.fit(X, np.asarray(y).reshape(-1))
        self.n_components_ = components
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.transform(X), dtype=np.float64)


class QuietCubist(Cubist):
    """Silence Cubist's internal synthetic-feature-name warning for array pipelines."""

    def predict(self, X):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names")
            return super().predict(X)


class CascadeForestRegressor(RegressorMixin, BaseEstimator):
    """Compact gcForest-style cascade with OOF feature augmentation."""

    def __init__(
        self,
        n_layers: int = 2,
        n_estimators: int = 80,
        min_samples_leaf: int = 3,
        cv: int = 3,
        random_state: int = RANDOM_STATE,
        n_jobs: int = -1,
    ):
        self.n_layers = n_layers
        self.n_estimators = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _estimators(self, layer: int):
        seed = self.random_state + layer * 10
        return (
            RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=0.8,
                n_jobs=self.n_jobs,
                random_state=seed,
            ),
            ExtraTreesRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.min_samples_leaf,
                max_features=0.8,
                n_jobs=self.n_jobs,
                random_state=seed + 1,
            ),
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        features = np.asarray(X, dtype=np.float32)
        target = np.asarray(y, dtype=float).reshape(-1)
        self.layers_: list[tuple[RegressorMixin, RegressorMixin]] = []
        splitter = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for layer in range(self.n_layers):
            fitted: list[RegressorMixin] = []
            oof_columns: list[np.ndarray] = []
            for estimator in self._estimators(layer):
                oof_columns.append(
                    cross_val_predict(clone(estimator), features, target, cv=splitter, n_jobs=1)
                )
                fitted_estimator = clone(estimator).fit(features, target)
                fitted.append(fitted_estimator)
            self.layers_.append((fitted[0], fitted[1]))
            if layer < self.n_layers - 1:
                features = np.column_stack((features, *oof_columns)).astype(np.float32)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        features = np.asarray(X, dtype=np.float32)
        predictions: list[np.ndarray] = []
        for layer, estimators in enumerate(self.layers_):
            predictions = [np.asarray(estimator.predict(features)) for estimator in estimators]
            if layer < len(self.layers_) - 1:
                features = np.column_stack((features, *predictions)).astype(np.float32)
        return np.mean(predictions, axis=0)


def _spectral_pipeline(reducer: TransformerMixin, model: RegressorMixin) -> TransformedTargetRegressor:
    pipeline = Pipeline(
        [
            ("snv", SNVTransformer()),
            ("wavelength_scale", StandardScaler()),
            ("latent", reducer),
            ("latent_scale", StandardScaler()),
            ("model", model),
        ]
    )
    return TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_model(family: str, *, seed: int, n_jobs: int = -1) -> TransformedTargetRegressor:
    if family == "PLS–RBF-SVR":
        return _spectral_pipeline(PLSScoreTransformer(10), SVR(kernel="rbf", C=10.0, epsilon=0.1))
    if family == "PLS–Cubist":
        return _spectral_pipeline(
            PLSScoreTransformer(10),
            QuietCubist(n_rules=100, n_committees=10, neighbors=5, random_state=seed),
        )
    if family == "PLS–Extra Trees":
        return _spectral_pipeline(
            PLSScoreTransformer(15),
            ExtraTreesRegressor(
                n_estimators=300,
                min_samples_leaf=3,
                max_features=0.8,
                n_jobs=n_jobs,
                random_state=seed,
            ),
        )
    if family == "PCA–Gaussian Process":
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=2.0,
            length_scale_bounds="fixed",
            nu=1.5,
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds="fixed")
        return _spectral_pipeline(
            PCA(n_components=20, svd_solver="randomized", random_state=seed),
            GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=None, random_state=seed),
        )
    if family == "PLS–Cascade Forest":
        return _spectral_pipeline(
            PLSScoreTransformer(10),
            CascadeForestRegressor(random_state=seed, n_jobs=n_jobs),
        )
    raise ValueError(f"Model family tidak dikenal: {family}")


def summarize_suite(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["mae", "rmse", "r2", "rpiq", "baseline_rmse", "rmse_improvement_pct", "fit_seconds"]
    rows: list[dict[str, object]] = []
    for (family, target, budget), group in metrics.groupby(["family", "target", "budget"], sort=True):
        row: dict[str, object] = {
            "family": family,
            "target": target,
            "budget": int(budget),
            "n_folds": len(group),
            "model_bytes_median": float(group["model_bytes"].median()),
        }
        for name in metric_names:
            values = group[name].to_numpy(dtype=float)
            row[f"{name}_median"] = float(np.nanmedian(values))
            row[f"{name}_q025"] = float(np.nanquantile(values, 0.025))
            row[f"{name}_q975"] = float(np.nanquantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def benchmark_family(
    dataset: SpectralDataset,
    family: str,
    budget: int,
    *,
    outer_repeats: int = 1,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = dataset.subset(budget)
    X = subset.spectra
    outer = RepeatedKFold(n_splits=5, n_repeats=outer_repeats, random_state=RANDOM_STATE + budget)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for target in DEFAULT_TARGETS:
        y = subset.metadata[target.code].to_numpy(dtype=float)
        for split_index, (train_index, test_index) in enumerate(outer.split(X), start=1):
            repeat = (split_index - 1) // 5 + 1
            fold = (split_index - 1) % 5 + 1
            model = build_model(family, seed=RANDOM_STATE + budget + split_index, n_jobs=n_jobs)
            started = perf_counter()
            model.fit(X[train_index], y[train_index])
            fit_seconds = perf_counter() - started
            predicted = np.asarray(model.predict(X[test_index])).reshape(-1)
            baseline = np.full(test_index.size, np.median(y[train_index]), dtype=float)
            model_metrics = regression_metrics(y[test_index], predicted)
            baseline_metrics = regression_metrics(y[test_index], baseline)
            model_bytes = len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL))
            metric_rows.append(
                {
                    "family": family,
                    "target": target.code,
                    "budget": budget,
                    "repeat": repeat,
                    "fold": fold,
                    "model": family,
                    "preprocessing": "SNV + latent scores",
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
                y[test_index],
                predicted,
                baseline,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "family": family,
                        "target": target.code,
                        "budget": budget,
                        "repeat": repeat,
                        "fold": fold,
                        "sample_id": subset.metadata.iloc[position]["sample_id"],
                        "observed": observed,
                        "predicted": estimate,
                        "baseline_predicted": baseline_value,
                        "residual": observed - estimate,
                        "model": family,
                        "preprocessing": "SNV + latent scores",
                    }
                )
            print(
                f"{family} target={target.code} n={budget} repeat={repeat} fold={fold} "
                f"fit={fit_seconds:.2f}s",
                flush=True,
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def fit_final_models(
    dataset: SpectralDataset,
    artifact_dir: Path,
    *,
    families: tuple[str, ...],
    n_jobs: int = -1,
) -> pd.DataFrame:
    full = dataset.subset(MAX_BUDGET)
    rows: list[dict[str, object]] = []
    for family in families:
        if family == "Tiny 1D CNN":
            continue
        for target in DEFAULT_TARGETS:
            model = build_model(family, seed=RANDOM_STATE, n_jobs=n_jobs)
            started = perf_counter()
            model.fit(full.spectra, full.metadata[target.code].to_numpy(dtype=float))
            fit_seconds = perf_counter() - started
            path = artifact_dir / f"zoo_{MODEL_SLUGS[family]}_{target.code.lower()}.joblib"
            joblib.dump(model, path)
            rows.append(
                {
                    "family": family,
                    "target": target.code,
                    "fit_seconds": fit_seconds,
                    "model_bytes": path.stat().st_size,
                    "artifact": path.name,
                }
            )
            print(f"Final {family} target={target.code} fit={fit_seconds:.2f}s", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(artifact_dir / "zoo_final_models.csv", index=False)
    return result


def run_model_zoo(
    project_root: Path,
    *,
    budgets: tuple[int, ...] = BUDGETS,
    families: tuple[str, ...] = MODEL_FAMILIES,
    outer_repeats: int = 1,
    n_jobs: int = -1,
    fit_final: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = perf_counter()
    dataset = load_processed_dataset(project_root)
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for budget in budgets:
        for family in families:
            if family == "Tiny 1D CNN":
                from .cnn import benchmark_cnn

                predictions, metrics = benchmark_cnn(dataset, budget, outer_repeats=outer_repeats)
            else:
                predictions, metrics = benchmark_family(
                    dataset,
                    family,
                    budget,
                    outer_repeats=outer_repeats,
                    n_jobs=n_jobs,
                )
            prediction_frames.append(predictions)
            metric_frames.append(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = summarize_suite(metrics)
    artifact_dir = project_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(artifact_dir / "zoo_predictions.csv", index=False)
    metrics.to_csv(artifact_dir / "zoo_fold_metrics.csv", index=False)
    summary.to_csv(artifact_dir / "zoo_summary.csv", index=False)
    manifest = {
        "families": list(families),
        "budgets": list(budgets),
        "outer_folds": 5,
        "outer_repeats": outer_repeats,
        "random_state": RANDOM_STATE,
    }
    if fit_final:
        final_models = fit_final_models(dataset, artifact_dir, families=families, n_jobs=n_jobs)
        manifest["final_models"] = final_models.to_dict(orient="records")
    if fit_final and "Tiny 1D CNN" in families:
        from .cnn import fit_final_cnn

        trained_cnn = fit_final_cnn(dataset, artifact_dir)
        manifest["cnn_final"] = {
            "best_epoch": trained_cnn.best_epoch,
            "validation_loss": trained_cnn.validation_loss,
            "parameter_count": trained_cnn.model.parameter_count,
            "device": trained_cnn.device,
        }
    manifest["elapsed_seconds"] = perf_counter() - started
    (artifact_dir / "zoo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return predictions, metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    parser.add_argument("--families", nargs="+", choices=MODEL_FAMILIES, default=list(MODEL_FAMILIES))
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()
    _, _, summary = run_model_zoo(
        args.project_root.resolve(),
        budgets=tuple(args.budgets),
        families=tuple(args.families),
        outer_repeats=args.repeats,
        n_jobs=args.jobs,
        fit_final=not args.skip_final,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
