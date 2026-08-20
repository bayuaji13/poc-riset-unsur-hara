"""Repeated nested-CV benchmark for the 60→1000 sample learning curve."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import BUDGETS, DEFAULT_TARGETS, MAX_BUDGET, RANDOM_STATE, TargetSpec
from .dataset import SpectralDataset, load_processed_dataset
from .preprocessing import SNVTransformer, SavitzkyGolayDerivative


def build_search(inner_cv: KFold, n_jobs: int = -1) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("spectral", "passthrough"),
            ("scale", StandardScaler()),
            ("reduce", "passthrough"),
            ("model", Ridge()),
        ]
    )
    estimator = TransformedTargetRegressor(
        regressor=pipeline,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )
    grids = [
        {
            "regressor__spectral": [SNVTransformer(), SavitzkyGolayDerivative()],
            "regressor__reduce": ["passthrough"],
            "regressor__model": [PLSRegression(scale=False, max_iter=1000)],
            "regressor__model__n_components": [2, 5, 10],
        },
        {
            "regressor__spectral": [SNVTransformer()],
            "regressor__reduce": [PCA(n_components=0.95, svd_solver="full")],
            "regressor__model": [Ridge()],
            "regressor__model__alpha": [1.0],
        },
    ]
    return GridSearchCV(
        estimator,
        param_grid=grids,
        scoring="neg_root_mean_squared_error",
        cv=inner_cv,
        n_jobs=n_jobs,
        refit=True,
        error_score="raise",
    )


def describe_model(best_params: dict[str, object]) -> tuple[str, str]:
    spectral = best_params["regressor__spectral"]
    model = best_params["regressor__model"]
    preprocessing = "SG derivative" if isinstance(spectral, SavitzkyGolayDerivative) else "SNV"
    if isinstance(model, PLSRegression):
        return f"PLS ({best_params['regressor__model__n_components']} komponen)", preprocessing
    return "PCA 95% + Ridge", preprocessing


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    iqr = float(np.subtract(*np.percentile(y_true, [75, 25])))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
        "rpiq": iqr / rmse if rmse > 0 else float("nan"),
    }


def benchmark_target(
    dataset: SpectralDataset,
    target: TargetSpec,
    budget: int,
    *,
    outer_repeats: int = 2,
    n_jobs: int = -1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = dataset.subset(budget)
    X = subset.spectra
    y = subset.metadata[target.code].to_numpy(dtype=float)
    outer = RepeatedKFold(n_splits=5, n_repeats=outer_repeats, random_state=RANDOM_STATE + budget)
    prediction_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []

    for split_index, (train_index, test_index) in enumerate(outer.split(X), start=1):
        repeat = (split_index - 1) // 5 + 1
        fold = (split_index - 1) % 5 + 1
        inner = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + split_index)
        search = build_search(inner, n_jobs=n_jobs)
        search.fit(X[train_index], y[train_index])
        predicted = np.asarray(search.predict(X[test_index])).reshape(-1)
        baseline = np.full(test_index.size, np.median(y[train_index]), dtype=float)
        model_name, preprocessing = describe_model(search.best_params_)
        model_metrics = regression_metrics(y[test_index], predicted)
        baseline_metrics = regression_metrics(y[test_index], baseline)
        metric_rows.append(
            {
                "target": target.code,
                "budget": budget,
                "repeat": repeat,
                "fold": fold,
                "model": model_name,
                "preprocessing": preprocessing,
                **model_metrics,
                "baseline_rmse": baseline_metrics["rmse"],
                "rmse_improvement_pct": 100 * (baseline_metrics["rmse"] - model_metrics["rmse"]) / baseline_metrics["rmse"],
            }
        )
        for position, observed, estimate, baseline_value in zip(
            test_index, y[test_index], predicted, baseline, strict=True
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
                    "model": model_name,
                    "preprocessing": preprocessing,
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(metric_rows)


def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_names = ["mae", "rmse", "r2", "rpiq", "baseline_rmse", "rmse_improvement_pct"]
    rows: list[dict[str, object]] = []
    for (target, budget), group in metrics.groupby(["target", "budget"], sort=True):
        row: dict[str, object] = {"target": target, "budget": int(budget), "n_folds": len(group)}
        for name in metric_names:
            values = group[name].to_numpy(dtype=float)
            row[f"{name}_median"] = float(np.nanmedian(values))
            row[f"{name}_q025"] = float(np.nanquantile(values, 0.025))
            row[f"{name}_q975"] = float(np.nanquantile(values, 0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def fit_final_models(dataset: SpectralDataset, artifact_dir: Path, n_jobs: int = -1) -> pd.DataFrame:
    full = dataset.subset(MAX_BUDGET)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for target in DEFAULT_TARGETS:
        inner = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        search = build_search(inner, n_jobs=n_jobs)
        search.fit(full.spectra, full.metadata[target.code].to_numpy(dtype=float))
        model_name, preprocessing = describe_model(search.best_params_)
        joblib.dump(search.best_estimator_, artifact_dir / f"model_{target.code.lower()}.joblib")
        rows.append(
            {
                "target": target.code,
                "model": model_name,
                "preprocessing": preprocessing,
                "inner_cv_rmse": -float(search.best_score_),
            }
        )
    return pd.DataFrame(rows)


def run_benchmark(
    project_root: Path,
    *,
    budgets: tuple[int, ...] = BUDGETS,
    outer_repeats: int = 2,
    n_jobs: int = -1,
    fit_final: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    started = perf_counter()
    dataset = load_processed_dataset(project_root)
    prediction_frames: list[pd.DataFrame] = []
    metric_frames: list[pd.DataFrame] = []
    for budget in budgets:
        for target in DEFAULT_TARGETS:
            print(f"Benchmark target={target.code} n={budget}", flush=True)
            predictions, metrics = benchmark_target(
                dataset, target, budget, outer_repeats=outer_repeats, n_jobs=n_jobs
            )
            prediction_frames.append(predictions)
            metric_frames.append(metrics)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics = pd.concat(metric_frames, ignore_index=True)
    summary = summarize_metrics(metrics)
    artifact_dir = project_root / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(artifact_dir / "predictions.csv", index=False)
    metrics.to_csv(artifact_dir / "fold_metrics.csv", index=False)
    summary.to_csv(artifact_dir / "summary.csv", index=False)
    if fit_final:
        fit_final_models(dataset, artifact_dir, n_jobs=n_jobs).to_csv(
            artifact_dir / "final_models.csv", index=False
        )
    run_manifest = {
        "budgets": list(budgets),
        "outer_folds": 5,
        "outer_repeats": outer_repeats,
        "inner_folds": 3,
        "random_state": RANDOM_STATE,
        "elapsed_seconds": perf_counter() - started,
        "targets": [target.to_dict() for target in DEFAULT_TARGETS],
    }
    (artifact_dir / "benchmark_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return predictions, metrics, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--skip-final", action="store_true")
    args = parser.parse_args()
    _, _, summary = run_benchmark(
        args.project_root.resolve(),
        outer_repeats=args.repeats,
        n_jobs=args.jobs,
        fit_final=not args.skip_final,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
