import numpy as np
import pandas as pd

from npk_spectra.benchmark import regression_metrics, summarize_metrics


def test_regression_metrics_perfect_prediction():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    metrics = regression_metrics(values, values)
    assert metrics["rmse"] == 0
    assert metrics["mae"] == 0
    assert metrics["r2"] == 1


def test_summary_has_empirical_intervals():
    rows = []
    for fold, rmse in enumerate([1.0, 2.0, 3.0], start=1):
        rows.append(
            {
                "target": "N", "budget": 60, "fold": fold, "mae": rmse,
                "rmse": rmse, "r2": 0.1, "rpiq": 1.2,
                "baseline_rmse": 4.0, "rmse_improvement_pct": 25.0,
            }
        )
    summary = summarize_metrics(pd.DataFrame(rows))
    assert summary.loc[0, "rmse_median"] == 2.0
    assert summary.loc[0, "rmse_q025"] < summary.loc[0, "rmse_q975"]

