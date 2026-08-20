"""Leakage-safe row-wise spectral preprocessing transformers."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter
from sklearn.base import BaseEstimator, TransformerMixin


class SNVTransformer(TransformerMixin, BaseEstimator):
    """Standard normal variate, independently applied to each spectrum."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SNVTransformer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=float)
        means = values.mean(axis=1, keepdims=True)
        stds = values.std(axis=1, keepdims=True)
        if np.any(np.isclose(stds, 0)):
            raise ValueError("SNV tidak dapat diterapkan pada spektrum konstan.")
        return (values - means) / stds


class SavitzkyGolayDerivative(TransformerMixin, BaseEstimator):
    """First derivative with fixed MIR grid spacing."""

    def __init__(self, window_length: int = 21, polyorder: int = 3, delta: float = 2.0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.delta = delta

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "SavitzkyGolayDerivative":
        if self.window_length % 2 == 0 or self.window_length <= self.polyorder:
            raise ValueError("Window Savitzky–Golay harus ganjil dan lebih besar dari polyorder.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return savgol_filter(
            np.asarray(X, dtype=float),
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=1,
            delta=self.delta,
            axis=1,
            mode="interp",
        )

