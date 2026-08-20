"""Reusable OSSL MIR-to-NPK proof-of-concept components."""

from .config import BUDGETS, DEFAULT_TARGETS, MAX_BUDGET, MIR_GRID, TargetSpec
from .dataset import SpectralDataset, load_processed_dataset

__all__ = [
    "BUDGETS",
    "DEFAULT_TARGETS",
    "MAX_BUDGET",
    "MIR_GRID",
    "SpectralDataset",
    "TargetSpec",
    "load_processed_dataset",
]
