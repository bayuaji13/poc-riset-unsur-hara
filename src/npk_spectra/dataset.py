"""Processed spectral dataset contract shared by notebook, models, and app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SpectralDataset:
    metadata: pd.DataFrame
    spectra: np.ndarray
    grid: np.ndarray

    def __post_init__(self) -> None:
        if self.spectra.ndim != 2:
            raise ValueError("Matriks spektrum harus dua dimensi.")
        if len(self.metadata) != self.spectra.shape[0]:
            raise ValueError("Jumlah metadata dan spektrum tidak sama.")
        if self.spectra.shape[1] != self.grid.size:
            raise ValueError("Jumlah kanal spektrum dan grid tidak sama.")
        if not np.all(np.isfinite(self.spectra)):
            raise ValueError("Spektrum mengandung NaN atau infinity.")

    @property
    def n_samples(self) -> int:
        return self.spectra.shape[0]

    @property
    def n_features(self) -> int:
        return self.spectra.shape[1]

    def subset(self, size: int) -> "SpectralDataset":
        if size > self.n_samples:
            raise ValueError(f"Subset {size} melebihi {self.n_samples} sampel tersedia.")
        ordered = self.metadata.sort_values("sample_order")
        positions = ordered.index[:size].to_numpy()
        return SpectralDataset(
            metadata=self.metadata.loc[positions].reset_index(drop=True),
            spectra=self.spectra[positions],
            grid=self.grid.copy(),
        )


def save_processed_dataset(dataset: SpectralDataset, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.metadata.to_csv(output_dir / "metadata.csv", index=False)
    np.savez_compressed(output_dir / "spectra.npz", spectra=dataset.spectra, grid=dataset.grid)


def load_processed_dataset(project_root: Path | str) -> SpectralDataset:
    root = Path(project_root) / "data" / "processed"
    metadata_path = root / "metadata.csv"
    spectra_path = root / "spectra.npz"
    if not metadata_path.exists() or not spectra_path.exists():
        raise FileNotFoundError("Dataset PoC belum dibuat. Jalankan `python scripts/prepare_data.py`.")
    arrays = np.load(spectra_path)
    return SpectralDataset(
        metadata=pd.read_csv(metadata_path),
        spectra=arrays["spectra"].astype(float),
        grid=arrays["grid"].astype(float),
    )

