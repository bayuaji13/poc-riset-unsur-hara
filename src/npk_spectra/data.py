"""Stream OSSL data and create a compact, deterministic MIR/NPK benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

from .config import (
    DEFAULT_TARGETS,
    JOIN_COLUMNS,
    MAX_BUDGET,
    MIR_COLUMNS,
    MIR_GRID,
    OSSL_LAB_URL,
    OSSL_MIR_URL,
    OSSL_VERSION,
    RANDOM_STATE,
)
from .dataset import SpectralDataset, save_processed_dataset
from .preprocessing import SNVTransformer


MIR_METADATA_COLUMNS = (
    "id.scan_local_c",
    "scan.mir.model.name_utf8_txt",
    "scan.mir.model.code_any_txt",
    "scan.mir.method.preparation_any_txt",
    "scan.mir.license.title_ascii_txt",
    "scan.mir.license.address_idn_url",
    "scan.mir.doi_idf_url",
)


def stable_priority(dataset_code: str, layer_id: str, seed: int = RANDOM_STATE) -> int:
    digest = hashlib.sha256(f"{seed}:{dataset_code}:{layer_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def load_complete_lab_rows(url: str = OSSL_LAB_URL) -> pd.DataFrame:
    target_columns = [target.column for target in DEFAULT_TARGETS]
    frame = pd.read_csv(url, compression="gzip", usecols=[*JOIN_COLUMNS, *target_columns])
    frame = frame.dropna(subset=target_columns).drop_duplicates(list(JOIN_COLUMNS), keep="first")
    frame = frame[(frame[target_columns] >= 0).all(axis=1)].copy()
    for target in DEFAULT_TARGETS:
        frame[target.code] = frame[target.column].astype(float)
    return frame[[*JOIN_COLUMNS, *[target.code for target in DEFAULT_TARGETS]]]


def collect_candidate_mir(
    lab: pd.DataFrame,
    *,
    mir_url: str = OSSL_MIR_URL,
    candidate_count: int = 2400,
    chunk_size: int = 256,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep the lowest deterministic hashes while streaming the large MIR file."""
    lab_indexed = lab.set_index(list(JOIN_COLUMNS), drop=False)
    eligible = set(lab_indexed.index)
    usecols = [*JOIN_COLUMNS, *MIR_METADATA_COLUMNS, *MIR_COLUMNS]
    kept: list[tuple[int, dict[str, object], np.ndarray]] = []

    for chunk in pd.read_csv(mir_url, compression="gzip", usecols=usecols, chunksize=chunk_size):
        keys = list(zip(chunk[JOIN_COLUMNS[0]], chunk[JOIN_COLUMNS[1]], strict=False))
        mask = np.fromiter((key in eligible for key in keys), dtype=bool, count=len(keys))
        if not mask.any():
            continue
        selected = chunk.loc[mask].copy()
        matrix = selected.loc[:, MIR_COLUMNS].to_numpy(dtype=float)
        finite = np.all(np.isfinite(matrix), axis=1)
        selected = selected.loc[finite].reset_index(drop=True)
        matrix = matrix[finite]
        for row_index, row in selected.iterrows():
            key = (row[JOIN_COLUMNS[0]], row[JOIN_COLUMNS[1]])
            priority = stable_priority(str(key[0]), str(key[1]))
            lab_values = lab_indexed.loc[key]
            if isinstance(lab_values, pd.DataFrame):
                lab_values = lab_values.iloc[0]
            metadata = {column: row[column] for column in [*JOIN_COLUMNS, *MIR_METADATA_COLUMNS]}
            metadata.update({target.code: float(lab_values[target.code]) for target in DEFAULT_TARGETS})
            kept.append((priority, metadata, matrix[row_index].astype(np.float32)))
        if len(kept) > candidate_count * 2:
            kept = sorted(kept, key=lambda item: item[0])[:candidate_count]

    kept = sorted(kept, key=lambda item: item[0])[:candidate_count]
    if len(kept) < MAX_BUDGET:
        raise RuntimeError(
            f"Hanya ditemukan {len(kept)} pasangan MIR–NPK lengkap; dibutuhkan {MAX_BUDGET}."
        )
    metadata = pd.DataFrame(item[1] for item in kept)
    spectra = np.vstack([item[2] for item in kept])
    return metadata, spectra


def _diversity_features(metadata: pd.DataFrame, spectra: np.ndarray) -> np.ndarray:
    snv = SNVTransformer().fit_transform(spectra)
    spectral_scores = PCA(n_components=min(12, len(spectra) - 1), random_state=RANDOM_STATE).fit_transform(snv)
    target_values = metadata[[target.code for target in DEFAULT_TARGETS]].to_numpy(dtype=float)
    features = np.column_stack((spectral_scores, RobustScaler().fit_transform(np.log1p(target_values))))
    return RobustScaler().fit_transform(features)


def diversity_order(metadata: pd.DataFrame, spectra: np.ndarray, max_samples: int = MAX_BUDGET) -> np.ndarray:
    """Kennard–Stone-like greedy order on spectral PCs plus robust target ranks."""
    features = _diversity_features(metadata, spectra)

    center = np.median(features, axis=0)
    first = int(np.argmax(np.linalg.norm(features - center, axis=1)))
    selected = [first]
    minimum_distance = np.linalg.norm(features - features[first], axis=1)
    minimum_distance[first] = -np.inf
    while len(selected) < max_samples:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        distance = np.linalg.norm(features - features[next_index], axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def extend_diversity_order(
    metadata: pd.DataFrame,
    spectra: np.ndarray,
    initial_indices: np.ndarray,
    max_samples: int = MAX_BUDGET,
) -> np.ndarray:
    """Extend an existing nested selection without changing its order."""
    features = _diversity_features(metadata, spectra)
    selected = [int(index) for index in initial_indices]
    minimum_distance = np.full(len(features), np.inf)
    for index in selected:
        minimum_distance = np.minimum(minimum_distance, np.linalg.norm(features - features[index], axis=1))
    minimum_distance[selected] = -np.inf
    while len(selected) < max_samples:
        next_index = int(np.argmax(minimum_distance))
        selected.append(next_index)
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(features - features[next_index], axis=1),
        )
        minimum_distance[selected] = -np.inf
    return np.asarray(selected, dtype=int)


def prepare_dataset(project_root: Path, candidate_count: int = 2400) -> SpectralDataset:
    lab = load_complete_lab_rows()
    metadata, spectra = collect_candidate_mir(lab, candidate_count=candidate_count, chunk_size=4096)
    priorities = np.asarray(
        [
            stable_priority(str(row[JOIN_COLUMNS[0]]), str(row[JOIN_COLUMNS[1]]))
            for _, row in metadata.iterrows()
        ],
        dtype=np.uint64,
    )
    legacy_pool = np.argsort(priorities)[:900]
    legacy_local_order = diversity_order(
        metadata.iloc[legacy_pool].reset_index(drop=True),
        spectra[legacy_pool],
        max_samples=300,
    )
    legacy_order = legacy_pool[legacy_local_order]
    order = extend_diversity_order(metadata, spectra, legacy_order, max_samples=MAX_BUDGET)
    metadata = metadata.iloc[order].reset_index(drop=True)
    spectra = spectra[order]
    metadata.insert(0, "sample_order", np.arange(1, len(metadata) + 1))
    metadata.insert(1, "sample_id", metadata[JOIN_COLUMNS[1]])
    dataset = SpectralDataset(metadata=metadata, spectra=spectra, grid=MIR_GRID.copy())
    output_dir = project_root / "data" / "processed"
    save_processed_dataset(dataset, output_dir)
    manifest = {
        "source_version": OSSL_VERSION,
        "lab_url": OSSL_LAB_URL,
        "mir_url": OSSL_MIR_URL,
        "join_columns": list(JOIN_COLUMNS),
        "selection": "preserved legacy 300-sample diversity order, then greedy spectral/target diversity extension to 1000",
        "candidate_count": candidate_count,
        "n_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "grid_cm-1": [float(dataset.grid[0]), float(dataset.grid[-1]), 2.0],
        "targets": [target.to_dict() for target in DEFAULT_TARGETS],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return dataset


def prepare_transfer_source(project_root: Path, *, pool_size: int = 10000, candidate_count: int = 25000) -> SpectralDataset:
    """Create a larger deterministic OSSL source pool for fold-local transfer selection."""
    if candidate_count < pool_size:
        raise ValueError("candidate_count harus setidaknya sebesar pool_size.")
    lab = load_complete_lab_rows()
    metadata, spectra = collect_candidate_mir(lab, candidate_count=candidate_count)
    order = diversity_order(metadata, spectra, max_samples=pool_size)
    metadata, spectra = metadata.iloc[order].reset_index(drop=True), spectra[order]
    metadata.insert(0, "sample_order", np.arange(1, len(metadata) + 1))
    metadata.insert(1, "sample_id", metadata[JOIN_COLUMNS[1]])
    dataset = SpectralDataset(metadata, spectra, MIR_GRID.copy())
    output = project_root / "data" / "ossl_transfer"
    save_processed_dataset(dataset, output)
    (output / "manifest.json").write_text(json.dumps({"purpose": "deterministic OSSL source pool for per-fold local spectral nearest-neighbor selection", "pool_size": pool_size, "candidate_count": candidate_count, "targets": [target.to_dict() for target in DEFAULT_TARGETS]}, indent=2), encoding="utf-8")
    return dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--candidates", type=int, default=2400)
    args = parser.parse_args()
    dataset = prepare_dataset(args.project_root.resolve(), args.candidates)
    print(f"Dataset siap: {dataset.n_samples} sampel × {dataset.n_features} kanal")


if __name__ == "__main__":
    main()
