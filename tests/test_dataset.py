import numpy as np
import pandas as pd
import pytest

from npk_spectra.dataset import SpectralDataset


def make_dataset() -> SpectralDataset:
    metadata = pd.DataFrame(
        {
            "sample_id": ["c", "a", "b"],
            "sample_order": [3, 1, 2],
            "N": [3.0, 1.0, 2.0],
            "P": [30.0, 10.0, 20.0],
            "K": [0.3, 0.1, 0.2],
        }
    )
    spectra = np.asarray([[30, 31], [10, 11], [20, 21]], dtype=float)
    return SpectralDataset(metadata=metadata, spectra=spectra, grid=np.asarray([600, 602], dtype=float))


def test_nested_subset_uses_sample_order():
    subset = make_dataset().subset(2)
    assert subset.metadata["sample_id"].tolist() == ["a", "b"]
    assert subset.spectra[:, 0].tolist() == [10, 20]


def test_dataset_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Jumlah metadata"):
        SpectralDataset(pd.DataFrame({"x": [1]}), np.ones((2, 3)), np.arange(3))

