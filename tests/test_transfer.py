import numpy as np
import pandas as pd
import torch

from npk_spectra.dataset import SpectralDataset
from npk_spectra.hybrid import HybridSpectralRegressor
from npk_spectra.transfer import select_nearest_ossl


def test_hybrid_shape_and_encoder_freeze():
    model = HybridSpectralRegressor()
    assert model(torch.zeros(3, 63)).shape == (3, 3)
    model.freeze_encoder()
    assert not any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())


def test_nearest_ossl_selection_uses_requested_count():
    spectra = np.asarray([[0, 1, 0, 1], [1, 2, 3, 4], [2, 4, 6, 8], [4, 3, 2, 1]], dtype=float)
    metadata = pd.DataFrame({"sample_id": ["a", "b", "c", "d"], "N": [1, 1, 1, 1], "P": [1, 1, 1, 1], "K": [1, 1, 1, 1]})
    dataset = SpectralDataset(metadata, spectra, np.arange(4, dtype=float))
    selected, distances = select_nearest_ossl(dataset, spectra[1:2], 2)
    assert selected.n_samples == 2
    assert np.all(np.diff(distances) >= 0)
    assert "similarity_distance" in selected.metadata
