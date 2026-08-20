import numpy as np

from npk_spectra.preprocessing import SNVTransformer, SavitzkyGolayDerivative


def test_snv_is_rowwise_centered_and_scaled():
    transformed = SNVTransformer().fit_transform(np.asarray([[1, 2, 3], [10, 20, 30]], dtype=float))
    np.testing.assert_allclose(transformed.mean(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(transformed.std(axis=1), 1, atol=1e-12)


def test_savgol_preserves_shape():
    x = np.linspace(0, 1, 31)
    matrix = np.vstack([x**2, x**3])
    transformed = SavitzkyGolayDerivative(window_length=7, polyorder=2).fit_transform(matrix)
    assert transformed.shape == matrix.shape
    assert np.all(np.isfinite(transformed))

