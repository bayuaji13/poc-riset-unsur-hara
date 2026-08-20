import numpy as np
import torch

from npk_spectra.cnn import CNNConfig, TinySpectralCNN, train_cnn


def test_tiny_cnn_shape_and_size():
    model = TinySpectralCNN()
    output = model(torch.zeros(4, 1701))
    assert output.shape == (4, 3)
    assert 40_000 < model.parameter_count < 60_000


def test_cnn_training_smoke_on_cpu():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 63)).astype(np.float32)
    y = np.column_stack(
        (np.exp(0.2 * X[:, 0]) + 0.1, np.exp(0.1 * X[:, 1]) + 1, np.exp(0.15 * X[:, 2]) + 0.3)
    ).astype(np.float32)
    config = CNNConfig(
        channels=(4, 8),
        kernels=(5, 3),
        pooled_length=2,
        hidden_size=8,
        dropout=0,
        batch_size=8,
        max_epochs=2,
        patience=1,
    )
    trained = train_cnn(X, y, config=config, device="cpu")
    predictions = trained.predict(X[:3])
    assert predictions.shape == (3, 3)
    assert np.isfinite(predictions).all()
