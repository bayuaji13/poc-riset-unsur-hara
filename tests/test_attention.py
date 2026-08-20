import numpy as np
import torch

from npk_spectra.attention import AttentionConfig, TinySpectralTransformer, train_attention


def test_tiny_transformer_shape_and_size():
    model = TinySpectralTransformer()
    output = model(torch.zeros(4, 1701))

    assert output.shape == (4, 3)
    assert 100_000 < model.parameter_count < 120_000


def test_attention_training_smoke_on_cpu():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(24, 31)).astype(np.float32)
    y = np.column_stack(
        (
            np.exp(0.2 * X[:, 0]) + 0.1,
            np.exp(0.1 * X[:, 1]) + 1.0,
            np.exp(0.15 * X[:, 2]) + 0.3,
        )
    ).astype(np.float32)
    config = AttentionConfig(
        patch_size=8,
        d_model=16,
        n_heads=2,
        n_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        batch_size=8,
        max_epochs=2,
        patience=1,
    )

    trained = train_attention(X, y, config=config, device="cpu")
    predictions = trained.predict(X[:3])

    assert predictions.shape == (3, 3)
    assert np.isfinite(predictions).all()
