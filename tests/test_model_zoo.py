import numpy as np

from npk_spectra.model_zoo import CascadeForestRegressor, PLSScoreTransformer, build_model


def spectral_fixture():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(36, 31))
    y = np.exp(0.3 * X[:, 0] - 0.15 * X[:, 1]) + 0.1
    return X, y


def test_pls_score_transformer_reduces_features():
    X, y = spectral_fixture()
    scores = PLSScoreTransformer(5).fit_transform(X, y)
    assert scores.shape == (36, 5)


def test_small_data_models_fit_and_predict():
    X, y = spectral_fixture()
    for family in ("PLS–RBF-SVR", "PLS–Cubist", "PCA–Gaussian Process"):
        model = build_model(family, seed=42, n_jobs=1).fit(X[:30], y[:30])
        predictions = model.predict(X[30:])
        assert predictions.shape == (6,)
        assert np.isfinite(predictions).all()


def test_cascade_forest_oof_layers_predict():
    X, y = spectral_fixture()
    model = CascadeForestRegressor(n_layers=2, n_estimators=5, cv=3, n_jobs=1).fit(X[:30], y[:30])
    predictions = model.predict(X[30:])
    assert predictions.shape == (6,)
    assert np.isfinite(predictions).all()
