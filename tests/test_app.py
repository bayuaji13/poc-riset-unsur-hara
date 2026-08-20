from pathlib import Path

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ZOO_MODELS = [
    "PLS scores → RBF-SVR",
    "PLS scores → Cubist",
    "PLS → Extra Trees",
    "PCA → Gaussian Process",
    "PLS → Deep Forest/gcForest",
    "Tiny 1D CNN",
]


def test_app_compiles():
    source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    compile(source, "app.py", "exec")


def test_all_dashboard_pages_smoke():
    app = AppTest.from_file(PROJECT_ROOT / "app.py", default_timeout=180).run()
    assert not app.exception
    pages = [
        "Dataset & target",
        "Spectrum explorer",
        "Eksplorasi NPK",
        "Learning curve",
        "Prediksi & residual",
        "Metodologi",
    ]
    for page in pages:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, f"Dashboard gagal pada halaman {page}: {app.exception}"

    app.sidebar.radio[0].set_value("Learning curve").run()
    app.selectbox[0].set_value("Self-attention · patch transformer").run()
    assert not app.exception
    assert any(metric.value == "0.815" for metric in app.metric)
    for model in ZOO_MODELS:
        app.selectbox[0].set_value(model).run()
        assert not app.exception, f"Learning curve gagal untuk {model}: {app.exception}"

    app.sidebar.radio[0].set_value("Prediksi & residual").run()
    for model in ["Self-attention · patch transformer", *ZOO_MODELS]:
        app.selectbox[0].set_value(model).run()
        assert not app.exception, f"Prediksi gagal untuk {model}: {app.exception}"
