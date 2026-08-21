"""Interactive OSSL MIR-to-NPK proof-of-concept dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from npk_spectra import BUDGETS, DEFAULT_TARGETS, MAX_BUDGET, load_processed_dataset  # noqa: E402
from npk_spectra.config import target_by_code  # noqa: E402


TARGET_COLORS = {target.code: target.color for target in DEFAULT_TARGETS}
MODEL_REGISTRY = {
    "Klasik · PLS/PCA–Ridge": ("classical", None),
    "Self-attention · patch transformer": ("attention", None),
    "PLS scores → RBF-SVR": ("zoo", "PLS–RBF-SVR"),
    "PLS scores → Cubist": ("zoo", "PLS–Cubist"),
    "PLS → Extra Trees": ("zoo", "PLS–Extra Trees"),
    "PCA → Gaussian Process": ("zoo", "PCA–Gaussian Process"),
    "PLS → Deep Forest/gcForest": ("zoo", "PLS–Cascade Forest"),
    "Tiny 1D CNN": ("zoo", "Tiny 1D CNN"),
}
PAGES = (
    "Dataset & target",
    "Spectrum explorer",
    "Eksplorasi NPK",
    "Learning curve",
    "Prediksi & residual",
    "Transfer Lokal",
    "Metodologi",
)


st.set_page_config(page_title="NPK Spectral Bench", layout="wide", initial_sidebar_state="expanded")


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700&family=Fira+Code:wght@400;500;600&family=Fira+Sans:wght@400;500;600&display=swap');
        :root {
          --ink:#0B132B; --panel:#13213C; --paper:#F4F7F6; --line:#CBD7D5;
          --soft:#526562; --cyan:#008C95; --n:#168657; --p:#C06C00; --k:#6D48C7;
        }
        html, body, [class*="css"] { font-family:"Fira Sans",sans-serif; color:var(--ink); }
        .stApp { background:var(--paper); }
        .block-container { max-width:1480px; padding-top:1.25rem; padding-bottom:4rem; }
        h1,h2,h3 { font-family:"Barlow Condensed",sans-serif!important; letter-spacing:-.01em; }
        code,pre,[data-testid="stMetricValue"],[data-testid="stCaptionContainer"] { font-family:"Fira Code",monospace; }
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] p,
        [data-testid="stMain"] label,
        [data-testid="stMain"] [data-testid="stMetricLabel"],
        [data-testid="stMain"] [data-testid="stMetricValue"] { color:var(--ink)!important; }
        [data-testid="stMain"] [data-testid="stCaptionContainer"] { color:var(--soft)!important; }
        [data-testid="stSidebar"] { background:var(--panel); border-right:1px solid #2B3B58; }
        [data-testid="stSidebar"] * { color:#EAF2F1; }
        [data-testid="stSidebar"] [role="radiogroup"] label { padding:.48rem .65rem; border-left:3px solid transparent; }
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) { background:#1D3154; border-left-color:#35CBD0; }
        .brand { border-bottom:1px solid #3A4964; margin-bottom:1rem; padding:.25rem 0 1rem; }
        .brand strong { display:block; font:600 1.25rem "Barlow Condensed"; letter-spacing:.08em; text-transform:uppercase; }
        .brand span { color:#9CB2B3!important; font:.67rem "Fira Code"; letter-spacing:.1em; text-transform:uppercase; }
        .instrument-hero { display:grid; grid-template-columns:minmax(280px,.8fr) minmax(420px,1.2fr); gap:2rem; align-items:center; background:var(--ink); color:#ECF7F5; border-top:4px solid var(--cyan); padding:2rem 2.2rem; margin-bottom:1rem; overflow:hidden; }
        .hero-kicker { color:#54D5D8; font:.72rem "Fira Code"; letter-spacing:.12em; text-transform:uppercase; }
        .instrument-hero h1 { color:white; font-size:clamp(3rem,6vw,5.3rem); line-height:.86; margin:.45rem 0 .8rem; text-transform:uppercase; }
        .instrument-hero p { color:#B9CACB; line-height:1.55; max-width:540px; }
        .trace svg { width:100%; height:170px; overflow:visible; }
        .trace-grid { stroke:#31415C; stroke-width:1; }
        .trace-line { fill:none; stroke:#35CBD0; stroke-width:3; vector-effect:non-scaling-stroke; }
        .ruler { display:flex; justify-content:space-between; color:#87A2A5; font:.65rem "Fira Code"; }
        .warning { background:#FFF4DF; border:1px solid #E9C382; border-left:4px solid var(--p); color:#533400; padding:.8rem 1rem; margin:1rem 0 1.4rem; }
        .section-head { display:grid; grid-template-columns:1fr minmax(280px,.6fr); gap:2rem; align-items:end; border-bottom:1px solid var(--line); padding-bottom:.7rem; margin:.25rem 0 1rem; }
        .section-head h2 { font-size:clamp(2rem,4vw,3.3rem); line-height:.95; margin:0; text-transform:uppercase; }
        .section-head p { color:var(--soft); line-height:1.5; margin:0; }
        [data-testid="stMetric"] { background:white; border:1px solid var(--line); border-top:3px solid var(--cyan); border-radius:4px; padding:.8rem; }
        [data-testid="stDataFrame"], .js-plotly-plot { border:1px solid var(--line); background:white; }
        .target-card { background:white; border:1px solid var(--line); color:var(--ink)!important; min-height:152px; padding:1rem; }
        .target-card b,.target-card p,.target-card strong { color:var(--ink)!important; }
        .target-card b { display:block; font:600 2rem "Barlow Condensed"; }
        .target-card code { color:var(--soft); font-size:.7rem; word-break:break-all; }
        .method-note { background:#E7F3F1; border-left:4px solid var(--cyan); padding:.8rem 1rem; }
        button,[role="button"] { cursor:pointer; transition:background-color 180ms ease,border-color 180ms ease; }
        button:focus-visible,[role="button"]:focus-visible,input:focus-visible { outline:3px solid #35CBD0!important; outline-offset:2px; }
        @media (prefers-reduced-motion:reduce) { *,*::before,*::after { transition:none!important; } }
        @media (max-width:800px) { .instrument-hero,.section-head { grid-template-columns:1fr; } .trace { display:none; } .block-container { padding-top:.6rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero() -> None:
    st.markdown(
        """
        <section class="instrument-hero">
          <div>
            <div class="hero-kicker">OSSL MIR / calibration proof-of-concept</div>
            <h1>NPK Spectral<br>Bench</h1>
            <p>Uji learning curve hingga 1.000 sampel untuk memodelkan nitrogen, fosfor, dan kalium dari sidik-jari spektrum tanah.</p>
          </div>
          <div class="trace" aria-label="Ilustrasi jejak spektrum MIR dari 4000 hingga 600 inverse centimeter">
            <svg viewBox="0 0 720 170" role="img" aria-labelledby="mir-trace-title">
              <title id="mir-trace-title">Jejak spektrum MIR dari 4000 hingga 600 inverse centimeter</title>
              <line class="trace-grid" x1="0" y1="25" x2="720" y2="25"/><line class="trace-grid" x1="0" y1="85" x2="720" y2="85"/><line class="trace-grid" x1="0" y1="145" x2="720" y2="145"/>
              <path class="trace-line" d="M0,122 C35,118 55,117 78,120 C105,124 116,98 138,104 C168,112 180,125 207,111 C226,101 233,47 253,52 C278,58 280,112 309,105 C329,99 338,37 362,43 C389,51 391,120 421,111 C445,103 451,73 476,78 C502,83 502,121 531,114 C557,107 564,89 587,91 C614,93 620,120 649,113 C678,106 695,104 720,108"/>
            </svg>
            <div class="ruler"><span>4000 cm⁻¹</span><span>3000</span><span>2000</span><span>1000</span><span>600</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, description: str) -> None:
    st.markdown(
        f'<div class="section-head"><h2>{title}</h2><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_data():
    return load_processed_dataset(PROJECT_ROOT)


@st.cache_data(show_spinner=False)
def load_artifacts(model_label: str = "Klasik · PLS/PCA–Ridge") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    source, family = MODEL_REGISTRY[model_label]
    prefix = "attention_" if source == "attention" else "zoo_" if source == "zoo" else ""
    frames = (
        pd.read_csv(PROJECT_ROOT / "artifacts" / f"{prefix}predictions.csv"),
        pd.read_csv(PROJECT_ROOT / "artifacts" / f"{prefix}fold_metrics.csv"),
        pd.read_csv(PROJECT_ROOT / "artifacts" / f"{prefix}summary.csv"),
    )
    if source == "zoo":
        return tuple(frame[frame["family"] == family].copy() for frame in frames)
    return frames


def data_ready() -> bool:
    return (PROJECT_ROOT / "data" / "processed" / "spectra.npz").exists()


def artifacts_ready(model_label: str = "Klasik · PLS/PCA–Ridge") -> bool:
    source, family = MODEL_REGISTRY[model_label]
    prefix = "attention_" if source == "attention" else "zoo_" if source == "zoo" else ""
    files_ready = all(
        (PROJECT_ROOT / "artifacts" / f"{prefix}{name}").exists()
        for name in ("predictions.csv", "fold_metrics.csv", "summary.csv")
    )
    if not files_ready or source != "zoo":
        return files_ready
    available = pd.read_csv(PROJECT_ROOT / "artifacts" / "zoo_summary.csv", usecols=["family"])["family"]
    return bool((available == family).any())


def missing_state(kind: str, model_label: str = "Klasik · PLS/PCA–Ridge") -> None:
    source, _ = MODEL_REGISTRY.get(model_label, ("classical", None))
    if kind == "dataset":
        command = "python scripts/prepare_data.py"
    elif source == "attention":
        command = "python scripts/run_attention_benchmark.py"
    elif source == "zoo":
        command = "python scripts/run_model_zoo.py"
    else:
        command = "python scripts/run_benchmark.py"
    st.error(f"{kind.title()} belum tersedia. Jalankan `{command}` dari root project.")


def model_selector(key: str) -> str:
    return st.selectbox("Model family", list(MODEL_REGISTRY), key=key)


def page_dataset(dataset) -> None:
    section("Dataset & definisi target", "Kontrak kimia lebih penting daripada sekadar label N, P, dan K.")
    cols = st.columns(4)
    cols[0].metric("Sampel lengkap", f"{dataset.n_samples:,}")
    cols[1].metric("Kanal MIR", f"{dataset.n_features:,}")
    cols[2].metric("Rentang", "600–4000 cm⁻¹")
    cols[3].metric("Learning curve", "60 → 1.000")
    st.markdown("### Target aktif")
    target_columns = st.columns(3)
    for column, target in zip(target_columns, DEFAULT_TARGETS, strict=True):
        column.markdown(
            f'<div class="target-card" style="border-top:4px solid {target.color}"><b>{target.code} · {target.label}</b><p>{target.method}<br><strong>{target.unit}</strong></p><code>{target.column}</code></div>',
            unsafe_allow_html=True,
        )
    st.markdown("### Asal sampel")
    sources = dataset.metadata.groupby("dataset.code_ascii_txt", as_index=False).size().sort_values("size", ascending=False)
    left, right = st.columns([1.2, 1])
    left.plotly_chart(
        px.bar(sources, x="dataset.code_ascii_txt", y="size", labels={"size": "Sampel", "dataset.code_ascii_txt": "Library"}, color_discrete_sequence=["#008C95"]),
        width="stretch",
    )
    right.dataframe(
        dataset.metadata[["sample_id", "dataset.code_ascii_txt", "scan.mir.model.code_any_txt", "sample_order"]],
        hide_index=True,
        width="stretch",
        height=340,
    )


def page_spectra(dataset) -> None:
    section("Spectrum explorer", "Lihat absorbance MIR asli pada grid OSSL yang sudah distandardisasi.")
    selected = st.selectbox("Pilih sampel", dataset.metadata["sample_id"].tolist())
    row = int(dataset.metadata.index[dataset.metadata["sample_id"] == selected][0])
    fig = go.Figure(go.Scatter(x=dataset.grid, y=dataset.spectra[row], mode="lines", line={"color": "#008C95", "width": 1.7}, name=selected))
    fig.update_layout(xaxis_title="Wavenumber (cm⁻¹)", yaxis_title="Absorbance (log10)", xaxis={"autorange": "reversed"}, height=470)
    st.plotly_chart(fig, width="stretch")
    readouts = st.columns(5)
    readouts[0].metric("Sample order", int(dataset.metadata.iloc[row]["sample_order"]))
    for idx, target in enumerate(DEFAULT_TARGETS, start=1):
        readouts[idx].metric(target.code, f"{dataset.metadata.iloc[row][target.code]:.3g} {target.unit}")
    readouts[4].metric("Library", dataset.metadata.iloc[row]["dataset.code_ascii_txt"])


def page_targets(dataset) -> None:
    section("Eksplorasi NPK", "Distribusi target dan hubungan antar-metode pada 1.000 pasangan MIR–lab lengkap.")
    selected_target = st.selectbox("Target", [target.code for target in DEFAULT_TARGETS], format_func=lambda code: f"{code} · {target_by_code(code).label}")
    target = target_by_code(selected_target)
    left, right = st.columns([1.1, 1])
    left.plotly_chart(
        px.histogram(dataset.metadata, x=selected_target, color="dataset.code_ascii_txt", marginal="box", nbins=30, labels={selected_target: f"{target.label} ({target.unit})", "dataset.code_ascii_txt": "Library"}),
        width="stretch",
    )
    corr = dataset.metadata[[target.code for target in DEFAULT_TARGETS]].corr()
    right.plotly_chart(
        px.imshow(corr, text_auto=".2f", color_continuous_scale=[[0, "#F4F7F6"], [1, "#008C95"]], zmin=-1, zmax=1, labels={"color": "Korelasi"}),
        width="stretch",
    )
    st.dataframe(dataset.metadata[["sample_id", "dataset.code_ascii_txt", "N", "P", "K"]].describe().T, width="stretch")


def page_learning_curve() -> None:
    section("Learning curve", "Median outer-fold CV; pita menunjukkan rentang 2,5–97,5% antar-fold.")
    model_label = model_selector("learning_model")
    if not artifacts_ready(model_label):
        missing_state("artifact benchmark", model_label)
        return
    _, _, summary = load_artifacts(model_label)
    selected_target = st.radio("Target", [target.code for target in DEFAULT_TARGETS], horizontal=True)
    metric = st.selectbox("Metrik", ["rmse", "mae", "r2", "rpiq", "rmse_improvement_pct"], format_func=lambda value: value.upper().replace("_", " "))
    filtered = summary[summary["target"] == selected_target].sort_values("budget")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered["budget"], y=filtered[f"{metric}_q975"], mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=filtered["budget"], y=filtered[f"{metric}_q025"], mode="lines", fill="tonexty", fillcolor="rgba(0,140,149,.14)", line={"width": 0}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=filtered["budget"], y=filtered[f"{metric}_median"], mode="lines+markers", line={"color": TARGET_COLORS[selected_target], "width": 3}, marker={"size": 9}, name=model_label))
    available_comparisons = [label for label in MODEL_REGISTRY if label != model_label and artifacts_ready(label)]
    preferred = "Klasik · PLS/PCA–Ridge" if model_label != "Klasik · PLS/PCA–Ridge" else "Self-attention · patch transformer"
    default_comparison = preferred if preferred in available_comparisons else (available_comparisons[0] if available_comparisons else None)
    comparison_options = ["Tanpa pembanding", *available_comparisons]
    comparison_index = comparison_options.index(default_comparison) if default_comparison else 0
    comparison_label = st.selectbox("Bandingkan dengan", comparison_options, index=comparison_index)
    if comparison_label != "Tanpa pembanding":
        _, _, comparison = load_artifacts(comparison_label)
        comparison = comparison[comparison["target"] == selected_target].sort_values("budget")
        fig.add_trace(go.Scatter(x=comparison["budget"], y=comparison[f"{metric}_median"], mode="lines+markers", line={"color": "#526562", "width": 2, "dash": "dash"}, marker={"size": 7}, name=comparison_label))
    fig.update_layout(xaxis_title="Jumlah sampel training", yaxis_title=metric.upper(), height=470)
    st.plotly_chart(fig, width="stretch")
    latest = filtered.iloc[-1]
    cols = st.columns(4)
    latest_budget = int(latest["budget"])
    budget_label = f"{latest_budget:,}".replace(",", ".")
    st.caption(f"{model_label} · {int(latest['n_folds'])} outer validation folds per budget")
    cols[0].metric(f"R² median @{budget_label}", f"{latest['r2_median']:.3f}")
    cols[1].metric(f"RMSE median @{budget_label}", f"{latest['rmse_median']:.3g}")
    cols[2].metric(f"RPIQ median @{budget_label}", f"{latest['rpiq_median']:.3f}")
    cols[3].metric("vs baseline", f"{latest['rmse_improvement_pct_median']:.1f}%")
    leaderboard_rows: list[dict[str, object]] = []
    for label in MODEL_REGISTRY:
        if not artifacts_ready(label):
            continue
        _, _, candidate = load_artifacts(label)
        row = candidate[(candidate["target"] == selected_target) & (candidate["budget"] == latest_budget)]
        if row.empty:
            continue
        value = row.iloc[0]
        leaderboard_rows.append(
            {
                "Model": label,
                "R² median": value["r2_median"],
                "RMSE median": value["rmse_median"],
                "RPIQ median": value["rpiq_median"],
                "vs baseline (%)": value["rmse_improvement_pct_median"],
                "Outer folds": int(value["n_folds"]),
            }
        )
    if leaderboard_rows:
        st.markdown(f"### Leaderboard {selected_target} @{budget_label}")
        leaderboard = pd.DataFrame(leaderboard_rows).sort_values("RMSE median")
        st.dataframe(
            leaderboard.style.format(
                {"R² median": "{:.3f}", "RMSE median": "{:.4g}", "RPIQ median": "{:.3f}", "vs baseline (%)": "{:.1f}"}
            ),
            hide_index=True,
            width="stretch",
        )


def page_predictions() -> None:
    section("Prediksi & residual", "Semua titik berasal dari fold validasi—bukan hasil fit pada sampel yang sama.")
    model_label = model_selector("prediction_model")
    if not artifacts_ready(model_label):
        missing_state("artifact benchmark", model_label)
        return
    predictions, metrics, _ = load_artifacts(model_label)
    target_code = st.selectbox("Target", [target.code for target in DEFAULT_TARGETS])
    budget = st.select_slider("Jumlah sampel", options=list(BUDGETS), value=MAX_BUDGET)
    subset = predictions[(predictions["target"] == target_code) & (predictions["budget"] == budget)]
    aggregated = subset.groupby("sample_id", as_index=False).agg(observed=("observed", "first"), predicted=("predicted", "mean"), residual=("residual", "mean"))
    target = target_by_code(target_code)
    left, right = st.columns(2)
    scatter = px.scatter(aggregated, x="observed", y="predicted", hover_name="sample_id", labels={"observed": f"Observed ({target.unit})", "predicted": f"Predicted ({target.unit})"}, color_discrete_sequence=[target.color])
    bounds = [min(aggregated["observed"].min(), aggregated["predicted"].min()), max(aggregated["observed"].max(), aggregated["predicted"].max())]
    scatter.add_trace(go.Scatter(x=bounds, y=bounds, mode="lines", line={"color": "#526562", "dash": "dash"}, name="1:1"))
    left.plotly_chart(scatter, width="stretch")
    right.plotly_chart(px.scatter(aggregated, x="predicted", y="residual", hover_name="sample_id", color_discrete_sequence=[target.color]), width="stretch")
    st.dataframe(metrics[(metrics["target"] == target_code) & (metrics["budget"] == budget)], hide_index=True, width="stretch")


def page_transfer() -> None:
    section("Transfer OSSL → Lokal", "Leave-one-county-out untuk local-first, OSSL zero-shot, dan head-only fine-tuning.")
    root = PROJECT_ROOT / "artifacts"
    paths = [root / "transfer_predictions.csv", root / "transfer_fold_metrics.csv", root / "transfer_summary.csv"]
    if not all(path.exists() for path in paths):
        st.error("Hasil transfer belum tersedia. Jalankan `python scripts/run_transfer_benchmark.py`.")
        return
    predictions, metrics, summary = (pd.read_csv(path) for path in paths)
    target = st.radio("Target", [item.code for item in DEFAULT_TARGETS], horizontal=True, key="transfer_target")
    st.dataframe(summary[summary["target"] == target].sort_values("rmse_median"), hide_index=True, width="stretch")
    view = predictions[predictions["target"] == target]
    figure = px.scatter(view, x="observed", y="predicted", color="variant", symbol="held_out_county", hover_name="sample_id")
    bounds = [min(view.observed.min(), view.predicted.min()), max(view.observed.max(), view.predicted.max())]
    figure.add_trace(go.Scatter(x=bounds, y=bounds, mode="lines", line={"dash": "dash", "color": "#526562"}, name="1:1"))
    st.plotly_chart(figure, width="stretch")
    workbook = root / "NPK_Filled_Soil_Data_v2_with_transfer_results.xlsx"
    if workbook.exists():
        st.download_button("Download workbook hasil transfer", workbook.read_bytes(), file_name=workbook.name, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.caption("Workbook hasil akan tersedia setelah `python scripts/export_transfer_workbook.py` dijalankan.")
    st.warning("P dan K memakai kecocokan metode OSSL provisional karena SOP laboratorium lokal belum diketahui.")


def page_methodology() -> None:
    section("Metodologi", "Batas interpretasi ditampilkan bersama hasil, bukan disembunyikan di catatan kaki.")
    st.markdown(
        """
        <div class="method-note"><strong>Yang dibuktikan:</strong> pipeline data, preprocessing, nested cross-validation, learning curve, dan pelaporan ketidakpastian dapat bekerja pada 60–1.000 pasangan spectrum–lab.</div>

        ### Alur evaluasi

        1. Join data MIR dan laboratorium menggunakan ID resmi OSSL.
        2. Pilih 1.000 complete cases secara deterministik dan susun subset nested.
        3. Benchmark klasik memakai inner CV untuk memilih SNV/turunan Savitzky–Golay serta PLS/PCA–Ridge.
        4. Model zoo memakai konfigurasi yang ditetapkan sebelum outer CV: PLS–SVR, PLS–Cubist, PLS–ExtraTrees, PCA–GPR, dan cascade forest dua-layer dengan OOF augmentation.
        5. Tiny CNN memakai konvolusi lokal; self-attention memakai patch 16 kanal. Keduanya memprediksi N/P/K bersama dan early stopping hanya melihat split di dalam outer-training fold.
        6. Semua metrik dihitung pada outer validation fold dalam unit laboratorium asli.
        7. Median training-fold menjadi baseline yang wajib ditampilkan.

        Cascade forest di PoC ini adalah implementasi pure scikit-learn bergaya gcForest. Package DF21 resmi tidak dipakai karena belum menyediakan wheel Python 3.12.

        ### Yang belum dibuktikan

        - Transfer performa ke instrumen Shimadzu/ATR lokal.
        - Kesetaraan total N OSSL dengan metode N laboratorium yang kelak dipakai.
        - Kesetaraan Olsen P atau NH4OAc K dengan metode ekstraksi lain.
        - Akurasi untuk tanah Indonesia di luar domain spektral dan rentang target training.

        ### Migrasi data lokal

        Isi `data/local_manifest_template.csv`, samakan unit dan metode target, lalu buat loader Shimadzu yang menghasilkan kontrak internal yang sama. Gunakan group CV berdasarkan lokasi atau batch sampling agar replikasi dekat tidak bocor antar-fold.
        """,
        unsafe_allow_html=True,
    )
    manifest_path = PROJECT_ROOT / "data" / "processed" / "manifest.json"
    if manifest_path.exists():
        with st.expander("Manifest dataset reproducible"):
            st.json(json.loads(manifest_path.read_text(encoding="utf-8")))


inject_styles()
with st.sidebar:
    st.markdown('<div class="brand"><strong>NPK Spectral Bench</strong><span>research instrument / poc-01</span></div>', unsafe_allow_html=True)
    page = st.radio("Navigasi", PAGES)
    st.caption("OSSL v1.2 · MIR 600–4000 cm⁻¹")

hero()
st.markdown('<div class="warning"><strong>Batas interpretasi:</strong> performa OSSL tidak membuktikan model akan akurat untuk spektrum Shimadzu/ATR atau metode laboratorium lokal.</div>', unsafe_allow_html=True)

if page == "Metodologi":
    page_methodology()
elif not data_ready():
    missing_state("dataset")
else:
    dataset = load_data()
    if page == "Dataset & target":
        page_dataset(dataset)
    elif page == "Spectrum explorer":
        page_spectra(dataset)
    elif page == "Eksplorasi NPK":
        page_targets(dataset)
    elif page == "Learning curve":
        page_learning_curve()
    elif page == "Transfer Lokal":
        page_transfer()
    else:
        page_predictions()
