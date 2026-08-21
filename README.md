# NPK Spectral Bench

Proof-of-concept reproducible untuk menguji prediksi nitrogen, fosfor, dan kalium dari spektrum MIR tanah dengan learning curve `60 → 120 → 180 → 240 → 300 → 1000` sampel.

Project ini memakai [Open Soil Spectral Library v1.2](https://docs.soilspectroscopy.org/db-access.html). OSSL menyediakan MIR absorbance 600–4000 cm⁻¹ setiap 2 cm⁻¹ dan data laboratorium yang dapat digabungkan melalui `dataset.code_ascii_txt` dan `id.layer_uuid_txt`.

> **Batas utama:** performa pada OSSL bukan bukti model akan akurat untuk spektrum Shimadzu/ATR, tanah Indonesia, atau metode laboratorium yang berbeda. PoC ini membuktikan pipeline dan mengukur kelayakan awal—bukan menghasilkan alat diagnosis agronomi.

## Target default

| Target | Kolom OSSL | Metode | Unit |
|---|---|---|---|
| N | `n.tot_usda.a623_w.pct` | total N, USDA a623/NCS | % berat |
| P | `p.ext_usda.a274_mg.kg` | available P, Olsen | mg/kg |
| K | `k.ext_usda.a725_cmolc.kg` | exchangeable K, NH4OAc pH 7 | cmolc/kg |

Definisi ini sengaja eksplisit. Jangan menggabungkan P atau K dari metode ekstraksi berbeda hanya karena sama-sama berlabel “P” atau “K”. Saat data lokal tersedia, ubah target agar identik dengan metode laboratorium aktual.

## Menjalankan

Python 3.12 dan `uv` direkomendasikan:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements-dev.txt
source .venv/bin/activate
python scripts/prepare_data.py
python scripts/run_benchmark.py
python scripts/run_attention_benchmark.py
python scripts/run_model_zoo.py
python scripts/prepare_transfer_ossl.py
python scripts/run_transfer_benchmark.py
python scripts/export_transfer_workbook.py
streamlit run app.py
```

Persiapan pertama men-stream file MIR OSSL sekitar 420 MB dan membuat turunan lokal berisi tepat 1.000 sampel. File mentah tidak disimpan atau dimasukkan Git. Dataset kecil yang dihasilkan berada di `data/processed/` bersama manifest sumber, pemilihan sampel, metode, dan lisensi.

Benchmark default menjalankan repeated nested cross-validation dan dapat memerlukan beberapa menit. Untuk eksperimen cepat dari Python:

```python
from pathlib import Path
from npk_spectra.benchmark import run_benchmark

run_benchmark(Path.cwd(), budgets=(60,), outer_repeats=1, n_jobs=1, fit_final=False)
```

## Self-attention ringan

Dashboard juga mendukung patch transformer gabungan untuk N/P/K. Spektrum 1.701 kanal dibagi menjadi 107 patch berukuran 16 kanal, ditambah satu token `CLS`, lalu diproses oleh 3 encoder layer (`d_model=64`, 4 head, feed-forward 128). Totalnya hanya **108.803 parameter**, sehingga layak dilatih pada MacBook Pro M1 8 GB melalui backend MPS.

```bash
python scripts/run_attention_benchmark.py --repeats 1 --epochs 80
```

Perintah di atas menjalankan pilot 5-fold untuk setiap budget. Untuk menyamai benchmark klasik 2×5-fold, hilangkan opsi atau gunakan `--repeats 2 --epochs 150`. Early stopping memakai validation split yang hanya berasal dari outer-training fold. Hasil disimpan sebagai `artifacts/attention_*.csv`, checkpoint final sebagai `artifacts/attention_model.pt`, dan muncul otomatis sebagai opsi model di dashboard.

## Small-data model zoo

Enam family tambahan menguji nonlinearitas tanpa memberikan 1.701 kanal mentah langsung ke model berkapasitas tinggi:

| Family | Representasi | Karakter |
|---|---|---|
| RBF-SVR | 10 PLS scores | Kernel nonlinear untuk data kecil |
| Cubist | 10 PLS scores | Rules + local linear regressions |
| Extra Trees | 15 PLS scores | Randomized tree ensemble |
| Gaussian Process | 20 PCA scores | Smooth probabilistic regression |
| Cascade Forest | 10 PLS scores | Dua layer RF + ExtraTrees dengan OOF augmentation |
| Tiny 1D CNN | Spektrum SNV | Local convolutions, joint N/P/K |

```bash
python scripts/run_model_zoo.py --repeats 1 --jobs -1
```

Run default memakai fixed, predeclared hyperparameters dan 5 outer folds per budget. Cascade forest adalah implementasi pure scikit-learn bergaya gcForest karena wheel resmi `deep-forest` belum mendukung Python 3.12; feature augmentation antar-layer dibuat dari out-of-fold predictions. Artefak gabungan ditulis ke `artifacts/zoo_*.csv` dan checkpoint CNN ke `artifacts/cnn_model.pt`.

## Metodologi

- Subset nested 60/120/180/240/300/1000 berasal dari reservoir complete cases yang reproducible dan diurutkan dengan keragaman spektral serta target.
- Outer repeated 5-fold CV mengukur generalisasi; inner 3-fold CV memilih preprocessing dan model.
- Kandidat: PLS Regression pada SNV atau turunan pertama Savitzky–Golay, serta PCA 95% + Ridge.
- Kandidat neural: patch transformer kecil dengan SNV, position embedding, dan head regresi gabungan N/P/K.
- Transformasi `log1p` target terjadi di dalam estimator dan prediksi dikembalikan ke unit asli.
- MAE, RMSE, R², RPIQ, interval empiris, dan perbandingan baseline median dilaporkan dari outer folds.
- Seluruh preprocessing yang belajar dari populasi training berada di pipeline CV untuk mencegah leakage.

Tidak ada ambang R² yang dipaksakan. Jika model tidak mengalahkan baseline, itu adalah hasil PoC yang sah dan harus ditampilkan apa adanya.

## Struktur

```text
app.py                         Dashboard Streamlit
scripts/prepare_data.py        Join dan seleksi OSSL streaming
scripts/run_benchmark.py       Learning curve nested-CV
scripts/run_attention_benchmark.py  Learning curve self-attention
scripts/run_model_zoo.py       Enam nonlinear small-data benchmarks
src/npk_spectra/attention.py   Arsitektur, training, dan evaluasi neural
src/npk_spectra/model_zoo.py   SVR, Cubist, ExtraTrees, GPR, cascade forest
src/npk_spectra/cnn.py         Tiny 1D CNN joint N/P/K
src/npk_spectra/               Kontrak data, preprocessing, modeling
notebooks/crisp_dm_npk.ipynb   Audit CRISP-DM dari artefak yang sama
data/local_manifest_template.csv
tests/                         Unit dan smoke tests
```

## Migrasi ke 300 sampel lokal

1. Tetapkan metode dan unit laboratorium N/P/K sebelum sampling.
2. Isi `data/local_manifest_template.csv`; satu baris harus mewakili satu titik sampel independen.
3. Buat loader spektrum Shimadzu yang menghasilkan absorbance pada grid bersama tanpa mengubah kode benchmark.
4. Gunakan grouped CV berdasarkan lokasi, plot, atau batch sampling—bukan random split biasa—agar sampel yang sangat berdekatan tidak bocor antar-fold.
5. Uji similarity/domain coverage sebelum model digunakan pada wilayah atau instrumen baru.

## Pengujian

```bash
pytest
python -m compileall src app.py
```

Sumber: [deskripsi database OSSL](https://docs.soilspectroscopy.org/db-desc.html), [akses data](https://docs.soilspectroscopy.org/db-access.html), dan [peringatan/modeling framework OSSL](https://docs.soilspectroscopy.org/prediction-models.html).

## Transfer OSSL → Jawa Tengah

`scripts/run_transfer_benchmark.py` membandingkan hybrid CNN+attention yang dilatih dari nol pada data lokal, model OSSL zero-shot, dan model OSSL-first dengan head-only fine-tuning. Ia memasangkan 60 ekspor Shimadzu dari sibling project `../unsur-hara` dengan `NPK_Filled_Soil_Data_v2.xlsx`, lalu memakai leave-one-county-out untuk Grobogan, Pati, dan Rembang.

Jalankan `python scripts/prepare_transfer_ossl.py` sekali untuk menyiapkan pool sumber OSSL 10.000 complete cases di `data/ossl_transfer/`. Seleksi spectral-nearest 5.000 sampel dilakukan hanya terhadap dua kabupaten training pada setiap fold. P dan K tetap diberi peringatan metode provisional sampai SOP laboratorium lokal dikonfirmasi.

Setelah benchmark selesai, `python scripts/export_transfer_workbook.py` membuat `artifacts/NPK_Filled_Soil_Data_v2_with_transfer_results.xlsx`. Workbook asli tidak diubah; salinan tersebut menambahkan sheet `Transfer Summary`, `Fold Metrics`, `Transfer Predictions`, dan `Transfer Manifest`.

## Handoff untuk agent / mesin berikutnya

### Yang sudah tersedia di repository

- `NPK_Filled_Soil_Data_v2.xlsx`: target lab dan metadata lokal.
- Pipeline transfer, exporter workbook, halaman Streamlit, serta `uv.lock` yang dapat direproduksi.
- Data OSSL transfer besar dan artefak benchmark **sengaja tidak di-commit**. Bangun kembali di mesin tujuan.

### Prasyarat data lokal

Clone atau salin sibling project `unsur-hara` sehingga ekspor Shimadzu berada di:

```text
../unsur-hara/sample-bu-Yayuk-0303/Bu Yayuk-0303/{GB,PT,RB}/2-*.txt
```

Pipeline mencocokkan nama file tersebut dengan kode `GB/PT/RB <nomor>` pada kolom `Komoditas pada lahan` di workbook. Hasil yang benar adalah 60 pasangan spektrum–lab: 20 Grobogan, 20 Pati, dan 20 Rembang.

### Menjalankan pada RTX 4060

```bash
uv sync --extra dev
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
uv run python scripts/prepare_transfer_ossl.py
uv run python scripts/run_transfer_benchmark.py --device cuda --epochs 80
uv run python scripts/export_transfer_workbook.py
uv run streamlit run app.py
```

`prepare_transfer_ossl.py` mengunduh dan menulis `data/ossl_transfer/` (~58 MB terkompresi). Benchmark menulis `artifacts/transfer_*.csv` dan manifest; exporter lalu membuat workbook hasil yang dapat diunduh dari halaman **Transfer Lokal**.

### Guardrails penelitian

- Evaluasi utama adalah leave-one-county-out; jangan mengganti dengan random split sebagai angka headline.
- Pilih 5.000 OSSL terdekat hanya dari dua kabupaten training pada setiap fold.
- N/P/K lokal tetap memakai label metode provisional. Jangan menyatakan P Olsen atau K NH4OAc setara dengan SOP lokal sebelum metode laboratoriumnya dikonfirmasi.
