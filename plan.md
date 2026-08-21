# Plan Riset: Transfer Learning FTIR Tanah dari OSSL ke Data Lokal Jawa

## 1. Tujuan Utama

Membangun model prediksi unsur hara tanah dari spektrum FTIR dengan pendekatan **transfer learning**, memanfaatkan data besar dari **Open Soil Spectral Library (OSSL)** sebagai source domain, kemudian melakukan **fine-tuning menggunakan data lokal** yang jumlahnya terbatas.

Target utama:

- Prediksi **N**
- Prediksi **P**
- Prediksi **K**

Model utama yang direncanakan:

> **Hybrid 1D-CNN + Small Self-Attention**

Model ini dipilih karena:

- 1D-CNN cocok untuk menangkap pola lokal pada spektrum seperti peak, shoulder, slope, dan bentuk pita absorpsi.
- Self-attention digunakan untuk mempelajari hubungan antar-region spektrum yang berjauhan.
- Model dapat di-pretrain pada data OSSL yang besar, kemudian diadaptasi ke data lokal dengan jumlah sampel terbatas.

---

## 2. Konsep Utama

Alur utama:

```text
OSSL
  ↓
pilih MIR spectra + target N/P/K
  ↓
pilih subset OSSL yang relevan dengan domain lokal
  ↓
harmonisasi dan preprocessing spektrum
  ↓
pretrain Hybrid 1D-CNN + Self-Attention
  ↓
model pretrained
  ↓
freeze sebagian besar layer
  ↓
fine-tune layer akhir menggunakan data lokal
  ↓
model lokal
  ↓
evaluasi pada data lokal yang benar-benar held-out
```

Prinsipnya:

> Model tidak belajar dari nol menggunakan data lokal yang sedikit.  
> Model terlebih dahulu belajar representasi umum spektrum tanah dari OSSL, kemudian pengetahuan tersebut disesuaikan dengan karakteristik tanah, instrumen, dan kondisi laboratorium lokal.

---

## 3. Dataset

### 3.1 Source Domain — OSSL

Sumber:

- Open Soil Spectral Library
- Dokumentasi: https://docs.soilspectroscopy.org/

Data yang dibutuhkan:

- MIR / FTIR spectra
- N laboratory measurement
- P laboratory measurement
- K laboratory measurement
- metadata sampel
- metadata dataset / laboratory / instrument jika tersedia

Catatan penting:

- OSSL merupakan **open data**.
- Lisensi dan provenance masing-masing contributing dataset harus tetap dicatat.
- Tidak semua sampel OSSL memiliki N, P, dan K secara lengkap.
- Metode laboratorium untuk N, P, dan K harus diperiksa karena satu variabel dapat diukur menggunakan metode kimia yang berbeda.

---

### 3.2 Target Domain — Data Lokal Jawa

Data awal:

- sekitar **60 sampel**
- lokasi:
  - Grobogan
  - Pati
  - Blora
- instrumen:
  - Shimadzu IRSpirit-TX
- spektrum:
  - sekitar 400–4000 cm⁻¹
  - sekitar 5000 titik per sampel
  - format awal `%T`

Data target yang diperlukan:

- nilai laboratorium N
- nilai laboratorium P
- nilai laboratorium K

Data lokal ini akan digunakan sebagai:

1. domain reference untuk mencari data OSSL yang relevan,
2. fine-tuning / local calibration,
3. evaluasi generalisasi model.

---

## 4. Harmonisasi Spektrum

OSSL dan data lokal berasal dari sistem pengukuran yang berbeda.

Contoh perbedaan:

| Aspek | OSSL | Data Lokal |
|---|---|---|
| Instrument | beberapa sumber/instrumen | Shimadzu IRSpirit-TX |
| Spectral range | standardized MIR | sekitar 400–4000 cm⁻¹ |
| Grid | standardized | sekitar 5000 titik |
| Intensity | absorbance | % transmittance |
| Lab/SOP | multi-source | SOP lokal |

Karena itu, sebelum transfer learning dilakukan, data harus diharmonisasi.

### 4.1 Konversi %T ke Absorbance

Untuk data lokal:

```text
A = -log10(T / 100)
```

dengan:

- `T` = transmittance dalam persen
- `A` = absorbance

### 4.2 Common Spectral Range

Gunakan spectral range yang kompatibel antara OSSL dan data lokal.

Rencana awal:

```text
600–4000 cm⁻¹
```

### 4.3 Resampling

Data lokal diinterpolasi ke grid yang sama dengan OSSL.

Contoh:

```text
600, 602, 604, ..., 4000 cm⁻¹
```

Jika menggunakan interval 2 cm⁻¹:

```text
1701 spectral features
```

### 4.4 Preprocessing yang Akan Dibandingkan

Kandidat:

- raw absorbance
- Standard Normal Variate (SNV)
- normalization
- detrending
- Savitzky-Golay smoothing
- first derivative
- second derivative

Preprocessing tidak dipilih berdasarkan asumsi, tetapi berdasarkan hasil cross-validation.

---

## 5. Pemilihan Data OSSL yang Relevan

Tidak langsung menggunakan seluruh OSSL secara buta.

Tujuan:

> mencari bagian dari OSSL yang paling relevan terhadap domain spektrum lokal.

### 5.1 Spectral Similarity

Gunakan spektrum lokal sebagai query terhadap OSSL.

Metode kandidat:

- PCA distance
- Euclidean distance pada latent space
- Mahalanobis distance
- Spectral Angle Mapper
- nearest-neighbor search pada spectral embedding

Alur:

```text
OSSL MIR
   +
local FTIR
   ↓
same preprocessing
   ↓
common spectral representation
   ↓
PCA / embedding
   ↓
measure spectral similarity
   ↓
select relevant OSSL subset
```

### 5.2 Jangan Menganggap Spectral Similarity = Same Soil Condition

Spectral similarity berarti:

> spektrum MIR memiliki fingerprint yang mirip.

Ini tidak otomatis berarti:

- kondisi tanah sama,
- metode pemupukan sama,
- kadar N/P/K sama,
- parent material sama,
- microbiology sama.

Karena itu, spectral similarity sebaiknya dikombinasikan dengan metadata seperti:

- pH
- SOC / organic carbon
- clay
- sand
- silt
- soil class / mineralogy jika tersedia
- climate / geography jika relevan

---

## 6. Mencegah Data Leakage

Pemilihan OSSL berdasarkan spectral similarity merupakan bagian dari **training pipeline**.

Karena itu, held-out test data tidak boleh digunakan untuk memilih OSSL subset.

Contoh evaluasi Leave-One-Location-Out:

```text
Fold 1:

Training local:
Grobogan + Pati
        ↓
gunakan HANYA data training untuk mencari OSSL yang mirip
        ↓
pretraining / adaptation
        ↓
test:
Blora
```

Blora tidak boleh ikut:

- memilih OSSL subset,
- menentukan preprocessing,
- memilih hyperparameter,
- fine-tuning.

Kemudian ulangi:

```text
Train Grobogan + Blora → Test Pati

Train Pati + Blora → Test Grobogan
```

---

## 7. Arsitektur Model Utama

### Hybrid 1D-CNN + Small Self-Attention

Konsep:

```text
FTIR spectrum
     ↓
Conv1D Block 1
     ↓
Conv1D Block 2
     ↓
Conv1D Block 3
     ↓
Small Self-Attention
     ↓
Global Pooling
     ↓
Regression Head
     ↓
N / P / K
```

Peran masing-masing:

### 7.1 1D-CNN

Belajar local spectral features:

- peaks
- shoulders
- slopes
- local curvature
- broad absorptions
- neighboring band relationships

### 7.2 Self-Attention

Belajar hubungan antar spectral regions yang berjauhan.

Contoh:

```text
feature sekitar 1000 cm⁻¹
        ↕
feature sekitar 1600 cm⁻¹
        ↕
feature sekitar 3400 cm⁻¹
```

### 7.3 Regression Head

Mengubah learned spectral representation menjadi nilai numerik:

```text
N
P
K
```

---

## 8. Phase 1 — Pretraining pada OSSL

Semua layer model terlebih dahulu dilatih menggunakan OSSL.

Contoh konseptual model 5 bagian:

```text
Layer 1 / CNN Block 1       ✅ train
Layer 2 / CNN Block 2       ✅ train
Layer 3 / CNN Block 3       ✅ train
Layer 4 / Self-Attention    ✅ train
Layer 5 / Regression Head   ✅ train
```

Tujuan pretraining:

- belajar generic spectral patterns,
- belajar representation tanah,
- belajar hubungan awal antara spectral representation dan target N/P/K.

Output:

```text
pretrained model weights
```

---

## 9. Phase 2 — Fine-Tuning dengan Data Lokal

Setelah pretraining, model diadaptasi ke domain lokal.

### Stage A — Head-Only Fine-Tuning

Awalnya model dibuat konservatif:

```text
CNN Block 1       🔒 frozen
CNN Block 2       🔒 frozen
CNN Block 3       🔒 frozen
Self-Attention    🔒 frozen
Regression Head   🔓 trainable
```

Hanya prediction head yang dilanjutkan training menggunakan data lokal.

Tujuan:

> mempertahankan generic spectral knowledge dari OSSL dan hanya menyesuaikan mapping akhir terhadap domain lokal.

---

### Stage B — Partial Fine-Tuning

Jika Stage A belum cukup:

```text
CNN Block 1       🔒 frozen
CNN Block 2       🔒 frozen
CNN Block 3       🔒 / 🔓 experiment
Self-Attention    🔓 trainable
Regression Head   🔓 trainable
```

Gunakan learning rate lebih kecil dibanding training awal.

Contoh konseptual:

```text
Regression Head:
learning rate lebih besar

Attention / CNN akhir:
learning rate sangat kecil

CNN awal:
frozen
```

---

### Stage C — Full Fine-Tuning

Hanya dilakukan sebagai eksperimen jika data lokal sudah cukup.

```text
CNN Block 1       🔓
CNN Block 2       🔓
CNN Block 3       🔓
Self-Attention    🔓
Regression Head   🔓
```

Dengan sekitar 60 sampel, full fine-tuning berisiko tinggi menyebabkan overfitting.

Karena itu bukan strategi utama.

---

## 10. Instrument / Hardware Domain Shift

Perbedaan instrumen dianggap sebagai bagian penting dari penelitian.

Transfer yang sebenarnya terjadi adalah:

```text
OSSL soil
+
OSSL instruments
+
OSSL laboratories
        ↓
        ↓ transfer learning
        ↓
Central Java soil
+
Shimadzu IRSpirit-TX
+
local laboratory SOP
```

Risiko:

- baseline shift
- amplitude differences
- resolution differences
- detector response
- spectral alignment
- noise differences
- sample preparation differences

Mitigasi:

- common spectral grid
- absorbance conversion
- preprocessing
- spectral normalization
- augmentation yang realistis
- local fine-tuning
- strict laboratory SOP

---

## 11. Spectral Augmentation untuk Pretraining

Untuk meningkatkan robustness terhadap perbedaan alat, pretraining dapat menggunakan spectral augmentation ringan.

Contoh:

```text
original spectrum
      ↓
small random noise
small scale change
small baseline shift
small spectral shift
      ↓
augmented spectrum
```

Secara konseptual:

```text
x' = a*x + b + noise
```

Semua augmentasi harus tetap berada pada range yang masuk akal secara fisik.

Tujuan:

> membuat encoder tidak terlalu tergantung pada karakteristik spesifik satu instrumen.

---

## 12. Baseline Model

Deep learning tidak boleh diuji sendirian.

Model pembanding:

### Classical Chemometrics

- PLSR

### Classical Machine Learning

- Ridge Regression
- Elastic Net
- SVR

### Transfer Baseline

- PLSR local only
- PLSR OSSL only
- PLSR + local spiking
- localized OSSL + PLSR

### Deep Learning

- local 1D-CNN from scratch
- OSSL pretrained 1D-CNN
- OSSL pretrained 1D-CNN + attention
- head-only fine-tuning
- partial fine-tuning

---

## 13. Eksperimen Utama

Minimal eksperimen:

| Experiment | Source Data | Local Adaptation |
|---|---|---|
| PLSR Local | Local | none |
| PLSR OSSL | OSSL | none |
| PLSR Spiked | OSSL + Local | recalibration |
| Local CNN | Local | trained from scratch |
| OSSL CNN | OSSL | zero-shot |
| OSSL CNN + Head FT | OSSL | head fine-tuning |
| OSSL CNN + Partial FT | OSSL | partial fine-tuning |
| OSSL CNN+Attention + FT | OSSL | partial fine-tuning |

Pertanyaan yang ingin dijawab:

> Apakah pretraining pada large soil spectral library dapat meningkatkan prediksi N/P/K ketika hanya tersedia sedikit data kalibrasi lokal?

---

## 14. Learning Curve Experiment

Karena masalah utama adalah jumlah data lokal yang terbatas, lakukan eksperimen:

```text
10 local samples
20 local samples
30 local samples
40 local samples
50 local samples
60 local samples
```

Untuk setiap jumlah sampel:

1. fine-tune model,
2. evaluasi pada held-out local test data,
3. catat RMSE, MAE, R².

Output:

```text
number of local samples
        ↓
prediction error
```

Tujuan:

> mengukur seberapa banyak data lokal yang benar-benar dibutuhkan setelah model mendapat pretraining dari OSSL.

Eksperimen ini juga dapat digunakan untuk merencanakan pengambilan sampel berikutnya.

---

## 15. Evaluation Strategy

Karena data lokal terbatas dan berasal dari beberapa lokasi, gunakan grouped evaluation.

Strategi utama:

### Leave-One-Location-Out

```text
GB + PT → test RB
GB + RB → test PT
PT + RB → test GB
```

Jika hyperparameter tuning dibutuhkan:

```text
Outer CV:
Leave-One-Location-Out

Inner CV:
CV pada training locations
```

Ini merupakan bentuk nested cross-validation.

---

## 16. Metrics

Untuk setiap target N, P, dan K:

- RMSE
- MAE
- R²
- Bias

Tambahan:

- standard deviation antar-fold
- learning curve
- prediction vs laboratory plot
- residual plot
- domain similarity / representativeness score

---

## 17. Target Strategy

N, P, dan K tidak boleh diperlakukan sebagai label generik tanpa memperhatikan metode laboratorium.

Harus ditentukan:

```text
N = metode laboratorium apa?
P = metode laboratorium apa?
K = metode laboratorium apa?
```

Contoh:

- total N
- available P
- exchangeable / available K

Pemilihan field OSSL harus kompatibel dengan metode laboratorium lokal sebisa mungkin.

Jika metode berbeda secara prinsip, model tidak boleh menganggap target tersebut identik tanpa justifikasi.

---

## 18. Tahapan CRISP-DM

### 18.1 Business Understanding

Tujuan:

> memprediksi unsur hara tanah dari FTIR menggunakan model yang dapat bekerja dengan jumlah data lokal terbatas.

Pertanyaan riset utama:

> Apakah pretraining menggunakan OSSL dan fine-tuning menggunakan sedikit data lokal dapat meningkatkan performa dibanding model yang hanya menggunakan data lokal?

---

### 18.2 Data Understanding

- audit OSSL
- audit data lokal
- visualisasi spektrum
- cek missing values
- cek distribusi target
- cek lokasi
- cek perbedaan spectral grid
- PCA
- outlier analysis
- domain similarity analysis

---

### 18.3 Data Preparation

- %T → absorbance
- spectral cropping
- interpolation
- common grid
- preprocessing
- target filtering
- OSSL localization
- train / validation / test grouping

---

### 18.4 Modeling

Bangun:

1. PLSR
2. SVR
3. 1D-CNN
4. Hybrid 1D-CNN + Self-Attention
5. pretrained model
6. fine-tuned model

---

### 18.5 Evaluation

Bandingkan:

- local only
- OSSL zero-shot
- OSSL + local adaptation
- classical ML
- deep learning

Gunakan held-out location evaluation.

---

### 18.6 Deployment

Target akhir:

```text
Streamlit App
```

Input:

- Shimadzu `.txt`

Pipeline:

```text
upload spectrum
      ↓
parse
      ↓
%T → absorbance
      ↓
resample
      ↓
same preprocessing
      ↓
trained model
      ↓
predict
      ↓
N / P / K
```

Output:

- prediksi N
- prediksi P
- prediksi K
- plot spektrum
- confidence / uncertainty jika tersedia
- warning jika spectrum berada di luar calibration domain

---

## 19. Struktur Repository yang Direncanakan

```text
riset-unsur-hara/
│
├── README.md
├── plan.md
│
├── data/
│   ├── local/
│   ├── ossl/
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_spectral_harmonization.ipynb
│   ├── 03_ossl_localization.ipynb
│   ├── 04_baseline_plsr.ipynb
│   ├── 05_cnn_pretraining.ipynb
│   ├── 06_local_finetuning.ipynb
│   └── 07_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   ├── similarity.py
│   ├── models.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   ├── pretrained/
│   └── finetuned/
│
└── app/
    └── streamlit_app.py
```

---

## 20. Milestone

### Milestone 1 — Data Understanding

- [ ] Download / prepare OSSL MIR dataset
- [ ] Identify compatible N/P/K fields
- [ ] Load local Shimadzu spectra
- [ ] Plot local spectra
- [ ] Convert %T to absorbance
- [ ] Harmonize spectral range
- [ ] PCA and outlier analysis

### Milestone 2 — Domain Analysis

- [ ] Compare OSSL vs local spectral space
- [ ] Measure spectral similarity
- [ ] Identify OSSL subset closest to local domain
- [ ] Inspect metadata of nearest OSSL samples
- [ ] Quantify representativeness

### Milestone 3 — Classical Baseline

- [ ] PLSR local
- [ ] PLSR OSSL
- [ ] PLSR localized OSSL
- [ ] PLSR spiking / recalibration
- [ ] SVR baseline

### Milestone 4 — Deep Pretraining

- [ ] Implement 1D-CNN
- [ ] Add small self-attention
- [ ] Train on selected OSSL data
- [ ] Save pretrained weights
- [ ] Evaluate OSSL validation performance

### Milestone 5 — Local Fine-Tuning

- [ ] Head-only fine-tuning
- [ ] Partial fine-tuning
- [ ] Compare freeze strategies
- [ ] Evaluate Leave-One-Location-Out

### Milestone 6 — Learning Curve

- [ ] 10 samples
- [ ] 20 samples
- [ ] 30 samples
- [ ] 40 samples
- [ ] 50 samples
- [ ] 60 samples

### Milestone 7 — Final Model

- [ ] Select final preprocessing
- [ ] Select final architecture
- [ ] Train final model
- [ ] Save reproducible pipeline
- [ ] Document limitations

### Milestone 8 — Deployment

- [ ] Build Streamlit interface
- [ ] Shimadzu `.txt` parser
- [ ] Prediction output
- [ ] Spectrum visualization
- [ ] Domain warning
- [ ] Validation on unseen samples

---

## 21. Risiko Utama

### Risiko 1 — OSSL tidak cukup representatif terhadap tanah lokal

Mitigasi:

- spectral similarity analysis
- localization
- local fine-tuning
- out-of-domain detection

### Risiko 2 — Instrument shift

Mitigasi:

- absorbance conversion
- common grid
- preprocessing
- data augmentation
- fine-tuning

### Risiko 3 — Target lab methods tidak kompatibel

Mitigasi:

- pilih target berdasarkan analytical method
- dokumentasikan definisi target
- jangan mencampur incompatible methods tanpa harmonisasi

### Risiko 4 — Local data terlalu sedikit

Mitigasi:

- pretrained encoder
- freezing
- regularization
- small architecture
- PLSR baseline
- grouped CV
- learning curve analysis

### Risiko 5 — Overfitting

Mitigasi:

- jangan full fine-tune sejak awal
- early stopping
- regularization
- nested / grouped CV
- small model
- compare against simple baselines

### Risiko 6 — Data leakage

Mitigasi:

- OSSL localization dilakukan hanya dengan training fold
- preprocessing fit hanya pada training fold
- held-out location tidak digunakan untuk tuning
- test set tidak digunakan untuk model selection

---

## 22. Hipotesis Awal

### H1

Pretraining pada OSSL akan menghasilkan representasi spektrum yang lebih baik dibanding melatih neural network dari nol menggunakan sekitar 60 data lokal.

### H2

Fine-tuning menggunakan data lokal akan meningkatkan performa dibanding OSSL zero-shot model karena adanya:

- soil domain shift
- instrument shift
- laboratory/SOP shift

### H3

Hybrid 1D-CNN + small self-attention yang di-pretrain pada OSSL dan di-fine-tune pada data lokal akan mengungguli local-only deep learning.

### H4

PLSR tetap menjadi baseline yang kompetitif pada jumlah data lokal yang sangat kecil.

### H5

Jumlah local calibration samples yang dibutuhkan dapat dikurangi dengan memanfaatkan pretrained spectral representation.

---

## 23. Proposed Research Question

Versi umum:

> **Can pretraining on a large open soil spectral library improve FTIR-based soil nutrient prediction under limited local calibration data?**

Versi lebih spesifik:

> **Can a hybrid 1D-CNN–self-attention model pretrained on the Open Soil Spectral Library and fine-tuned with limited Central Java FTIR samples improve the prediction of soil N, P, and K compared with local-only calibration?**

---

## 24. Prinsip Penting

1. OSSL digunakan sebagai **source knowledge**, bukan dianggap sebagai pengganti data lokal.
2. Data lokal tetap menjadi ground truth utama untuk deployment lokal.
3. Spectral similarity tidak otomatis berarti kondisi tanah identik.
4. OSSL subset selection merupakan bagian dari training pipeline.
5. Test data tidak boleh ikut menentukan OSSL subset.
6. Hardware/instrument difference diperlakukan sebagai domain shift.
7. PLSR wajib menjadi baseline.
8. Deep learning harus membuktikan manfaatnya terhadap baseline sederhana.
9. Fine-tuning dimulai secara konservatif dengan freezing.
10. Semua keputusan preprocessing dan model harus ditentukan berdasarkan validation, bukan test set.

---

## 25. Ringkasan Singkat

```text
OSSL
 │
 │ large open soil spectral dataset
 ↓
select relevant MIR + compatible N/P/K
 ↓
localize using training-domain spectra
 ↓
harmonize spectra
 ↓
PRETRAIN
Hybrid 1D-CNN + Self-Attention
 ↓
generic soil spectral knowledge
 ↓
FREEZE early layers
 ↓
FINE-TUNE final layers
using limited local Shimadzu samples
 ↓
LOCALIZED MODEL
 ↓
Leave-One-Location-Out evaluation
 ↓
compare against PLSR / SVR / local-only CNN
 ↓
final model
 ↓
Streamlit deployment
```

---

## Status

**Current stage:** Research planning / CRISP-DM Business Understanding + Data Understanding preparation.

**Next recommended task:** audit OSSL targets and build the spectral harmonization + similarity analysis notebook before any model training.
