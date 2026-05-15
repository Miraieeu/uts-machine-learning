# 🧠 Machine Learning — Ujian Tengah Semester (UTS)

Repository ini berisi tiga project Machine Learning yang mencakup **tiga paradigma utama ML**: Klasifikasi, Regresi, dan Clustering. Seluruh project dikerjakan menggunakan **Python** di platform **Kaggle Notebook**.

---

## 📁 Daftar Project

| # | Notebook | Jenis Masalah | Dataset | Model Utama |
|---|----------|---------------|---------|-------------|
| 1 | [`uts-01.ipynb`](uts-01.ipynb) | Klasifikasi Biner | Train Transaction Detection | LightGBM |
| 2 | [`uts-02.ipynb`](uts-02.ipynb) | Regresi | Million Song Dataset (subset) | LightGBM |
| 3 | [`uts-03.ipynb`](uts-03.ipynb) | Clustering | Credit Card Customer Data | K-Means |

---

## 🔴 Project 1 — Train Transaction (Klasifikasi)

**Notebook:** [`uts-01.ipynb`](uts-01.ipynb)

### Deskripsi
Membangun model untuk mendeteksi **transaksi penipuan (fraud)** pada data transaksi e-commerce. Dataset bersumber dari kompetisi [Train Transaction Detection](https://www.kaggle.com/c/ieee-fraud-detection) di Kaggle.

### Tantangan Utama
- Dataset sangat besar (~590K baris × 394 kolom)
- **Class imbalance** yang ekstrem — hanya **3.5%** transaksi fraud
- Banyak missing values (beberapa kolom >90% kosong)

### Pipeline
1. **Data Loading** — `train_transaction.csv` & `test_transaction.csv`
2. **EDA Minimal** — distribusi target & analisis missing values
3. **Feature Engineering** — log amount, waktu transaksi (hari/jam), parsing email domain
4. **Feature Selection** — buang kolom missing >90% dan kolom konstant
5. **Preprocessing** — median imputation (numerik), OrdinalEncoder (kategorikal)
6. **Baseline Model** — Logistic Regression (`class_weight="balanced"`)
7. **Model Utama** — LightGBM dengan `scale_pos_weight` untuk handling imbalance
8. **Hyperparameter Tuning** — Optuna (20 trials)
9. **Experiment Tracking** — MLflow
10. **Submission** — pembuatan `submission.csv`

### Hasil

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression (baseline) | 0.8636 |
| **LightGBM + Optuna** | **0.9689** |

---

## 🟢 Project 2 — Prediksi Tahun Rilis Lagu (Regresi)

**Notebook:** [`uts-02.ipynb`](uts-02.ipynb)

### Deskripsi
Memprediksi **tahun rilis** sebuah lagu berdasarkan **fitur karakteristik sinyal audio** (timbre — frekuensi, pitch, amplitudo). Dataset merupakan subset dari [Million Song Dataset](http://millionsongdataset.com/).

### Detail Dataset
- **515.345 baris × 91 kolom** (1 target + 90 fitur audio)
- Target: tahun rilis (contoh: 2001)
- Fitur: nilai-nilai numerik hasil ekstraksi sinyal audio

### Pipeline
1. **Data Loading** — CSV tanpa header
2. **Data Cleaning** — median imputation
3. **Outlier Handling** — IQR clipping
4. **Feature Selection** — korelasi absolut >0.05 → 42 fitur terpilih dari 90
5. **Scaling** — RobustScaler
6. **Model** — LightGBM Regressor
7. **Hyperparameter Tuning** — Optuna (20 trials) + early stopping
8. **Experiment Tracking** — MLflow (SQLite backend)
9. **Interpretasi Model** — LIME (Local Interpretable Model-agnostic Explanations)

### Hasil

| Metrik | Nilai |
|--------|-------|
| **RMSE** | **9.10** |
| Best Params | `n_estimators=937, lr=0.067, max_depth=7, num_leaves=99` |

---

## 🔵 Project 3 — Segmentasi Pelanggan Kartu Kredit (Clustering)

**Notebook:** [`uts-03.ipynb`](uts-03.ipynb)

### Deskripsi
Mengelompokkan **pelanggan kartu kredit** ke dalam segmen-segmen berdasarkan **perilaku transaksi** menggunakan algoritma **K-Means**. Tujuannya adalah memahami pola perilaku pelanggan untuk strategi bisnis yang lebih targeted.

### Detail Dataset
- **8.950 pelanggan × 17 fitur**
- Fitur utama: `BALANCE`, `PURCHASES`, `CASH_ADVANCE`, `CREDIT_LIMIT`, `PAYMENTS`, `MINIMUM_PAYMENTS`, `TENURE`, dll.

### Pipeline
1. **Data Loading** — drop `CUST_ID` (bukan fitur perilaku)
2. **Data Cleaning** — median imputation
3. **Outlier Handling** — IQR clipping (penting karena K-Means sensitif terhadap outlier)
4. **Scaling** — StandardScaler (wajib karena K-Means berbasis jarak Euclidean)
5. **Pemilihan K Optimal** — Elbow Method (WCSS) + Silhouette Score, K=2 s/d 9
6. **Modeling** — K-Means (K=4)
7. **Visualisasi** — PCA (2 komponen) untuk scatter plot 2D

### Hasil

| Parameter | Nilai |
|-----------|-------|
| **K Optimal** | **4 cluster** |
| Validasi | Silhouette Score tertinggi pada K=4 |

---

## 🛠️ Tech Stack

| Kategori | Tools |
|----------|-------|
| **Bahasa** | Python 3.12 |
| **Platform** | Kaggle Notebook |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, LightGBM |
| **Hyperparameter Tuning** | Optuna |
| **Experiment Tracking** | MLflow |
| **Interpretability** | LIME |
| **Visualisasi** | Matplotlib, Seaborn |
| **Dimensionality Reduction** | PCA (Scikit-learn) |

---

## 🚀 Cara Menjalankan

### Opsi 1: Kaggle (Recommended)
1. Upload notebook ke [Kaggle](https://www.kaggle.com/)
2. Tambahkan dataset yang sesuai ke notebook
3. Jalankan seluruh cell

### Opsi 2: Lokal
```bash
# Clone repository
git clone https://github.com/<username>/ml-uts.git
cd ml-uts

# Install dependencies
pip install numpy pandas scikit-learn lightgbm optuna mlflow lime matplotlib seaborn

# Jalankan notebook
jupyter notebook
```
---

## 📊 Ringkasan Perbandingan

```
┌─────────────┬──────────────────┬──────────────────┬──────────────────┐
│   Aspek      │   UTS-01 (Klasifikasi) │  UTS-02 (Regresi)│ UTS-03 (Cluster) │
├─────────────┼──────────────────┼──────────────────┼──────────────────┤
│ Masalah      │ Klasifikasi      │ Regresi          │ Clustering       │
│ Dataset      │ ~590K × 394      │ ~515K × 91       │ ~9K × 17         │
│ Model        │ LightGBM         │ LightGBM         │ K-Means          │
│ Tuning       │ Optuna           │ Optuna           │ Elbow+Silhouette │
│ Metrik       │ ROC-AUC: 0.969   │ RMSE: 9.1        │ K=4 clusters     │
│ Tracking     │ MLflow           │ MLflow           │ —                │
└─────────────┴──────────────────┴──────────────────┴──────────────────┘
```

---

## Identitas

Nama : Faris Aryaputra
NIM  : 101032300067
Kelas: TK-47-GAB0102