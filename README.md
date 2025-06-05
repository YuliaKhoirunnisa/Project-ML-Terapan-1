# Laporan Proyek Machine Learning - Yulia Khoirunnisa
## Domain Proyek
Penyakit kardiovaskular (CVD) merupakan penyebab utama kematian global, menyumbang sekitar 32% dari seluruh kematian di dunia, atau sekitar 17,9 juta orang setiap tahunnya [1]. Penyakit ini mencakup gangguan pada jantung dan pembuluh darah, seperti penyakit jantung koroner, penyakit serebrovaskular, dan hipertensi. Banyak faktor risiko yang berkontribusi terhadap terjadinya penyakit ini, termasuk hipertensi, hiperglikemia, hiperkolesterolemia, merokok, obesitas, dan gaya hidup tidak aktif [2].

Deteksi dini terhadap risiko CVD sangat penting untuk menurunkan angka morbiditas dan mortalitas. Namun, metode konvensional seperti pemeriksaan laboratorium atau imaging tidak hanya mahal tetapi juga membutuhkan infrastruktur medis yang memadai, yang seringkali tidak tersedia di wilayah dengan sumber daya terbatas. Oleh karena itu, diperlukan pendekatan berbasis data yang lebih efisien, cepat, dan terjangkau. Machine learning (ML) menawarkan solusi potensial dengan kemampuan untuk menganalisis data klinis secara lebih cepat dan akurat, sehingga memungkinkan identifikasi individu berisiko tinggi bahkan sebelum timbulnya gejala klinis [3].

Dalam proyek ini, saya membangun model klasifikasi machine learning menggunakan dataset Cardiovascular Disease dari Kaggle, dengan tujuan untuk memprediksi risiko CVD berdasarkan data klinis seperti usia, tekanan darah, kadar kolesterol, gaya hidup, dan faktor lainnya. Dengan model prediktif ini, diharapkan dapat membantu tenaga medis dalam pengambilan keputusan klinis secara lebih tepat, cepat, dan berbasis data.

### Referensi:
[1] World Health Organization. (2021). Cardiovascular diseases (CVDs). Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

[2] Mendis, S., Puska, P., & Norrving, B. (2011). Global Atlas on Cardiovascular Disease Prevention and Control. Geneva: World Health Organization.

[3] Dilsizian, S. E., & Siegel, E. L. (2014). Artificial intelligence in medicine and cardiac imaging: Harnessing big data and advanced computing to provide personalized medical diagnosis and treatment. Current Cardiology Reports, 16(1), 441.
## Business Understanding
### Problem Statements
- **Pernyataan Masalah 1**: Penyakit kardiovaskular merupakan penyebab utama kematian di dunia, namun metode deteksi dini yang tersedia saat ini masih mahal dan tidak terjangkau bagi seluruh populasi.

- **Pernyataan Masalah 2**: Identifikasi faktor risiko CVD sulit dilakukan tanpa bantuan alat berbasis data yang efisien.

- **Pernyataan Masalah 3**: Belum tersedia model machine learning yang akurat dan efisien berbasis data klinis sederhana untuk memprediksi risiko penyakit kardiovaskular.

### Goals
- **Goal 1**: Mengembangkan model yang mampu prediksi risiko penyakit kardiovaskular dengan akurasi tinggi.

- **Goal 2**: Membandingkan efektivitas dua algoritma machine learning dalam klasifikasi risiko penyakit kardiovaskular.

- **Goal 3**: Menyediakan solusi prediksi dini yang lebih terjangkau dan mudah diakses sebagai alternatif deteksi konvensional.

### Solution Statements
- **Solusi 1**: Menggunakan Decision Tree Classifier sebagai baseline model karena sifatnya yang interpretable dan cepat dalam membangun pohon keputusan dari data klinis.

- **Solusi 2**: Menggunakan Random Forest Classifier dengan tuning hyperparameter menggunakan RandomizedSearchCV untuk meningkatkan performa model dari baseline.

- **Solution statements**: Model akan dievaluasi menggunakan Accuracy dan ROC-AUC Score untuk mengukur kinerja klasifikasi dan kemampuan membedakan antara kelas pasien yang sehat dan berisiko.

## Data Understanding
Dataset yang digunakan adalah **Cardiovascular Disease Dataset** dari [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset). Dataset ini berisi data klinis pasien yang terdiri dari berbagai parameter medis yang umum dikumpulkan dalam pemeriksaan kesehatan rutin.

### Variabel-variabel:
- `age`: usia pasien dalam satuan hari.
- `gender`: jenis kelamin (1: perempuan, 2: laki-laki).
- `height`: tinggi badan dalam sentimeter.
- `weight`: berat badan dalam kilogram.
- `ap_hi`: tekanan darah sistolik.
- `ap_lo`: tekanan darah diastolik.
- `cholesterol`: tingkat kolesterol (1: normal, 2: di atas normal, 3: jauh di atas normal).
- `gluc`: tingkat glukosa (1: normal, 2: di atas normal, 3: jauh di atas normal).
- `smoke`: status merokok (0: tidak, 1: ya).
- `alco`: konsumsi alkohol (0: tidak, 1: ya).
- `active`: aktivitas fisik (0: tidak aktif, 1: aktif).
- `cardio`: label target (0: tidak ada penyakit kardiovaskular, 1: ada penyakit kardiovaskular).

### Exploratory Data Analysis (EDA)
- Melihat distribusi umur pasien setelah dikonversi ke tahun.
- Visualisasi korelasi antar fitur menggunakan heatmap.
- Melihat distribusi kelas target (`cardio`) untuk memahami apakah ada imbalance.

## Data Preparation
- Penghapusan kolom `id` karena hanya identifier.
- Mengonversi `age` dari hari menjadi tahun.
- One-Hot Encoding pada variabel kategorikal `gender`, `cholesterol`, dan `gluc`.
- Standardisasi fitur numerik dengan StandardScaler.
- Membagi data menjadi 80% train dan 20% test untuk evaluasi model.
Tahapan ini diperlukan untuk membersihkan data dari noise, memudahkan interpretasi, mempercepat proses training, dan meningkatkan performa model.

## Modeling
### Model 1: Decision Tree Classifier
- **Kelebihan**: Mudah dipahami, cepat.
- **Kekurangan**: Cenderung overfitting.
- **Parameter**: default (`random_state=42`).

### Model 2: Random Forest Classifier (with RandomizedSearchCV)
- **Kelebihan**: Mengurangi overfitting, robust.
- **Kekurangan**: Training lebih lama.
- **Improvement**: Hyperparameter tuning dengan RandomizedSearchCV:
  - `n_estimators`: [100, 200, 300]
  - `max_depth`: [10, 20, 30, None]
  - `min_samples_split`: [2, 5, 10]
  - `min_samples_leaf`: [1, 2, 4]
  - Cross-validation 5-Fold, scoring menggunakan `accuracy`.
 
## Evaluation
### Metrik yang Digunakan
- **Accuracy**: Persentase prediksi benar.
  \[ \text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{Total}} \]
- **ROC-AUC Score**: Kemampuan membedakan antara kelas 0 dan 1.

### Hasil Evaluasi
| Model                          | Accuracy | ROC-AUC |
|---------------------------------|----------|---------|
| Decision Tree Classifier        | 62.9%    | 0.629   |
| Random Forest (RandomizedSearchCV) | 74.0%  | 0.802   |

### Kesimpulan
Model Random Forest Classifier dengan hyperparameter tuning dipilih sebagai model terbaik untuk prediksi risiko penyakit kardiovaskular karena menunjukkan performa terbaik pada data uji, berdasarkan nilai Accuracy dan ROC-AUC.
