# Laporan Proyek Machine Learning - Yulia Khoirunnisa
## Domain Proyek
Domain yang dipakai untuk proyek ini adalah penyakit **Cardiovascular Disease**
![kardiovascular](https://github.com/user-attachments/assets/cff4329c-61d2-45cb-89cf-5f41a0644962)

### Latar Belakang
Penyakit kardiovaskular (CVD) merupakan penyebab utama kematian global, menyumbang sekitar 32% dari seluruh kematian di dunia, atau sekitar 17,9 juta orang setiap tahunnya [1]. Penyakit ini mencakup gangguan pada jantung dan pembuluh darah, seperti penyakit jantung koroner, penyakit serebrovaskular, dan hipertensi. Banyak faktor risiko yang berkontribusi terhadap terjadinya penyakit ini, termasuk hipertensi, hiperglikemia, hiperkolesterolemia, merokok, obesitas, dan gaya hidup tidak aktif [2].

Deteksi dini terhadap risiko CVD sangat penting untuk menurunkan angka morbiditas dan mortalitas. Namun, metode konvensional seperti pemeriksaan laboratorium atau imaging tidak hanya mahal tetapi juga membutuhkan infrastruktur medis yang memadai, yang seringkali tidak tersedia di wilayah dengan sumber daya terbatas. Oleh karena itu, diperlukan pendekatan berbasis data yang lebih efisien, cepat, dan terjangkau. Machine learning (ML) menawarkan solusi potensial dengan kemampuan untuk menganalisis data klinis secara lebih cepat dan akurat, sehingga memungkinkan identifikasi individu berisiko tinggi bahkan sebelum timbulnya gejala klinis [3].

Dalam proyek ini, saya membangun model klasifikasi machine learning menggunakan dataset Cardiovascular Disease dari Kaggle, dengan tujuan untuk memprediksi risiko CVD berdasarkan data klinis seperti usia, tekanan darah, kadar kolesterol, gaya hidup, dan faktor lainnya. Dengan model prediktif ini, diharapkan dapat membantu tenaga medis dalam pengambilan keputusan klinis secara lebih tepat, cepat, dan berbasis data.

## Business Understanding
### Problem Statements
- **Pernyataan Masalah 1**: Penyakit kardiovaskular merupakan salah satu penyebab utama kematian di dunia, namun metode deteksi yang tersedia saat ini masih mahal dan tidak terjangkau bagi seluruh populasi.

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
Dataset yang digunakan adalah **Cardiovascular Disease Dataset** dari [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset). Dataset ini berisi **70,000 baris** dan **13 kolom** data klinis pasien yang terdiri dari berbagai parameter medis yang umum dikumpulkan dalam pemeriksaan kesehatan rutin.

![struktur dataset](https://github.com/user-attachments/assets/8238770c-c4fb-4252-b053-8c93bb22b620)


### **Kondisi Data Awal**
Berdasarkan hasil data.describe(), ditemukan beberapa potensi outlier, terutama pada kolom tekanan darah:
- ap_hi (Tekanan Darah Sistolik) memiliki nilai minimum negatif (-150) dan maksimum 16020, jauh di luar rentang normal.
- ap_lo (Tekanan Darah Diastolik) juga memiliki nilai minimum negatif (-70) dan maksimum 11000.

Nilai-nilai ekstrem ini menunjukkan adanya potensi anomali data, seperti kesalahan input. Namun, dalam proyek ini, tidak dilakukan penanganan outlier secara khusus agar menjaga keutuhan dataset dan menghindari penghilangan data secara agresif. Handling outlier seperti filtering atau winsorization dapat dipertimbangkan pada pengembangan model di tahap lanjutan.


### Variabel-variabel:
- `id`: Identifier unik untuk masing-masing pasien.
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
![distribusi target](https://github.com/user-attachments/assets/c7fa28ab-7f75-412a-8693-c713b6b988c3)

- Melihat distribusi kelas target (`cardio`) untuk memahami apakah ada imbalance

| Kelas `cardio` | Deskripsi                                | Jumlah Data | Persentase |
|---------------|------------------------------------------|-------------|------------|
| **0**         | Tidak memiliki penyakit kardiovaskular    | 35,021      | 50.03%     |
| **1**         | Memiliki penyakit kardiovaskular          | 34,979      | 49.97%     |


![distribusi usia pasien](https://github.com/user-attachments/assets/f519c9f2-b4e2-4662-a0dd-f66895d3ebfe)

- Melihat distribusi umur pasien setelah dikonversi ke tahun
  - Mayoritas pasien berada di usia produktif akhir (40–60 tahun)
  - Ini sesuai dengan fakta medis bahwa risiko penyakit kardiovaskular meningkat seiring bertambahnya usia, terutama setelah usia 40 tahun
  - Informasi ini penting untuk menentukan strategi pencegahan dan deteksi dini penyakit kardiovaskular


![heatmap](https://github.com/user-attachments/assets/ba1998cd-2309-462e-a1c9-99a04fb882ff)

- Visualisasi korelasi antar fitur menggunakan heatmap
  - Kolesterol (cholesterol) menunjukkan korelasi positif moderat dengan kadar glukosa (gluc) sebesar 0.45. Ini berarti pasien dengan kadar kolesterol tinggi cenderung memiliki kadar glukosa yang tinggi pula
  - Jenis kelamin (gender) berkorelasi positif dengan tekanan darah sistolik (ap_hi) sekitar 0.50, menunjukkan bahwa terdapat perbedaan tekanan darah yang cukup jelas antar gender
  - Usia (age_years) memiliki korelasi positif terhadap penyakit kardiovaskular (cardio) sebesar 0.24. Ini mendukung pemahaman bahwa risiko penyakit ini meningkat seiring bertambahnya usia
  - Kolesterol (cholesterol) juga berkorelasi positif terhadap target cardio dengan nilai 0.22, memperkuat fakta bahwa kadar kolesterol tinggi meningkatkan risiko penyakit kardiovaskular
  - Glukosa (gluc) memiliki korelasi positif lemah dengan cardio sebesar 0.089, namun tetap relevan dalam klasifikasi penyakit ini
  - Variabel lain seperti smoke, alco, dan active menunjukkan korelasi yang sangat rendah terhadap cardio, menandakan bahwa pengaruhnya terhadap prediksi risiko penyakit tidak terlalu besar
  - ID (id) tidak menunjukkan korelasi signifikan dengan fitur lain, yang memang diharapkan mengingat id hanya merupakan identifier unik
  - Korelasi antar fitur secara umum tergolong rendah, sehingga tidak ada indikasi multikolinearitas yang kuat dalam dataset ini — hal ini baik untuk model berbasis tree seperti Random Forest yang sensitif terhadap fitur redundant.

## Data Preparation
- Penghapusan kolom `id` karena hanya identifier.
- Mengonversi `age` dari hari menjadi tahun.
- One-Hot Encoding pada variabel kategorikal `gender`, `cholesterol`, dan `gluc`.
- Standardisasi fitur numerik dengan StandardScaler.
- Membagi data menjadi 80% train dan 20% test untuk evaluasi model.
Tahapan ini diperlukan untuk membersihkan data dari noise, memudahkan interpretasi, mempercepat proses training, dan meningkatkan performa model.

## Modeling
### Cara Kerja Algoritma

### Model 1: Decision Tree Classifier
Decision Tree membangun model berupa struktur pohon keputusan dengan memilih fitur yang paling baik memisahkan data berdasarkan Gini Impurity atau Entropy. Proses pemisahan dilakukan secara rekursif hingga data pada node homogen atau memenuhi kriteria penghentian.
- **Kelebihan**: Mudah dipahami, cepat.
- **Kekurangan**: Cenderung overfitting.
- **Parameter**: default (`random_state=42`).

### Model 2: Random Forest Classifier
Random Forest adalah metode ensemble yang menggabungkan banyak Decision Tree. Setiap pohon dilatih dengan data bootstrap sampling, dan subset fitur dipilih secara acak untuk split. Prediksi akhir menggunakan voting mayoritas dari semua pohon.
- **Kelebihan**: Mengurangi overfitting, robust.
- **Kekurangan**: Training lebih lama.

### Parameter Tuning (RandomizedSearchCV)
Parameter tuning dilakukan menggunakan RandomizedSearchCV dengan ruang pencarian:

| Parameter             | Distribusi Pencarian                        |
|-----------------------|---------------------------------------------|
| `n_estimators`         | randint(100, 300)                           |
| `max_depth`            | [10, 20, 30, None]                          |
| `min_samples_split`    | randint(2, 10)                              |
| `min_samples_leaf`     | randint(1, 4)                               |

Proses tuning mencari kombinasi parameter optimal berdasarkan akurasi menggunakan 5-Fold Cross-Validation.
 
## Evaluation
### Metrik yang Digunakan
- **Accuracy**: Persentase prediksi benar.
- **ROC-AUC Score**: Kemampuan membedakan antara kelas 0 dan 1.

  \\[  \text{Accuracy} = \frac{TP + TN}{\text{Total}}  \\]

### Hasil Evaluasi
| Model                          | Accuracy | ROC-AUC |
|---------------------------------|----------|---------|
| Decision Tree Classifier        | 62.9%    | 0.630   |
| Random Forest (RandomizedSearchCV) | 74.0%  | 0.802   |

### Kesimpulan
Model **Random Forest Classifier** dengan hyperparameter tuning dipilih sebagai model terbaik untuk prediksi risiko penyakit kardiovaskular karena menunjukkan performa terbaik pada data uji, dengan akurasi sebesar **74.0%** dan ROC-AUC sebesar **0.802**.

Model ini memberikan solusi terhadap **Pernyataan Masalah 1**, yaitu menyediakan metode deteksi dini yang lebih murah dan terjangkau, karena hanya menggunakan data klinis sederhana yang relatif mudah dikumpulkan tanpa pemeriksaan mahal seperti imaging.

Selain itu, model ini juga membantu **identifikasi faktor risiko** penyakit kardiovaskular (**Pernyataan Masalah 2**) dengan menunjukkan fitur-fitur penting seperti usia, kolesterol, dan glukosa dalam meningkatkan risiko.

Dengan demikian, model ini dapat berkontribusi dalam upaya deteksi dini dan pencegahan penyakit kardiovaskular secara lebih efisien dan berbasis data.

### Referensi:
[1] World Health Organization. (2021). Cardiovascular diseases (CVDs). Available: https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)

[2] Mendis, S., Puska, P., & Norrving, B. (2011). Global Atlas on Cardiovascular Disease Prevention and Control. Geneva: World Health Organization.

[3] Dilsizian, S. E., & Siegel, E. L. (2014). Artificial intelligence in medicine and cardiac imaging: Harnessing big data and advanced computing to provide personalized medical diagnosis and treatment. Current Cardiology Reports, 16(1), 441.
