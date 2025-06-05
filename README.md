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
