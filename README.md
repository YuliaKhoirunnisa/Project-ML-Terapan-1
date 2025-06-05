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

- **Pernyataan Masalah 2**: Banyak faktor risiko seperti hipertensi, hiperglikemia, hiperkolesterolemia, obesitas, kebiasaan merokok, dan gaya hidup sedentari sulit diidentifikasi secara dini tanpa alat bantu berbasis data.

- **Pernyataan Masalah 3**: Belum tersedia model machine learning yang akurat dan efisien berbasis data klinis sederhana untuk memprediksi risiko penyakit kardiovaskular.
