# Hotel Booking Demand / Meachine Learning Cancellation Predictor
---
Dibuat oleh : Regi Dwi Darmawan
---
# Background
---
**Blue Window** adalah perusahaan penyedia layanan pemesanan hotel berbasis digital dan fisik asal Portugal yang berdiri untuk menjawab tantangan dalam industri pariwisata modern. Seiring meningkatnya kebutuhan perjalanan wisata maupun bisnis, konsumen menuntut akses yang lebih cepat, mudah, dan transparan terhadap informasi akomodasi. Blue Window hadir dengan menawarkan solusi melalui platform digital yang memungkinkan pelanggan untuk membandingkan harga, meninjau ulasan, serta melakukan pemesanan hotel secara real-time.
---
# Problem Statement
---
1. Apakah pembatalan bookingan yang dilakukan oleh customer dapat di prediksi?
2. Bagaimana cara perusahaan memitigasi resiko yang timbul akibat pembatalan bookingan?
---
# Goals
---
Tujuan utamanya adalah membuat perhitungan yang akurat mengenai prediksi cancel bookingan agar dapat membuat langkah yang tepat untuk mitigasi resiko yang ada.

# Analytic Approach
---
Hal pertama yang akan  dilakukan adalah menganalisis data untuk menemukan pola customer yang melakukan cancel bookingan.
Kemudian kita akan membangun model klasifikasi yang akan membantu perusahaan untuk dapat memprediksi probabilitas apakah customer ini akan membatalkan bookingan atau tidak, dengan pendekatan:
1. Pemahaman Bisnis:
Menjelaskan permasalahan dan tujuan utama.
2. Eksplorasi Data:
Menganalisis dataset untuk memahami pola dan anomali.
3. Pra-pemrosesan Data:
Menangani data yang hilang.
Melakukan rekayasa fitur.
Encoding fitur kategorikal.
Normalisasi fitur numerik.
4. Pengembangan Model:
Membandingkan beberapa algoritma machine learning.
Memilih model terbaik berdasarkan metrik evaluasi.
5. Evaluasi:
Menilai performa model menggunakan metrik recall.
---
# Tools
---
Bahasa Pemrograman: Python
Perpustakaan Utama:
  - Analisis Data: Pandas, NumPy
  - Visualisasi: Matplotlib, Seaborn, ydata-profiling, shap
  - Machine Learning: Scikit-learn, Imbalanced-learn, Category_encoders, XGBoost, Hyperopt
  - Metode penyimpanan model: Joblib
Platform: Visual Studio Code, Jupyter Notebook, streamlit
---
# How To Run
---
