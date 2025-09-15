# Hotel Booking Demand / Meachine Learning Cancellation Predictor
Dibuat oleh : Regi Dwi Darmawan
---
# Background
*Blue Window* adalah perusahaan penyedia layanan pemesanan hotel berbasis digital dan fisik asal Portugal yang berdiri untuk menjawab tantangan dalam industri pariwisata modern. Seiring meningkatnya kebutuhan perjalanan wisata maupun bisnis, konsumen menuntut akses yang lebih cepat, mudah, dan transparan terhadap informasi akomodasi. Blue Window hadir dengan menawarkan solusi melalui platform digital maupun yang memungkinkan pelanggan untuk membandingkan harga, meninjau ulasan, serta melakukan pemesanan hotel secara real-time.
---
# Problem Statement
1. Apakah pembatalan bookingan yang dilakukan oleh customer dapat di prediksi?
2. Bagaimana cara perusahaan memitigasi resiko yang timbul akibat pembatalan bookingan?
---
# Goals
Tujuan utamanya adalah membuat perhitungan yang akurat mengenai prediksi cancel bookingan agar dapat membuat langkah yang tepat untuk mitigasi resiko yang ada.
---
# Analytic Approach
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
Bahasa Pemrograman: Python
Perpustakaan Utama:
  - Analisis Data: Pandas, NumPy
  - Visualisasi: Matplotlib, Seaborn, ydata-profiling, shap
  - Machine Learning: Scikit-learn, Imbalanced-learn, Category_encoders, XGBoost, Hyperopt
  - Metode penyimpanan model: Joblib
Platform: Visual Studio Code, Jupyter Notebook, streamlit
---
# How To Run
Step:
1. Download semua file yang tersedia
2. Pastikan Python ada berada dalam versi 3.12
3. Pastikan semua versi dalam [requirements.txt]([url](https://github.com/regidwid/ML-Cancellation-Prediction/blob/main/requirements.txt)) sudah terinstall sesuai versi
4. Pastikan sudah menginstall VSCode terlebih dahulu lalu buka file [jupyter](https://github.com/regidwid/ML-Cancellation-Prediction/blob/main/JCDS-0808-002-RegiDwiDarmawan.ipynb) yang tersedia.
5. Sesuaikan path file [data booking hotel](https://github.com/regidwid/ML-Cancellation-Prediction/blob/main/data_hotel_booking_demand.csv) yang ada dalam file jupyter.
6. Run all (jika tidak memungkinkan melakukan tuning load [model](https://github.com/regidwid/ML-Cancellation-Prediction/blob/main/hotel_booking_prediction_model.sav) pada bagian tuning)
7. Anda bisa mengotak atik bagian model, tuning, dll seesuka hati untuk mendapatkan hasil yang lebih optimal.
---
# Results
Model prediksi yang dikembangkan mampu:
- Dari semua pembatalan yang terjadi, model ini dapat mengidentifikasi dengan benar 81% di antaranya.
- Dari semua pemesanan yang tidak dibatalkan, model ini dapat mengidentifikasi dengan benar 78% di antaranya.
Misprediksi ini juga dapat menjadi evaluasi untuk pengembangan lebih lanjut.
---
# Streamlit
Streamlit dari model ini dapat dicoba dengan klik [disini.](https://ml-cancellation-prediction-5fsfzukaba5xl9s9zaupgn.streamlit.app/)
---
# Note
Hasil model cukup besar bisa menjadi opsi untuk tidak mendownload dan load sendiri saat menjalankan file [jupyter](https://github.com/regidwid/ML-Cancellation-Prediction/blob/main/JCDS-0808-002-RegiDwiDarmawan.ipynb) yang sudah didownload sebelumnya.
---
# Author

Regi Dwi Darmawan
kritik dan saran dapat sangat membantu
kirimkan melalui regi.dwi21.rd@gmail.com
