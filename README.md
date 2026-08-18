---
title: Heart Failure Survival Prediction - TabPFN V2
e'mojis': false
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: 1.40.0
app_file: app.py
pinned: false
---

# Prediksi Kelangsungan Hidup Pasien Gagal Jantung

Aplikasi Streamlit untuk penelitian **Heart Failure Survival Prediction using SHAP + TabPFN V2**.

Aplikasi menerima 11 fitur pasien, menjalankan model TabPFN V2, menampilkan probabilitas hidup/meninggal, kemudian menghitung **local SHAP values** untuk menjelaskan kontribusi setiap fitur terhadap prediksi pasien tersebut.

## Fitur input

- age
- anaemia
- creatinine_phosphokinase
- diabetes
- ejection_fraction
- high_blood_pressure
- platelets
- serum_creatinine
- serum_sodium
- sex
- smoking

Fitur `time` tidak digunakan dalam deployment.

## SHAP pada aplikasi

Grafik SHAP dihasilkan **setiap kali pengguna melakukan prediksi**. Jadi hasil penjelasan berubah ketika nilai input pasien berubah.

SHAP yang digunakan adalah model-agnostic `PermutationExplainer`, karena aplikasi menggunakan TabPFN client sebagai predictor black-box. Penjelasan difokuskan pada probabilitas kelas `DEATH_EVENT = 1` (meninggal).

Interpretasi:

- SHAP positif: fitur mendorong probabilitas meninggal menjadi lebih tinggi.
- SHAP negatif: fitur mendorong probabilitas meninggal menjadi lebih rendah / menuju hidup.
- Semakin besar nilai absolut SHAP, semakin besar kontribusi fitur pada prediksi pasien tersebut.

Global SHAP image tidak digunakan sebagai hasil prediksi pasien dan tidak diperlukan untuk menjalankan aplikasi.

## Hugging Face Spaces

1. Buat Space baru dengan SDK **Streamlit**.
2. Upload file pada repository ini.
3. Buka **Settings → Secrets**.
4. Tambahkan secret:

`TABPFN_TOKEN = <token TabPFN>`

Jangan hard-code token di `app.py`.

## Catatan deployment

Model difit pada seluruh deployment training dataset ketika resource model pertama kali dibuat. Model kemudian digunakan kembali melalui cache Streamlit.

Perhitungan local SHAP membutuhkan pemanggilan prediktor beberapa kali. Karena TabPFN client merupakan layanan berbasis API, waktu perhitungan SHAP dapat lebih lama daripada prediksi biasa.

Aplikasi ini merupakan prototipe penelitian dan bukan pengganti diagnosis, keputusan klinis, atau konsultasi tenaga kesehatan.
