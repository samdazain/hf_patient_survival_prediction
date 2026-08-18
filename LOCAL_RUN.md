# Menjalankan secara lokal

1. Buat virtual environment.
2. Install dependency:

```bash
pip install -r requirements.txt
```

3. Set token TabPFN pada environment variable `TABPFN_TOKEN`.
4. Jalankan:

```bash
streamlit run app.py
```

Setelah pengguna mengisi 11 fitur dan menekan **Mulai Prediksi**, aplikasi akan menampilkan hasil prediksi dan local SHAP contribution untuk pasien tersebut.
