# 📦 Customer Churn Predictor App (Streamlit)

Proyek ini adalah aplikasi web sederhana berbasis **Streamlit** untuk memprediksi kemungkinan customer melakukan **churn** (berhenti dari layanan bank) berdasarkan input karakteristik mereka.

---

## 🚀 Cara Menjalankan

### 1. Clone/Copy Project Directory
Pastikan struktur folder seperti berikut:

```
churn_streamlit/
├── streamlit_app.py
├── preprocess_input.py
├── generate_features.py
├── model_lgbm_optuna.pkl
├── best_threshold.pkl
├── rfe_selector.pkl
├── ordinal_encoder.pkl
├── final_columns.pkl
├── requirements.txt
└── README.md
```

### 2. Buat Virtual Environment (Opsional Tapi Direkomendasikan)
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate    # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi
```bash
cd churn_streamlit
streamlit run streamlit_app.py
```

---

## 🧠 Teknologi di Balik Aplikasi
- **Model:** LightGBM (hasil tuning Optuna)
- **Feature Engineering:** Modular via `generate_features.py`
- **Preprocessing:** Encoding, One-Hot, RFE via `preprocess_input.py`
- **Deployment:** Streamlit

---

## 📝 Catatan Tambahan
- Semua preprocessing mengikuti pipeline saat training.
- `preprocess_input.py` akan memanggil `generate_features.py` secara otomatis.
- Tidak perlu mendefinisikan ulang fungsi preprocessing di `streamlit_app.py` — cukup `import`.
- Hasil prediksi ditampilkan sebagai **probabilitas churn** dan status akhir (Churn / Not Churn).

---

## 📧 Kontak
Dibuat oleh: **ariel ariel**  
Jika ada pertanyaan: [arielmhaa@gmail.com](mailto:arielmhaa@gmail.com)
