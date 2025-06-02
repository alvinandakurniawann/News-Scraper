# Fake News Detector (ID/EN)

Aplikasi web untuk deteksi berita palsu (fake news) menggunakan model TF-IDF + Logistic Regression. Mendukung teks dalam Bahasa Indonesia dan Inggris.

## 🌟 Fitur Utama

- **Multi-language Support**: Deteksi berita dalam Bahasa Indonesia dan Inggris
- **Model Tunggal**: Menggunakan TF-IDF + Logistic Regression yang telah dioptimalkan
- **Highlight Kata Kunci**: Menampilkan kata-kata kunci yang berpengaruh dalam keputusan klasifikasi
- **Visualisasi Interaktif**: Menampilkan confidence score dan probabilitas prediksi
- **Riwayat Pengecekan**: Menyimpan riwayat pengecekan berita

## 📁 Struktur Project

```
fake-news-detector/
│
├── main.py                    # Aplikasi utama Streamlit
├── text_preprocessor.py       # Preprocessing teks (ID/EN)
├── tfidf_detector.py         # Model TF-IDF + Logistic Regression
├── visualizations.py          # Visualisasi hasil deteksi
├── database_supabase.py       # Koneksi database Supabase
├── config.py                  # Konfigurasi aplikasi
├── requirements.txt           # Daftar dependensi
├── README.md                 # File ini
├── setup.bat                 # Script setup Windows
└── .streamlit/
    └── config.toml          # Konfigurasi Streamlit
```

## 🚀 Cara Menggunakan

### Persyaratan
- Python 3.8+
- pip (package manager)
- Akun [Supabase](https://supabase.com) (opsional, untuk penyimpanan riwayat)

### Instalasi

1. Clone repository ini:
   ```bash
   git clone [repo-url]
   cd [repo-name]
   ```

2. Buat dan aktifkan environment virtual:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download data NLTK:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. Jalankan aplikasi:
   ```bash
   streamlit run main.py
   ```

### Konfigurasi

1. **Supabase (Opsional)**
   - Buat file `.streamlit/secrets.toml`
   - Tambahkan konfigurasi Supabase:
     ```toml
     [supabase]
     url = "your-supabase-url"
     key = "your-supabase-key"
     ```

## 🔍 Cara Kerja

1. **Input Teks**
   - Masukkan teks berita yang ingin diperiksa
   - Atau tempel URL berita (opsional)

2. **Preprocessing**
   - Pembersihan teks (URL, tanda baca, dll)
   - Tokenisasi dan normalisasi
   - Penghapusan stopwords
   - Stemming (untuk Bahasa Indonesia)

3. **Klasifikasi**
   - Ekstraksi fitur menggunakan TF-IDF
   - Prediksi menggunakan model Logistic Regression
   - Menghitung confidence score

4. **Visualisasi**
   - Menampilkan hasil prediksi (Real/Fake)
   - Confidence score
   - Kata-kata kunci yang berpengaruh

## 🛠️ Teknologi

- **Bahasa Pemrograman**: Python 3.8+
- **Machine Learning**: Scikit-learn, NLTK, Sastrawi
- **Web Framework**: Streamlit
- **Database**: Supabase (opsional)
- **Deployment**: Streamlit Cloud, Heroku, dll.

## 🔧 Preprocessing

- **Clean**: Menghapus karakter tidak perlu
- **Case Folding**: Mengubah ke huruf kecil
- **Tokenization**: Memecah teks menjadi kata-kata
- **Stopword Removal**: Menghapus kata umum
- **Stemming**: Mengubah kata ke bentuk dasarnya

## 📊 Model

- **TF-IDF + Logistic Regression**
  - Akurasi: > 90%
  - Support Bahasa Indonesia dan Inggris
  - Cepat dan ringan

## 🌐 Deployment

Aplikasi dapat di-deploy di:
- Streamlit Cloud
- Heroku
- Railway
- Platform cloud lainnya

## 🤝 Berkontribusi

Kontribusi terbuka untuk:
- Perbaikan kode
- Penambahan fitur
- Peningkatan akurasi model
- Terjemahan bahasa

## 📜 Lisensi

MIT License

---

Dikembangkan untuk Tugas Akhir - TelkomUniversity © 2025

## 📄 License

MIT License
