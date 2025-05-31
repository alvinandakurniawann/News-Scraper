# News Scraper & Fake News Detector

Aplikasi web untuk scraping berita dari berbagai situs Indonesia dan deteksi fake news menggunakan machine learning.

## 📁 Struktur Project

```
news-scraper-project/
│
├── main.py                    # Main Streamlit application (entry point)
├── news_extractor.py          # Module untuk ekstraksi berita
├── text_preprocessor.py       # Module untuk preprocessing teks
├── fake_news_detector.py      # Module untuk deteksi fake news
├── database_supabase.py       # Module untuk manajemen database Supabase
├── config.py                  # Module untuk konfigurasi aplikasi
├── visualizations.py          # Module untuk visualisasi data
├── utils.py                   # Module untuk utility functions
├── requirements.txt           # Daftar dependencies
├── SUPABASE_SETUP.md         # Panduan setup Supabase
├── README.md                 # File ini
├── setup.py                  # Script setup Python
├── setup.sh                  # Script setup Unix/Linux
├── setup.bat                 # Script setup Windows
├── .gitignore               # File git ignore
└── .streamlit/
    └── secrets.toml.example  # Template untuk kredensial Supabase
```

## 🚀 Cara Menjalankan

### Quick Setup (Recommended)

Gunakan script setup otomatis sesuai sistem operasi:

**Windows:**
```bash
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

**Python (All platforms):**
```bash
python setup.py
```

### Manual Setup

1. **Setup Supabase**

   Ikuti panduan lengkap di [SUPABASE_SETUP.md](SUPABASE_SETUP.md) untuk:
   - Membuat akun dan project Supabase
   - Membuat tabel database
   - Mendapatkan kredensial

2. **Konfigurasi Aplikasi**

   ```bash
   mkdir .streamlit
   cp .streamlit/secrets.toml.example .streamlit/secrets.toml
   ```

   Edit `.streamlit/secrets.toml` dengan kredensial Supabase Anda:
   ```toml
   SUPABASE_URL = "https://your-project.supabase.co"
   SUPABASE_KEY = "your-anon-public-key"
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK Data**

   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

5. **Jalankan Aplikasi**

   ```bash
   streamlit run main.py
   ```

## 📋 Fitur Utama

- **News Extraction**: Ekstraksi otomatis dari Detik, Kompas, Tribun, CNN Indonesia, Liputan6
- **Text Preprocessing**: Cleaning, tokenization, stopword removal, stemming
- **Fake News Detection**: Deteksi fake news dengan confidence score
- **Batch Processing**: Proses multiple URLs sekaligus
- **Cloud Database**: Menggunakan Supabase PostgreSQL
- **History Tracking**: Riwayat checking tersimpan di cloud
- **Analytics Dashboard**: Visualisasi statistik dan trend
- **Search & Filter**: Cari dan filter hasil berdasarkan domain, prediksi, dll

## 🔧 Konfigurasi

### Preprocessing Steps:
- `clean`: Membersihkan teks (lowercase, hapus URL, email, dll)
- `punctuation`: Menghapus tanda baca
- `tokenize`: Memecah teks menjadi kata-kata
- `stopwords`: Menghapus kata-kata umum
- `stem`: Mengubah kata ke bentuk dasar

### Model Types :
- TF-IDF + LSTM
- BERT + LogReg
- RoBERTa + GRU

## 🌐 Keuntungan Menggunakan Supabase

1. **Cloud-Based**: Data tersimpan di cloud, bisa diakses dari mana saja
2. **Scalable**: Otomatis scale sesuai kebutuhan
3. **Realtime**: Support realtime updates (optional)
4. **Secure**: Built-in authentication dan Row Level Security
5. **Free Tier**: Gratis untuk project kecil-menengah
6. **PostgreSQL**: Database powerful dengan fitur lengkap


## 🛡️ Security

- Jangan pernah commit file `.streamlit/secrets.toml`
- Gunakan environment variables untuk production
- Enable Row Level Security di Supabase untuk keamanan ekstra

## 🤝 Kontribusi

Feel free to contribute dengan:
- Menambahkan support untuk situs berita lain
- Implementasi model ML yang real
- Perbaikan UI/UX
- Optimasi performa database
- Bug fixes

## 📄 License

MIT License