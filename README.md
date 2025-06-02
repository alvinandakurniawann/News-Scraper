# Fake News Detector

<div align="center">
  <a href="#english"><img src="https://img.shields.io/badge/English-4285F4?style=for-the-badge&logo=google-translate&logoColor=white" alt="English"></a>
  <a href="#indonesian"><img src="https://img.shields.io/badge/Bahasa_Indonesia-FF5722?style=for-the-badge&logo=google-translate&logoColor=white" alt="Bahasa Indonesia"></a>
</div>

---

<div id="english">

## 🌟 About

A web application for detecting fake news using TF-IDF + Logistic Regression model. Supports text in both English and Indonesian.

## 🚀 Features

- **Multi-language Support**: Detect news in Indonesian and English
- **Single Optimized Model**: Uses optimized TF-IDF + Logistic Regression
- **Keyword Highlighting**: Shows influential keywords in classification
- **Interactive Visualization**: Displays confidence scores and prediction probabilities
- **Check History**: Saves news checking history

## 📦 Installation

### Prerequisites
- Python 3.8+
- pip (package manager)
- [Supabase](https://supabase.com) account (optional, for history storage)

### Installation Steps

1. Clone the repository:
   ```bash
   git clone [repo-url]
   cd [repo-name]
   ```

2. Create and activate a virtual environment:
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

4. Download NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. Run the application:
   ```bash
   streamlit run main.py
   ```

## ⚙️ Configuration

### Supabase (Optional)
1. Create a `.streamlit/secrets.toml` file
2. Add your Supabase configuration:
   ```toml
   [supabase]
   url = "your-supabase-url"
   key = "your-supabase-key"
   ```

## 🛠️ Technologies

- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn, NLTK, Sastrawi
- **Web Framework**: Streamlit
- **Database**: Supabase (optional)
- **Deployment**: Streamlit Cloud, Heroku, etc.

## 🔧 Preprocessing

- **Clean**: Remove unnecessary characters
- **Case Folding**: Convert to lowercase
- **Tokenization**: Split text into words
- **Stopword Removal**: Remove common words
- **Stemming**: Convert words to their base form

## 📊 Model

- **TF-IDF + Logistic Regression**
  - Accuracy: > 90%
  - Supports English and Indonesian
  - Fast and lightweight

## 🌐 Deployment

Application can be deployed on:
- Streamlit Cloud
- Heroku
- Railway
- Other cloud platforms

## 🤝 Contributing

Contributions are welcome for:
- Code improvements
- New features
- Model accuracy improvements
- Language translations

## 📄 License

MIT License

---

Developed for Final Project - TelkomUniversity © 2025

</div>

<div id="indonesian" style="display: none;">

## 🌟 Tentang

Aplikasi web untuk deteksi berita palsu (fake news) menggunakan model TF-IDF + Logistic Regression. Mendukung teks dalam Bahasa Indonesia dan Inggris.

## 🚀 Fitur Utama

- **Dukungan Multi-bahasa**: Deteksi berita dalam Bahasa Indonesia dan Inggris
- **Model Tunggal**: Menggunakan TF-IDF + Logistic Regression yang telah dioptimalkan
- **Highlight Kata Kunci**: Menampilkan kata-kata kunci yang berpengaruh dalam keputusan klasifikasi
- **Visualisasi Interaktif**: Menampilkan confidence score dan probabilitas prediksi
- **Riwayat Pengecekan**: Menyimpan riwayat pengecekan berita

## 📦 Instalasi

### Persyaratan
- Python 3.8+
- pip (package manager)
- Akun [Supabase](https://supabase.com) (opsional, untuk penyimpanan riwayat)

### Langkah-langkah Instalasi

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

3. Install dependensi:
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

## ⚙️ Konfigurasi

### Supabase (Opsional)
1. Buat file `.streamlit/secrets.toml`
2. Tambahkan konfigurasi Supabase:
   ```toml
   [supabase]
   url = "your-supabase-url"
   key = "your-supabase-key"
   ```

## 🛠️ Teknologi

- **Bahasa Pemrograman**: Python 3.8+
- **Machine Learning**: Scikit-learn, NLTK, Sastrawi
- **Web Framework**: Streamlit
- **Database**: Supabase (opsional)
- **Deployment**: Streamlit Cloud, Heroku, dll.

## 🔧 Preprocessing

- **Bersihkan**: Menghapus karakter tidak perlu
- **Case Folding**: Mengubah ke huruf kecil
- **Tokenisasi**: Memecah teks menjadi kata-kata
- **Stopword Removal**: Menghapus kata umum
- **Stemming**: Mengubah kata ke bentuk dasarnya

## 📊 Model

- **TF-IDF + Logistic Regression**
  - Akurasi: > 90%
  - Mendukung Bahasa Indonesia dan Inggris
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

An application for detecting fake news using machine learning models with a Streamlit interface.

## 🚀 Features

- Automatic news extraction from various news websites
- Fake news detection using machine learning models
- Multi-language support (English and Indonesian)
- Interactive user interface
- Prediction result visualization
- Checking history

## 🛠️ Technologies

- **Programming Language**: Python 3.8+
- **Machine Learning**: Scikit-learn, NLTK, Sastrawi
- **Web Framework**: Streamlit
- **Database**: Supabase (optional)
- **Deployment**: Streamlit Cloud, Heroku, etc.

## 🔧 Preprocessing

- **Clean**: Remove unnecessary characters
- **Case Folding**: Convert to lowercase
- **Tokenization**: Split text into words
- **Stopword Removal**: Remove common words
- **Stemming**: Convert words to their base form

## 📊 Models

- **TF-IDF + Logistic Regression**
  - Accuracy: > 90%
  - Supports English and Indonesian
  - Fast and lightweight

## 🌐 Deployment

Application can be deployed on:
- Streamlit Cloud
- Heroku
- Railway
- Other cloud platforms

## 🤝 Contributing

Contributions are welcome for:
- Code improvements
- New features
- Model accuracy improvements
- Language translations

## 📄 License

MIT License

---

Dikembangkan untuk Tugas Akhir - TelkomUniversity 2025

<div id="indonesian" style="display: none;">

# News Scraper & Detektor Berita Palsu

Aplikasi untuk mendeteksi berita palsu menggunakan model machine learning dengan antarmuka Streamlit.

## 🚀 Fitur

- Ekstraksi berita otomatis dari berbagai situs berita
- Deteksi berita palsu menggunakan model machine learning
- Dukungan multi-bahasa (Indonesia dan Inggris)
- Antarmuka pengguna yang interaktif
- Visualisasi hasil prediksi
- Riwayat pengecekan

## 🛠️ Teknologi

- **Bahasa Pemrograman**: Python 3.8+
- **Machine Learning**: Scikit-learn, NLTK, Sastrawi
- **Web Framework**: Streamlit
- **Database**: Supabase (opsional)
- **Deployment**: Streamlit Cloud, Heroku, dll.

## 🔧 Preprocessing

- **Bersihkan**: Menghapus karakter tidak perlu
- **Case Folding**: Mengubah ke huruf kecil
- **Tokenisasi**: Memecah teks menjadi kata-kata
- **Stopword Removal**: Menghapus kata umum
- **Stemming**: Mengubah kata ke bentuk dasarnya

## 📊 Model

- **TF-IDF + Logistic Regression**
  - Akurasi: > 90%
  - Mendukung Bahasa Indonesia dan Inggris
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

## 📄 License

MIT License

</div>

<div align="center">
  <a href="#english" class="lang-btn" style="padding: 8px 16px; margin: 0 5px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px;">English</a>
  <a href="#indonesia" class="lang-btn" style="padding: 8px 16px; margin: 0 5px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px;">Bahasa Indonesia</a>
</div>

<script>
  // Show language based on priority: URL hash > saved preference > default to English
  if (window.location.hash && ['english', 'indonesia'].includes(window.location.hash.substring(1))) {
    showLanguage(window.location.hash.substring(1));
  } else {
    showLanguage('english'); // Default to English
  }
  
  // Add click handlers to language buttons
  document.querySelectorAll('.lang-btn').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.preventDefault();
      const lang = this.getAttribute('href').substring(1);
      showLanguage(lang);
    });
  });
  
  function showLanguage(lang) {
    if (lang === 'english') {
      document.getElementById('indonesian').style.display = 'none';
    } else if (savedLang && ['english', 'indonesia'].includes(savedLang)) {
      showLanguage(savedLang);
    } else {
      showLanguage('english'); // Default to English
    }
    
    // Add click handlers to language buttons
    document.querySelectorAll('.lang-btn').forEach(btn => {
      btn.addEventListener('click', function(e) {
        e.preventDefault();
        const lang = this.getAttribute('href').substring(1);
        showLanguage(lang);
      });
    });
  };
</script>

