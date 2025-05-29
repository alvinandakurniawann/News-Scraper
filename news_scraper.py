import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from urllib.parse import urlparse

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK data: {str(e)}")

# Initialize Indonesian stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

class NewsExtractor:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def extract_from_url(self, url):
        """Extract title and content from news URL"""
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse domain to apply specific extraction rules
            domain = urlparse(url).netloc
            
            # Initialize results
            title = ""
            content = ""
            
            # Domain-specific extraction rules
            if 'detik.com' in domain:
                title, content = self._extract_detik(soup)
            elif 'kompas.com' in domain:
                title, content = self._extract_kompas(soup)
            elif 'tribunnews.com' in domain:
                title, content = self._extract_tribun(soup)
            elif 'cnnindonesia.com' in domain:
                title, content = self._extract_cnn(soup)
            elif 'liputan6.com' in domain:
                title, content = self._extract_liputan6(soup)
            else:
                # Generic extraction
                title, content = self._extract_generic(soup)
            
            return {
                'success': True,
                'title': title,
                'content': content,
                'url': url,
                'domain': domain,
                'extracted_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'url': url
            }
    
    def _extract_detik(self, soup):
        """Extract from detik.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', class_='detail__title') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', class_='detail__body-text') or soup.find('div', class_='itp_bodycontent')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_kompas(self, soup):
        """Extract from kompas.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', class_='read__title') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', class_='read__content') or soup.find('div', class_='content')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_tribun(self, soup):
        """Extract from tribunnews.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', id='arttitle') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', class_='content') or soup.find('div', class_='txt-article')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_cnn(self, soup):
        """Extract from cnnindonesia.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', class_='title') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', id='detikdetailtext') or soup.find('div', class_='detail-text')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_liputan6(self, soup):
        """Extract from liputan6.com"""
        title = ""
        content = ""
        
        # Title extraction
        title_elem = soup.find('h1', class_='read-page--header--title') or soup.find('h1')
        if title_elem:
            title = title_elem.get_text().strip()
        
        # Content extraction
        content_elem = soup.find('div', class_='article-content-body__item-content') or soup.find('div', class_='article-content')
        if content_elem:
            paragraphs = content_elem.find_all('p')
            content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        return title, content
    
    def _extract_generic(self, soup):
        """Generic extraction for unknown domains"""
        title = ""
        content = ""
        
        # Title extraction - try multiple selectors
        title_selectors = [
            'h1',
            'title',
            'meta[property="og:title"]',
            'meta[name="twitter:title"]'
        ]
        
        for selector in title_selectors:
            elem = soup.find(selector)
            if elem:
                if selector.startswith('meta'):
                    title = elem.get('content', ' ')
                else:
                    title = elem.get_text()
                title = title.strip()
                if title:
                    break
        
        # Content extraction - try to find article body
        content_selectors = [
            'article',
            'div[class*="content"]',
            'div[class*="article"]',
            'div[class*="post"]',
            'main'
        ]
        
        for selector in content_selectors:
            if '[' in selector:
                # Handle attribute selectors
                tag, attr = selector.split('[')
                attr = attr.rstrip(']')
                key, value = attr.split('*=') 
                elem = soup.find(tag, class_=lambda x: x and value.strip('"') in x)
            else:
                elem = soup.find(selector)
            
            if elem:
                paragraphs = elem.find_all('p')
                if paragraphs:
                    content = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
                    if content:
                        break
        
        return title, content

class TextPreprocessor:
    def __init__(self):
        # Indonesian stopwords
        self.stop_words = set(stopwords.words('indonesian'))
        # Add custom stopwords
        self.stop_words.update(['yang', 'untuk', 'pada', 'ke', 'di', 'dari', 'ini', 'itu', 'dengan', 'adalah', 'tersebut'])
        
    def clean_text(self, text):
        """Basic text cleaning"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def remove_punctuation(self, text):
        """Remove punctuation"""
        return text.translate(str.maketrans('', '', string.punctuation))
    
    def tokenize(self, text):
        """Tokenize text using split() as a fallback"""
        try:
            return word_tokenize(text)
        except LookupError:
            # Fallback to simple whitespace tokenizer if punkt is not available
            return text.split()
    
    def remove_stopwords(self, tokens):
        """Remove stopwords"""
        return [token for token in tokens if token not in self.stop_words]
    
    def stem_text(self, tokens):
        """Stem tokens"""
        return [stemmer.stem(token) for token in tokens]
    
    def preprocess_pipeline(self, text, steps):
        """Run preprocessing pipeline based on selected steps"""
        result = text
        tokens = None
        
        if 'clean' in steps:
            result = self.clean_text(result)
        
        if 'punctuation' in steps:
            result = self.remove_punctuation(result)
        
        if 'tokenize' in steps:
            tokens = self.tokenize(result)
            result = tokens
        
        if 'stopwords' in steps and tokens is not None:
            tokens = self.remove_stopwords(tokens)
            result = tokens
        
        if 'stem' in steps and tokens is not None:
            tokens = self.stem_text(tokens)
            result = tokens
        
        # Convert back to string if tokens
        if isinstance(result, list):
            result = ' '.join(result)
        
        return result

# Streamlit App
def main():
    st.set_page_config(
        page_title="News Scraper & Preprocessor",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 News Scraper & Preprocessor untuk Fake News Detection")
    st.markdown("---")
    
    # Initialize session state
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'preprocessed_text' not in st.session_state:
        st.session_state.preprocessed_text = None
    
    # Sidebar for preprocessing options
    with st.sidebar:
        st.header("⚙️ Preprocessing Options")
        
        preprocessing_steps = st.multiselect(
            "Pilih langkah preprocessing:",
            ['clean', 'punctuation', 'tokenize', 'stopwords', 'stem'],
            default=['clean', 'punctuation', 'tokenize', 'stopwords']
        )
        
        st.markdown("---")
        st.markdown("### 📝 Keterangan:")
        st.markdown("""        
        - **clean**: Membersihkan teks (lowercase, hapus URL, email, dll)
        - **punctuation**: Menghapus tanda baca
        - **tokenize**: Memecah teks menjadi kata-kata
        - **stopwords**: Menghapus kata-kata umum
        - **stem**: Mengubah kata ke bentuk dasar
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🔗 Input URL Berita")
        
        # URL input
        url_input = st.text_input(
            "Masukkan URL berita:",
            placeholder="https://www.detik.com/..."
        )
        
        # Extract button
        if st.button("🔍 Extract News", type="primary"):
            if url_input:
                with st.spinner("Mengekstrak berita..."):
                    extractor = NewsExtractor()
                    result = extractor.extract_from_url(url_input)
                    
                    if result['success']:
                        st.session_state.extracted_data = result
                        st.success("✅ Berhasil mengekstrak berita!")
                    else:
                        st.error(f"❌ Error: {result['error']}")
            else:
                st.warning("⚠️ Masukkan URL terlebih dahulu!")
    
    with col2:
        st.header("📊 Hasil Ekstraksi")
        
        if st.session_state.extracted_data and st.session_state.extracted_data['success']:
            data = st.session_state.extracted_data
            
            # Display extracted info
            st.subheader("📌 Judul:")
            st.write(data['title'])
            
            st.subheader("🌐 Domain:")
            st.write(data['domain'])
            
            st.subheader("📅 Waktu Ekstraksi:")
            st.write(data['extracted_at'])
            
            # Show content preview
            st.subheader("📄 Preview Konten:")
            with st.expander("Lihat konten lengkap"):
                st.text_area("Konten Berita", data['content'], height=200, disabled=True, label_visibility="collapsed")
    
    # Preprocessing section
    if st.session_state.extracted_data and st.session_state.extracted_data['success']:
        st.markdown("---")
        st.header("🔧 Text Preprocessing")
        
        col3, col4 = st.columns([1, 1])
        
        with col3:
            # Combine title and content for preprocessing
            full_text = f"{st.session_state.extracted_data['title']} {st.session_state.extracted_data['content']}"
            
            if st.button("🚀 Run Preprocessing", type="secondary"):
                with st.spinner("Memproses teks..."):
                    preprocessor = TextPreprocessor()
                    processed_text = preprocessor.preprocess_pipeline(full_text, preprocessing_steps)
                    st.session_state.preprocessed_text = processed_text
                    st.success("✅ Preprocessing selesai!")
        
        with col4:
            if st.session_state.preprocessed_text:
                st.subheader("📝 Hasil Preprocessing:")
                st.text_area("Hasil Preprocessing", st.session_state.preprocessed_text, height=200, disabled=True, label_visibility="collapsed")
                
                # Statistics
                st.subheader("📊 Statistik:")
                col5, col6, col7 = st.columns(3)
                with col5:
                    st.metric("Jumlah Karakter", len(st.session_state.preprocessed_text))
                with col6:
                    st.metric("Jumlah Kata", len(st.session_state.preprocessed_text.split()))
                with col7:
                    st.metric("Rata-rata Panjang Kata", 
                             f"{sum(len(word) for word in st.session_state.preprocessed_text.split()) / len(st.session_state.preprocessed_text.split()):.2f}")
        
        # Export section
        st.markdown("---")
        st.header("💾 Export Data")
        
        col8, col9 = st.columns(2)
        
        with col8:
            if st.button("📥 Download as CSV"):
                # Create dataframe
                df = pd.DataFrame([{
                    'url': st.session_state.extracted_data['url'],
                    'domain': st.session_state.extracted_data['domain'],
                    'title': st.session_state.extracted_data['title'],
                    'content': st.session_state.extracted_data['content'],
                    'preprocessed_text': st.session_state.preprocessed_text if st.session_state.preprocessed_text else '',
                    'extracted_at': st.session_state.extracted_data['extracted_at']
                }])
                
                # Convert to CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"news_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col9:
            if st.button("📥 Download as JSON"):
                import json
                # Create JSON data
                json_data = {
                    'extraction': st.session_state.extracted_data,
                    'preprocessing': {
                        'steps': preprocessing_steps,
                        'result': st.session_state.preprocessed_text if st.session_state.preprocessed_text else None
                    }
                }
                
                st.download_button(
                    label="Download JSON",
                    data=json.dumps(json_data, indent=2, ensure_ascii=False),
                    file_name=f"news_extract_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
