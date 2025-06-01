# text_preprocessor.py
"""
Module untuk preprocessing teks bahasa Indonesia
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

# Initialize stemmers
# Indonesian stemmer
factory = StemmerFactory()
id_stemmer = factory.create_stemmer()

# English stemmer
en_stemmer = PorterStemmer()


class TextPreprocessor:
    """Kelas untuk preprocessing teks bahasa Indonesia"""
    
    def __init__(self, language='indonesian'):
        """
        Inisialisasi TextPreprocessor
        
        Args:
            language (str, optional): Bahasa yang digunakan ('indonesian' atau 'english').
                                   Default: 'indonesian'
        """
        self.language = language.lower()
        self.stop_words = set()
        self._initialize_stopwords()
        
    def __str__(self):
        """
        Return string representation of the preprocessor
        
        Returns:
            str: String representation
        """
        return f"TextPreprocessor(language='{self.language}', stopwords_count={len(self.stop_words)})"
    
    def _initialize_stopwords(self):
        """Inisialisasi stopwords berdasarkan bahasa"""
        if self.language == 'english':
            self.stop_words = set(stopwords.words('english'))
        else:  # Default to Indonesian
            self.stop_words = set(stopwords.words('indonesian'))
            # Tambahkan stopwords kustom
            self.stop_words.update([
                'yang', 'untuk', 'pada', 'ke', 'di', 'dari', 'ini', 'itu', 
                'dengan', 'adalah', 'tersebut', 'dan', 'atau', 'tapi', 'juga',
                'saya', 'kamu', 'kita', 'mereka', 'kalian', 'kami', 'ini', 'itu',
                'sini', 'situ', 'sana', 'mana', 'siap', 'apa', 'mengapa', 'kenapa',
                'bagaimana', 'kapan', 'dimana', 'kemana', 'dari', 'ke', 'dari', 'pada',
                'jika', 'kalau', 'agar', 'supaya', 'sehingga', 'karena', 'sebab',
                'tetapi', 'namun', 'akan', 'telah', 'sudah', 'belum', 'tidak', 'bukan',
                'jangan', 'saja', 'sih', 'pun', 'lah', 'kah', 'tah', 'dll', 'dsb', 'dst',
                'tsb', 'dalam', 'atas', 'bawah', 'depan', 'belakang', 'samping', 'dll'
            ])
        
    def clean_text(self, text):
        """Basic text cleaning"""
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove special characters and numbers (keep only letters and basic punctuation)
        text = re.sub(r'[^\w\s.,!?]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
        
    def normalize_text(self, text):
        """
        Normalize text (slang, numbers, etc.)
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Common Indonesian normalizations
        normalizations = {
            # Common abbreviations
            'yg': 'yang',
            'dg': 'dengan',
            'dgn': 'dengan',
            'tdk': 'tidak',
            'gak': 'tidak',
            'ga': 'tidak',
            'jg': 'juga',
            'udh': 'sudah',
            'udah': 'sudah',
            'sdh': 'sudah',
            'sya': 'saya',
            'sy': 'saya',
            'aku': 'saya',
            'gw': 'saya',
            'gua': 'saya',
            'lu': 'kamu',
            'loe': 'kamu',
            'lo': 'kamu',
            'elu': 'kamu',
            'ente': 'kamu',
            'nyokap': 'ibu',
            'bokap': 'ayah',
            'bapak': 'ayah',
            'emak': 'ibu',
            'mama': 'ibu',
            'papa': 'ayah',
            'om': 'paman',
            'tante': 'bibi',
            'kalo': 'kalau',
            'klo': 'kalau',
            'kl': 'kalau',
            'jgn': 'jangan',
            'jangan2': 'jangan-jangan',
            'aja': 'saja',
            'ajah': 'saja',
            'sih': '',
            'deh': '',
            'dong': '',
            'kok': '',
            'ya': '',
            'yah': '',
            'nih': 'ini',
            'tuh': 'itu',
            'gitu': 'begitu',
            'gini': 'begini',
            'banget': 'sangat',
            'bgt': 'sangat',
            'bngt': 'sangat',
            'sgt': 'sangat',
            'amat': 'sangat',
            'sekali': 'sangat',
            'bener': 'benar',
            'bner': 'benar',
            'bkn': 'bukan',
            'enggak': 'tidak',
            'nggak': 'tidak',
            'gpp': 'tidak apa-apa',
            'gapapa': 'tidak apa-apa',
            'gimana': 'bagaimana',
            'gmn': 'bagaimana',
            'gmana': 'bagaimana',
            'gakbisa': 'tidak bisa',
            'gabisa': 'tidak bisa',
            'gbs': 'tidak bisa',
            'gakboleh': 'tidak boleh',
            'gaboleh': 'tidak boleh',
            'gblh': 'tidak boleh',
            'gakmau': 'tidak mau',
            'gamau': 'tidak mau'
        }
        
        # Apply normalizations
        words = text.split()
        normalized_words = [normalizations.get(word, word) for word in words]
        
        # Join back to string and clean up
        text = ' '.join(normalized_words)
        text = ' '.join(text.split())  # Remove extra spaces
        
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
    
    def stem_text(self, tokens, language='indonesian'):
        """
        Stem tokens based on specified language
        
        Args:
            tokens (list): List of tokens to be stemmed
            language (str): Language for stemming ('indonesian' or 'english')
            
        Returns:
            list: List of stemmed tokens
        """
        if language.lower() == 'english':
            return [en_stemmer.stem(token) for token in tokens]
        else:  # Default to Indonesian
            return [id_stemmer.stem(token) for token in tokens]
    
    def preprocess_pipeline(self, text, steps=None, language='indonesian'):
        """
        Run preprocessing pipeline based on selected steps
        
        Args:
            text (str): Input text to preprocess
            steps (list, optional): List of preprocessing steps to apply. 
                                 Default is ['clean', 'normalize', 'tokenize', 'stopwords', 'stem']
            language (str): Language for language-specific processing ('indonesian' or 'english')
            
        Returns:
            str: Preprocessed text
        """
        if steps is None:
            steps = ['clean', 'normalize', 'tokenize', 'stopwords', 'stem']
            
        if not text or not isinstance(text, str):
            return ""
            
        # Update language if different
        if language.lower() != self.language:
            self.language = language.lower()
            self._initialize_stopwords()
        
        result = text
        tokens = None
        
        try:
            # Clean text (always first step if included)
            if 'clean' in steps:
                result = self.clean_text(result)
                if not result:
                    return ""
            
            # Normalize text (after cleaning, before tokenization)
            if 'normalize' in steps:
                result = self.normalize_text(result)
                if not result:
                    return ""
            
            # Remove punctuation (if needed before tokenization)
            if 'punctuation' in steps:
                result = self.remove_punctuation(result)
            
            # Tokenize
            if 'tokenize' in steps:
                tokens = self.tokenize(result)
                result = tokens
            
            # Remove stopwords (requires tokenization)
            if 'stopwords' in steps and tokens is not None:
                tokens = self.remove_stopwords(tokens)
                result = tokens
            
            # Stemming (requires tokenization)
            if 'stem' in steps and tokens is not None:
                tokens = self.stem_text(tokens, language=language)
                result = tokens
            
            # Convert back to string if tokens
            if isinstance(result, list):
                result = ' '.join(result)
                
        except Exception as e:
            print(f"Error in preprocessing pipeline: {str(e)}")
            return ""
        
        return result


if __name__ == "__main__":
    # Contoh penggunaan
    sample_text = """
    Halo! Saya pengen nanya nih, gimana caranya biar bisa jago programming? 
    Soalnya saya udh coba belajar sendiri tp masih aja gak ngerti-ngerti. 
    Ada yg bisa bantu? Makasih sebelumnya! 😊
    """
    
    print("=== Contoh Penggunaan TextPreprocessor ===")
    
    # Inisialisasi preprocessor
    preprocessor = TextPreprocessor(language='indonesian')
    print(f"Preprocessor: {preprocessor}")
    
    # Preprocess teks
    processed_text = preprocessor.preprocess_pipeline(
        sample_text,
        steps=['clean', 'normalize', 'tokenize', 'stopwords', 'stem']
    )
    
    print("\n=== Hasil Preprocessing ===")
    print("Teks asli:")
    print(sample_text)
    print("\nTeks setelah preprocessing:")
    print(processed_text)
    
    # Contoh penggunaan langkah per langkah
    print("\n=== Langkah per Langkah ===")
    cleaned = preprocessor.clean_text(sample_text)
    print(f"1. Cleaned: {cleaned[:100]}...")
    
    normalized = preprocessor.normalize_text(cleaned)
    print(f"2. Normalized: {normalized[:100]}...")
    
    tokens = preprocessor.tokenize(normalized)
    print(f"3. Tokens: {tokens[:5]}... (total: {len(tokens)} tokens)")
    
    no_stopwords = preprocessor.remove_stopwords(tokens)
    print(f"4. After stopwords removal: {no_stopwords[:5]}... (total: {len(no_stopwords)} tokens)")
    
    stemmed = preprocessor.stem_text(no_stopwords)
    print(f"5. After stemming: {stemmed[:5]}... (total: {len(stemmed)} tokens)")
    
    final_text = ' '.join(stemmed)
    print(f"6. Final text: {final_text[:100]}...")