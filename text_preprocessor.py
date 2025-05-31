# text_preprocessor.py
"""
Module untuk preprocessing teks bahasa Indonesia
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

# Initialize Indonesian stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()


class TextPreprocessor:
    """Kelas untuk preprocessing teks bahasa Indonesia"""
    
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