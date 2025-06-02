import nltk
from text_preprocessor import TextPreprocessor

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt...")
    nltk.download('punkt')

def test_preprocessing():
    # Inisialisasi preprocessor
    preprocessor = TextPreprocessor(language='indonesian')
    
    # Teks contoh
    text = """
    KOMPAS.com - Presiden Joko Widodo (Jokowi) mengatakan bahwa bangsa Indonesia 
    memiliki keterkaitan dengan China secara biologis. Ia mengeklaim banyak DNA 
    warga Indonesia berasal dari etnis Tionghoa atau Tiongkok.
    
    "Kita ini sebenarnya banyak yang berasal dari Tiongkok. DNA-nya banyak yang 
    berasal dari Tionghoa," kata Jokowi dalam acara pembukaan pameran budaya 
    Tionghoa di Jakarta, Senin (1/1/2023).
    
    Kunjungi juga: https://example.com/berita-lainnya
    Email: kontak@example.com
    #Jokowi #China #DNA
    """
    
    # Langkah-langkah preprocessing
    steps = ['clean', 'normalize', 'punctuation', 'tokenize', 'stopwords', 'stem']
    
    print("\n" + "#" * 80)
    print("TESTING PREPROCESSING STEP BY STEP")
    print("#" * 80)
    
    # Test clean_text
    print("\n1. Testing clean_text:")
    cleaned = preprocessor.clean_text(text)
    
    # Test normalize_text
    print("\n2. Testing normalize_text:")
    normalized = preprocessor.normalize_text(cleaned)
    
    # Test remove_punctuation
    print("\n3. Testing remove_punctuation:")
    no_punct = preprocessor.remove_punctuation(normalized)
    
    # Test tokenize
    print("\n4. Testing tokenize:")
    tokens = preprocessor.tokenize(no_punct)
    
    # Test remove_stopwords
    print("\n5. Testing remove_stopwords:")
    no_stopwords = preprocessor.remove_stopwords(tokens)
    
    # Test stem_text
    print("\n6. Testing stem_text:")
    stemmed = preprocessor.stem_text(no_stopwords)
    
    # Test full pipeline
    print("\n" + "#" * 80)
    print("TESTING FULL PREPROCESSING PIPELINE")
    print("#" * 80)
    result = preprocessor.preprocess_pipeline(text, steps=steps)
    
    print("\n" + "#" * 80)
    print("FINAL RESULT:")
    print("#" * 80)
    print(result)

if __name__ == "__main__":
    test_preprocessing()
