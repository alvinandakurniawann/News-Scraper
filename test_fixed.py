from text_preprocessor import TextPreprocessor

def test_preprocessing():
    # Inisialisasi preprocessor
    print("\n" + "#" * 80)
    print("INITIALIZING TEXT PREPROCESSOR")
    print("#" * 80)
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
    
    # Test clean_text
    print("\n" + "#" * 60)
    print("TESTING clean_text")
    print("#" * 60)
    cleaned = preprocessor.clean_text(text)
    
    # Test normalize_text
    print("\n" + "#" * 60)
    print("TESTING normalize_text")
    print("#" * 60)
    normalized = preprocessor.normalize_text(cleaned)
    
    # Test remove_punctuation
    print("\n" + "#" * 60)
    print("TESTING remove_punctuation")
    print("#" * 60)
    no_punct = preprocessor.remove_punctuation(normalized)
    
    # Test tokenize
    print("\n" + "#" * 60)
    print("TESTING tokenize")
    print("#" * 60)
    tokens = preprocessor.tokenize(no_punct)
    
    # Test remove_stopwords
    print("\n" + "#" * 60)
    print("TESTING remove_stopwords")
    print("#" * 60)
    no_stopwords = preprocessor.remove_stopwords(tokens)
    
    # Test stem_text
    print("\n" + "#" * 60)
    print("TESTING stem_text")
    print("#" * 60)
    stemmed = preprocessor.stem_text(no_stopwords)
    
    # Test full pipeline
    print("\n" + "#" * 80)
    print("TESTING FULL PREPROCESSING PIPELINE")
    print("#" * 80)
    result = preprocessor.preprocess_pipeline(
        text,
        steps=['clean', 'normalize', 'punctuation', 'tokenize', 'stopwords', 'stem']
    )
    
    print("\n" + "#" * 80)
    print("FINAL RESULT:")
    print("#" * 80)
    print(result)

if __name__ == "__main__":
    test_preprocessing()
