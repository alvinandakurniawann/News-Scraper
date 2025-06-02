import nltk
import sys

print("Python version:", sys.version)
print("NLTK version:", nltk.__version__)

# Test NLTK downloads
try:
    nltk.download('punkt', quiet=False)
    nltk.download('stopwords', quiet=False)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Error downloading NLTK data: {str(e)}")

# Test NLTK tokenization
try:
    from nltk.tokenize import word_tokenize
    text = "Ini adalah contoh teks untuk di-tokenize."
    tokens = word_tokenize(text)
    print("\nTokenization test:")
    print(f"Original text: {text}")
    print(f"Tokens: {tokens}")
except Exception as e:
    print(f"Error in tokenization: {str(e)}")

# Test Sastrawi
try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    word = "menggunakan"
    stemmed = stemmer.stem(word)
    print("\nSastrawi stemmer test:")
    print(f"Original: {word}")
    print(f"Stemmed: {stemmed}")
except Exception as e:
    print(f"Error in Sastrawi: {str(e)}")
    print("Make sure Sastrawi is installed: pip install Sastrawi")
