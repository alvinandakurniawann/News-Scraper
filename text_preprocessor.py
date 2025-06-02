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

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt')
    nltk.download('stopwords')

# Initialize stemmers
try:
    factory = StemmerFactory()
    id_stemmer = factory.create_stemmer()
    en_stemmer = PorterStemmer()
    print("Stemmers initialized successfully")
except Exception as e:
    print(f"Error initializing stemmers: {str(e)}")
    raise


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
        try:
            print(f"Initializing TextPreprocessor with language: {self.language}")
            self._initialize_stopwords()
            print("Stopwords initialized successfully")
        except Exception as e:
            print(f"Error initializing TextPreprocessor: {str(e)}")
            raise
        
    def __str__(self):
        """
        Return string representation of the preprocessor
        
        Returns:
            str: String representation
        """
        return f"TextPreprocessor(language='{self.language}', stopwords_count={len(self.stop_words)})"
    
    def _initialize_stopwords(self):
        """Inisialisasi stopwords berdasarkan bahasa"""
        try:
            # Clear existing stopwords
            self.stop_words = set()
            
            # Load NLTK stopwords if available
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("Downloading NLTK stopwords...")
                nltk.download('stopwords', quiet=True)
            
            if self.language == 'english':
                # English stopwords from NLTK
                self.stop_words.update(stopwords.words('english'))
                print(f"Loaded {len(self.stop_words)} English stopwords from NLTK")
                
            else:  # Default to Indonesian
                # Try to load Indonesian stopwords from NLTK first
                try:
                    indonesian_stopwords = stopwords.words('indonesian')
                    if indonesian_stopwords:
                        self.stop_words.update(indonesian_stopwords)
                        print(f"Loaded {len(self.stop_words)} Indonesian stopwords from NLTK")
                except Exception as e:
                    print(f"Could not load Indonesian stopwords from NLTK: {str(e)}")
                
                # Add our custom Indonesian stopwords
                custom_id_stopwords = [
                    # Basic stopwords
                    'yang', 'untuk', 'pada', 'ke', 'di', 'dari', 'ini', 'itu', 
                    'dengan', 'adalah', 'tersebut', 'dan', 'atau', 'tapi', 'juga',
                    'saya', 'kamu', 'kita', 'mereka', 'kalian', 'kami', 'ini', 'itu',
                    'sini', 'situ', 'sana', 'mana', 'siap', 'apa', 'mengapa', 'kenapa',
                    'bagaimana', 'kapan', 'dimana', 'kemana', 'dari', 'ke', 'pada',
                    'jika', 'kalau', 'agar', 'supaya', 'sehingga', 'karena', 'sebab',
                    'tetapi', 'namun', 'akan', 'telah', 'sudah', 'belum', 'tidak', 'bukan',
                    'jangan', 'saja', 'sih', 'pun', 'lah', 'kah', 'tah', 'dll', 'dsb', 'dst',
                    'tsb', 'dalam', 'atas', 'bawah', 'depan', 'belakang', 'samping', 'dll',
                    
                    # Additional common words
                    'saya', 'aku', 'gue', 'gw', 'saya', 'aku', 'saya', 'aku', 'saya',
                    'kamu', 'anda', 'engkau', 'kau', 'kamu', 'anda', 'engkau', 'kau',
                    'kita', 'kami', 'kita', 'kami', 'mereka', 'dia', 'ia', 'beliau',
                    'ini', 'itu', 'anu', 'tersebut', 'begitu', 'begini', 'situ', 'sini',
                    'sana', 'situ', 'sini', 'sana', 'situ', 'sini', 'sana', 'situ',
                    
                    # Question words
                    'apa', 'siapa', 'mengapa', 'kenapa', 'bagaimana', 'kapan', 
                    'dimana', 'kemana', 'darimana', 'berapa', 'apakah', 'siapakah',
                    'mengapakah', 'bagaimanakah', 'kapankah', 'dimanakah', 'kemanakah',
                    'dari manakah', 'berapakah',
                    
                    # Common verbs
                    'adalah', 'ialah', 'merupakan', 'menjadi', 'ada', 'tidak', 'bukan',
                    'sudah', 'telah', 'belum', 'sedang', 'akan', 'mau', 'ingin', 'bisa',
                    'dapat', 'boleh', 'harus', 'perlu', 'biasa', 'mungkin', 'pasti',
                    'tentu', 'mungkin', 'biasanya', 'sering', 'kadang', 'jarang',
                    'pernah', 'belum pernah', 'tidak pernah', 'selalu', 'seringkali',
                    
                    # Common adjectives
                    'baik', 'buruk', 'bagus', 'jelek', 'besar', 'kecil', 'tinggi', 
                    'rendah', 'panjang', 'pendek', 'lebar', 'sempit', 'tebal', 'tipis',
                    'berat', 'ringan', 'mahal', 'murah', 'cepat', 'lambat', 'lama',
                    'baru', 'lama', 'tua', 'muda', 'banyak', 'sedikit', 'penuh', 'kosong',
                    'keras', 'lembut', 'kasar', 'halus', 'panas', 'dingin', 'hangat',
                    'sejuk', 'basah', 'kering', 'bersih', 'kotor', 'rapi', 'berantakan',
                    'indah', 'cantik', 'jelek', 'buruk', 'baik', 'jahat', 'benar', 'salah',
                    'senang', 'sedih', 'gembira', 'bahagia', 'susah', 'sulit', 'mudah',
                    'sulit', 'mudah', 'susah', 'sulit', 'mudah', 'susah', 'sulit', 'mudah'
                ]
                
                self.stop_words.update(custom_id_stopwords)
                print(f"Added {len(custom_id_stopwords)} custom Indonesian stopwords")
            
            # Add common punctuation and numbers as stopwords
            punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
            numbers = set("0123456789")
            
            self.stop_words.update(punctuation)
            self.stop_words.update(numbers)
            
            print(f"Total stopwords loaded: {len(self.stop_words)}")
            
        except Exception as e:
            print(f"Error initializing stopwords: {str(e)}")
            # Fallback to a minimal set of stopwords if all else fails
            self.stop_words = {
                'yang', 'untuk', 'pada', 'ke', 'di', 'dari', 'ini', 'itu', 'dan', 'dengan',
                'adalah', 'tersebut', 'atau', 'tapi', 'juga', 'saya', 'kamu', 'kita',
                'mereka', 'kalian', 'kami', 'sini', 'situ', 'sana', 'tidak', 'bukan',
                'jangan', 'sudah', 'belum', 'akan', 'telah', 'nanti', 'lagi', 'saja', 'sih',
                'pun', 'lah', 'kah', 'tah', 'dll', 'dsb', 'dst', 'tsb', 'dalam', 'atas',
                'bawah', 'depan', 'belakang', 'samping', 'dll', 'saya', 'aku', 'kamu',
                'anda', 'dia', 'mereka', 'kita', 'kami', 'kalian', 'ini', 'itu', 'anu',
                'tersebut', 'begitu', 'begini', 'situ', 'sini', 'sana', 'situ', 'sini',
                'sana', 'situ', 'sini', 'sana', 'situ', 'apa', 'siapa', 'mengapa',
                'kenapa', 'bagaimana', 'kapan', 'dimana', 'kemana', 'darimana', 'berapa',
                'apakah', 'siapakah', 'mengapakah', 'bagaimanakah', 'kapankah', 'dimanakah',
                'kemanakah', 'dari manakah', 'berapakah'
            }
            print("Using minimal fallback stopwords")
        
    def clean_text(self, text):
        """
        Clean and preprocess the input text by removing unwanted characters,
        normalizing whitespace, and performing other cleaning operations.
        
        Args:
            text (str): The input text to clean
            
        Returns:
            str: The cleaned text
        """
        print("\n=== clean_text ===")
        
        # Validate input
        if not text or not isinstance(text, str):
            print("Warning: Empty or invalid input text")
            return ""
            
        original_length = len(text)
        print(f"Input length: {original_length} characters")
        if original_length > 100:
            print(f"Input preview: {text[:100]}...")
        else:
            print(f"Input: {text}")
            
        try:
            # Make a copy to avoid modifying the original
            cleaned = text
            
            # Convert to lowercase
            cleaned = cleaned.lower()
            print("Applied: Converted to lowercase")
            
            # Remove URLs (http, https, www, ftp)
            cleaned = re.sub(
                r'\b(?:https?|ftp)://[-a-zA-Z0-9+&@#/%?=~_|!:,.;]*[-a-zA-Z0-9+&@#/%=~_|]',
                ' ', cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r'www\.[^\s]+', ' ', cleaned, flags=re.IGNORECASE)
            print("Applied: Removed URLs")
            
            # Remove email addresses
            cleaned = re.sub(
                r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?',
                ' ', cleaned)
            print("Applied: Removed email addresses")
            
            # Remove mentions and hashtags
            cleaned = re.sub(r'[@#][a-zA-Z0-9_]+', ' ', cleaned)
            print("Applied: Removed mentions and hashtags")
            
            # Remove HTML/XML tags
            cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
            print("Applied: Removed HTML/XML tags")
            
            # Remove special characters but keep letters, numbers, and basic punctuation
            # Keep Indonesian characters (à-ÿ) and common punctuation (.!?,;:)
            cleaned = re.sub(r'[^a-z0-9\sà-ÿ.!?,;:]', ' ', cleaned)
            print("Applied: Removed special characters")
            
            # Normalize whitespace and clean up
            cleaned = ' '.join(cleaned.split())
            print("Applied: Normalized whitespace")
            
            # Log results
            cleaned_length = len(cleaned)
            print(f"Cleaning complete. Length: {original_length} -> {cleaned_length} "
                  f"(Removed {max(0, original_length - cleaned_length)} characters)")
            
            if cleaned_length > 0:
                print(f"Cleaned preview: {cleaned[:200]}{'...' if len(cleaned) > 200 else ''}")
            else:
                print("Warning: Cleaned text is empty")
            
            return cleaned if cleaned.strip() else ""
            
        except Exception as e:
            print(f"Error in clean_text: {str(e)}")
            return ""  # Return empty string if error occurs
        
    # Dictionary untuk normalisasi kata-kata tidak baku
    _NORMALIZATION_MAP = {
        # Common abbreviations and slang
        'yg': 'yang', 'dg': 'dengan', 'dgn': 'dengan', 'tdk': 'tidak',
        'gak': 'tidak', 'ga': 'tidak', 'gx': 'tidak', 'g': 'tidak',
        'jg': 'juga', 'jga': 'juga', 'jgn': 'jangan', 'udh': 'sudah',
        'udah': 'sudah', 'sdh': 'sudah', 'sya': 'saya', 'sy': 'saya',
        'aku': 'saya', 'gw': 'saya', 'gua': 'saya', 'gue': 'saya',
        'lu': 'kamu', 'loe': 'kamu', 'lo': 'kamu', 'elu': 'kamu',
        'ente': 'kamu', 'kamu': 'anda',
        'nyokap': 'ibu', 'emak': 'ibu', 'mama': 'ibu', 'mamah': 'ibu',
        'bokap': 'ayah', 'bapak': 'ayah', 'papa': 'ayah', 'papah': 'ayah',
        'om': 'paman', 'tante': 'bibi', 'temen': 'teman', 'tmn': 'teman',
        'kalo': 'kalau', 'klo': 'kalau', 'kl': 'kalau', 'klw': 'kalau',
        'aja': 'saja', 'ajah': 'saja', 'aj': 'saja',
        'sih': '', 'deh': '', 'dong': '', 'kok': '', 'ya': '', 'yah': '',
        'nih': 'ini', 'tuh': 'itu', 'gitu': 'begitu', 'gini': 'begini',
        'banget': 'sangat', 'bgt': 'sangat', 'bngt': 'sangat',
        'sgt': 'sangat', 'amat': 'sangat', 'sekali': 'sangat',
        'bener': 'benar', 'bner': 'benar', 'bkn': 'bukan',
        'enggak': 'tidak', 'nggak': 'tidak', 'ngga': 'tidak',
        'gpp': 'tidak apa-apa', 'gapapa': 'tidak apa-apa',
        'gimana': 'bagaimana', 'gmn': 'bagaimana', 'gmana': 'bagaimana',
        'gakbisa': 'tidak bisa', 'gabisa': 'tidak bisa',
        'gbs': 'tidak bisa', 'gakboleh': 'tidak boleh',
        'gaboleh': 'tidak boleh', 'gblh': 'tidak boleh',
        'gakmau': 'tidak mau', 'gamau': 'tidak mau',
        'gausah': 'tidak usah', 'gausa': 'tidak usah',
        'gatau': 'tidak tahu', 'gatau': 'tidak tahu',
        'kagak': 'tidak', 'kgk': 'tidak', 'gx': 'tidak',
        'mksd': 'maksud', 'mksud': 'maksud', 'mskpn': 'meskipun',
        'mskpun': 'meskipun', 'msk': 'meski', 'pake': 'pakai',
        'pke': 'pakai', 'pkoknya': 'pokoknya', 'sm': 'sama',
        'sma': 'sama', 'smpe': 'sampai', 'sampe': 'sampai',
        'sampek': 'sampai', 'skrg': 'sekarang', 'skrng': 'sekarang',
        'skr': 'sekarang', 'skalian': 'sekalian', 'skli': 'sekali',
        'sll': 'selalu', 'slalu': 'selalu', 'slh': 'salah',
        'smoga': 'semoga', 'smga': 'semoga', 'smg': 'semoga',
        'smw': 'semua', 'smua': 'semua', 'spt': 'seperti',
        'sptinya': 'sepertinya', 'sptnya': 'sepertinya',
        'tgl': 'tanggal', 'tggl': 'tanggal', 'yg': 'yang'
    }

    def normalize_text(self, text):
        """
        Normalize text by converting slang, abbreviations, and common variations
        to their standard forms in Indonesian.
        
        Args:
            text (str): Input text to normalize
            
        Returns:
            str: Normalized text with standard forms
        """
        print("\n=== normalize_text ===")
        
        # Validate input
        if not text or not isinstance(text, str) or not text.strip():
            print("Warning: Empty or invalid input text")
            return ""
            
        try:
            # Convert to lowercase for case-insensitive matching
            text_lower = text.lower()
            words = text_lower.split()
            
            # Apply normalizations
            normalized_words = []
            for word in words:
                # Remove any non-alphanumeric characters from the end
                clean_word = word.rstrip('.,!?;:')
                # Get the normalized form or keep original if not in map
                normalized = self._NORMALIZATION_MAP.get(clean_word, clean_word)
                if normalized:  # Only add if not empty string
                    normalized_words.append(normalized)
            
            # Join back to string and clean up
            result = ' '.join(normalized_words)
            # Remove any extra spaces
            result = ' '.join(result.split())
            
            # Log the result
            print(f"Normalized: {text[:50]}... -> {result[:50]}...")
            return result
            
        except Exception as e:
            print(f"Error in normalize_text: {str(e)}")
            # Return original text if error occurs
            return text
    
    def remove_punctuation(self, text):
        """Remove punctuation"""
        print("\n=== remove_punctuation ===")
        print("Before punctuation removal:", text[:100] + "...")
        
        try:
            # Define custom punctuation including common Indonesian punctuations
            custom_punctuation = string.punctuation + '”“‘’' + '…' + '–' + '—' + '\'' + '"'
            
            # Create a translation table
            translator = str.maketrans('', '', custom_punctuation)
            
            # Remove punctuation
            result = text.translate(translator)
            
            # Remove any remaining special characters
            result = re.sub(r'[^\w\s]', ' ', result)
            
            # Clean up spaces
            result = ' '.join(result.split())
            
            print("After punctuation removal:", result[:100] + "...")
            return result
            
        except Exception as e:
            print(f"Error in remove_punctuation: {str(e)}")
            return text  # Return original text if error occurs
    
    def tokenize(self, text):
        """
        Tokenize text into words using NLTK's word_tokenize with fallback to simple split.
        
        Args:
            text (str): Input text to tokenize
            
        Returns:
            list: List of tokens
        """
        print("\n=== tokenize ===")
        
        # Validate input
        if not text or not isinstance(text, str):
            print("Warning: Empty or invalid text for tokenization")
            return []
            
        print(f"Input length: {len(text)} characters")
        print("Sample input:", text[:100] + ("..." if len(text) > 100 else ""))
        
        tokens = []
        
        # First try with NLTK's word_tokenize
        try:
            # Ensure punkt tokenizer is available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                print("Downloading punkt tokenizer...")
                nltk.download('punkt', quiet=True)
            
            # Tokenize using NLTK
            tokens = word_tokenize(text)
            
            # Filter out empty tokens and whitespace
            tokens = [token for token in tokens if token.strip()]
            
            print(f"Tokenized with NLTK: {len(tokens)} tokens")
            if tokens:
                print("Sample tokens (NLTK):", tokens[:min(10, len(tokens))])
                
            return tokens
            
        except Exception as e:
            print(f"NLTK tokenization failed: {str(e)}")
            print("Falling back to simple whitespace tokenizer")
        
        # Fallback to simple whitespace tokenizer if NLTK fails
        try:
            # Simple whitespace tokenizer with some basic splitting
            tokens = []
            for word in text.split():
                # Split on common punctuation
                word_tokens = re.findall(r"\w+(?:[-']\w+)*|['\".,!?;:]|\S+", word)
                tokens.extend(w for w in word_tokens if w.strip())
            
            print(f"Tokenized with fallback: {len(tokens)} tokens")
            if tokens:
                print("Sample tokens (fallback):", tokens[:min(10, len(tokens))])
                
            return tokens
            
        except Exception as e:
            print(f"Fallback tokenization failed: {str(e)}")
            # Last resort - split on whitespace
            tokens = [t for t in text.split() if t.strip()]
            print(f"Using simple split: {len(tokens)} tokens")
            return tokens
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from the list of tokens.
        
        Args:
            tokens (list): List of tokens to process
            
        Returns:
            list: List of tokens with stopwords removed
        """
        print("\n=== remove_stopwords ===")
        
        # Validate input
        if not tokens or not isinstance(tokens, (list, tuple)):
            print("Warning: No tokens to process or invalid input type")
            return []
            
        print(f"Input tokens: {len(tokens)}")
        if tokens:
            print("Sample input tokens:", tokens[:min(10, len(tokens))])
        
        try:
            # Ensure stopwords are initialized
            if not hasattr(self, 'stop_words') or not self.stop_words:
                print("Stopwords not initialized, initializing...")
                self._initialize_stopwords()
            
            # Convert stopwords to lowercase for case-insensitive matching
            stopwords_lower = {sw.lower() for sw in self.stop_words}
            
            # Filter out stopwords (case-insensitive check)
            filtered_tokens = [
                token for token in tokens 
                if (isinstance(token, str) and 
                    token.lower() not in stopwords_lower and 
                    token.strip())
            ]
            
            # If all tokens were removed, keep at least one non-empty token if available
            if not filtered_tokens and tokens:
                non_empty_tokens = [t for t in tokens if t and str(t).strip()]
                if non_empty_tokens:
                    filtered_tokens = [non_empty_tokens[0]]
            
            print(f"After stopword removal: {len(filtered_tokens)} tokens")
            if filtered_tokens:
                print("Sample output tokens:", filtered_tokens[:min(10, len(filtered_tokens))])
            else:
                print("Warning: All tokens were removed during stopword removal")
            
            return filtered_tokens
            
        except Exception as e:
            print(f"Error in remove_stopwords: {str(e)}")
            print("Returning original tokens")
            return tokens
    
    def stem_text(self, tokens, language='indonesian'):
        """
        Stem tokens based on specified language.
        
        This function supports both Indonesian (using Sastrawi) and English (using NLTK's PorterStemmer).
        If the specified stemmer fails, it will fall back to the next available option.
        
        Args:
            tokens (list): List of tokens to be stemmed
            language (str): Language for stemming ('indonesian' or 'english')
            
        Returns:
            list: List of stemmed tokens. Returns original tokens if stemming fails.
        """
        print("\n=== stem_text ===")
        
        # Validate input
        if not tokens or not isinstance(tokens, (list, tuple)):
            print("Warning: No tokens to process or invalid input type")
            return []
            
        print(f"Input tokens: {len(tokens)}")
        if tokens:
            print("Sample input tokens:", tokens[:min(10, len(tokens))])
        
        # If no tokens to process, return empty list
        if not any(isinstance(t, str) and t.strip() for t in tokens):
            print("No valid string tokens to stem")
            return []
        
        stemmed_tokens = []
        
        # Try language-specific stemmers first
        if language.lower() == 'indonesian':
            # Try Sastrawi first
            try:
                from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()
                
                for token in tokens:
                    if isinstance(token, str) and token.strip():
                        try:
                            stemmed = stemmer.stem(token)
                            if stemmed and stemmed.strip():  # Only add non-empty stems
                                stemmed_tokens.append(stemmed)
                        except Exception as e:
                            print(f"Warning: Error stemming token '{token}': {str(e)}")
                            stemmed_tokens.append(token)  # Keep original if stemming fails
                    else:
                        # Keep non-string tokens or empty strings as is
                        stemmed_tokens.append(token)
                
                print("Used Sastrawi stemmer for Indonesian")
                
            except ImportError:
                print("Sastrawi not available, trying NLTK's PorterStemmer...")
                # Fall through to NLTK stemmer
                language = 'english'
            except Exception as e:
                print(f"Error with Sastrawi stemmer: {str(e)}")
                print("Falling back to NLTK stemmer")
                language = 'english'
        
        # For English or as fallback for Indonesian
        if language.lower() == 'english' or not stemmed_tokens:
            try:
                from nltk.stem import PorterStemmer
                stemmer = PorterStemmer()
                
                # Only process if we haven't already stemmed with Sastrawi
                if not stemmed_tokens:
                    for token in tokens:
                        if isinstance(token, str) and token.strip():
                            try:
                                stemmed = stemmer.stem(token)
                                if stemmed and stemmed.strip():  # Only add non-empty stems
                                    stemmed_tokens.append(stemmed)
                            except Exception as e:
                                print(f"Warning: Error stemming token '{token}': {str(e)}")
                                stemmed_tokens.append(token)  # Keep original if stemming fails
                        else:
                            # Keep non-string tokens or empty strings as is
                            stemmed_tokens.append(token)
                
                print("Used NLTK's PorterStemmer")
                
            except ImportError:
                print("NLTK not available, skipping stemming")
                return tokens
            except Exception as e:
                print(f"Error with NLTK stemmer: {str(e)}")
                print("Skipping stemming")
                return tokens
        
        # If no tokens were processed (shouldn't happen with current logic)
        if not stemmed_tokens:
            print("Warning: No tokens were stemmed, returning original tokens")
            return tokens
        
        print(f"After stemming: {len(stemmed_tokens)} tokens")
        if stemmed_tokens:
            print("Sample output tokens:", stemmed_tokens[:min(10, len(stemmed_tokens))])
        
        return stemmed_tokens
    
    def preprocess_pipeline(self, text, steps=None, language='indonesian'):
        """
        Run preprocessing pipeline based on selected steps.
        
        This is the main method that orchestrates the text preprocessing pipeline.
        It applies each preprocessing step in sequence based on the provided steps list.
        
        Args:
            text (str): Input text to preprocess
            steps (list, optional): List of preprocessing steps to apply. 
                                 Default is ['clean', 'normalize', 'punctuation', 'tokenize', 'stopwords', 'stem']
            language (str): Language for language-specific processing ('indonesian' or 'english')
            
        Returns:
            str: Preprocessed text as a string
        """
        # Validate and set default steps if not provided
        valid_steps = ['clean', 'normalize', 'punctuation', 'tokenize', 'stopwords', 'stem']
        if steps is None:
            steps = valid_steps.copy()
        elif not isinstance(steps, (list, tuple)):
            print("Warning: steps must be a list, using default steps")
            steps = valid_steps.copy()
        else:
            # Filter out invalid steps
            steps = [step for step in steps if step in valid_steps]
        
        # Log start of preprocessing
        print("\n" + "="*80)
        print(f"PREPROCESSING PIPELINE - {language.upper()}")
        print("="*80)
        print(f"Input length: {len(text)} characters")
        print(f"Input preview: {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"Steps to apply: {', '.join(steps) if steps else 'None'}")
        
        # Validate input text
        if not text or not isinstance(text, str) or not text.strip():
            print("Warning: Empty or invalid input text")
            return ""
            
        # Update language if different and reinitialize stopwords
        language = language.lower()
        if language not in ['indonesian', 'english']:
            print(f"Warning: Unsupported language '{language}', defaulting to 'indonesian'")
            language = 'indonesian'
            
        if language != getattr(self, 'language', None):
            print(f"Updating language from {getattr(self, 'language', 'not set')} to {language}")
            self.language = language
            try:
                self._initialize_stopwords()
                print(f"Initialized stopwords for {language}")
            except Exception as e:
                print(f"Warning: Failed to initialize stopwords: {str(e)}")
        
        # Initialize result with input text
        result = text.strip()
        tokens = None
        
        try:
            # Track progress through the pipeline
            step_count = 1
            total_steps = len(steps)
            
            # Clean text (always first step if included)
            if 'clean' in steps:
                print(f"\n[{step_count}/{total_steps}] Cleaning text...")
                result = self.clean_text(result)
                print(f"After cleaning: {result[:200]}{'...' if len(result) > 200 else ''}")
                if not result.strip():
                    print("Warning: clean_text returned empty result")
                    return ""
                step_count += 1
            
            # Normalize text (after cleaning, before tokenization)
            if 'normalize' in steps and result:
                print(f"\n[{step_count}/{total_steps}] Normalizing text...")
                result = self.normalize_text(result)
                print(f"After normalization: {result[:200]}{'...' if len(result) > 200 else ''}")
                if not result.strip():
                    print("Warning: normalize_text returned empty result")
                    return ""
                step_count += 1
            
            # Remove punctuation (if needed before tokenization)
            if 'punctuation' in steps and result:
                print(f"\n[{step_count}/{total_steps}] Removing punctuation...")
                result = self.remove_punctuation(result)
                # Clean up any double spaces that might be left after punctuation removal
                result = re.sub(r'\s+', ' ', result).strip()
                print(f"After punctuation removal: {result[:200]}{'...' if len(result) > 200 else ''}")
                if not result.strip():
                    print("Warning: remove_punctuation returned empty result")
                    return ""
                step_count += 1
            
            # Tokenize (only if we still have text and tokenization is requested)
            if 'tokenize' in steps and result and result.strip():
                print(f"\n[{step_count}/{total_steps}] Tokenizing...")
                try:
                    tokens = self.tokenize(result)
                    if tokens and isinstance(tokens, list):
                        print(f"Tokenized into {len(tokens)} tokens")
                        if tokens:
                            print(f"Sample tokens: {tokens[:min(10, len(tokens))]}")
                        result = tokens
                    else:
                        print("Warning: Tokenization returned no valid tokens")
                        tokens = []
                        # If tokenization fails but we have text, split by space as fallback
                        if result.strip():
                            tokens = [t for t in result.split() if t.strip()]
                            print(f"Fallback to simple split: {len(tokens)} tokens")
                except Exception as e:
                    print(f"Error during tokenization: {str(e)}")
                    tokens = []
                    # If tokenization fails but we have text, split by space as fallback
                    if result.strip():
                        tokens = [t for t in result.split() if t.strip()]
                        print(f"Fallback to simple split after error: {len(tokens)} tokens")
                step_count += 1
            
            # Remove stopwords (requires tokenization)
            if 'stopwords' in steps and isinstance(tokens, list) and tokens:
                print(f"\n[{step_count}/{total_steps}] Removing stopwords...")
                try:
                    filtered_tokens = self.remove_stopwords(tokens)
                    if filtered_tokens is not None and isinstance(filtered_tokens, list):
                        tokens = filtered_tokens
                        print(f"After stopword removal: {len(tokens)} tokens")
                        if tokens:
                            print(f"Sample after stopword removal: {tokens[:min(10, len(tokens))]}")
                        result = tokens
                    else:
                        print("Warning: Stopword removal returned invalid result, using original tokens")
                except Exception as e:
                    print(f"Error during stopword removal: {str(e)}")
                    print("Using tokens without stopword removal")
                step_count += 1
            
            # Stemming (requires tokenization)
            if 'stem' in steps and isinstance(tokens, list) and tokens:
                print(f"\n[{step_count}/{total_steps}] Stemming tokens...")
                try:
                    stemmed_tokens = self.stem_text(tokens, language=language)
                    if stemmed_tokens is not None and isinstance(stemmed_tokens, list):
                        tokens = stemmed_tokens
                        print(f"After stemming: {len(tokens)} tokens")
                        if tokens:
                            print(f"Sample after stemming: {tokens[:min(10, len(tokens))]}")
                        result = tokens
                    else:
                        print("Warning: Stemming returned invalid result, using original tokens")
                except Exception as e:
                    print(f"Error during stemming: {str(e)}")
                    print("Using tokens without stemming")
                step_count += 1
            
            # Convert back to string if we have tokens
            if isinstance(result, list):
                try:
                    # Filter out any non-string tokens and empty strings
                    filtered_tokens = [str(token).strip() for token in result 
                                    if token is not None and str(token).strip()]
                    if filtered_tokens:
                        result = ' '.join(filtered_tokens)
                    else:
                        print("Warning: No valid tokens after filtering")
                        result = ''
                except Exception as e:
                    print(f"Error converting tokens to string: {str(e)}")
                    result = ' '.join([str(token) for token in result if token is not None])
            
            # Final cleanup
            if isinstance(result, str):
                result = re.sub(r'\s+', ' ', result).strip()
                
        except Exception as e:
            print(f"\n!!! ERROR in preprocessing pipeline: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return whatever we have so far, or empty string if nothing
            if not result:
                print("No result available, returning empty string")
                return ""
        
        # Log completion
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETE")
        print("="*80)
        if result and str(result).strip():
            result_str = str(result).strip()
            print(f"Final result length: {len(result_str)} characters")
            print(f"Preview: {result_str[:200]}{'...' if len(result_str) > 200 else ''}")
            
            # Additional stats for debugging
            if isinstance(tokens, list) and tokens:
                print(f"Final token count: {len(tokens)}")
                unique_tokens = set(str(t).lower() for t in tokens if t and str(t).strip())
                print(f"Unique tokens: {len(unique_tokens)}")
        else:
            print("Warning: Empty result after preprocessing")
            return ""
        
        return str(result).strip() if result is not None else ""


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