"""
Module untuk deteksi fake news menggunakan model BERT + Logistic Regression
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path
from sklearn.base import BaseEstimator

# Import library BERT (transformers dan torch)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    BERT_AVAILABLE = True
except ImportError:
    BERT_AVAILABLE = False
    print("Warning: torch or transformers not installed. BERT+LogReg model will not be available.")

# Buat dummy objek jika import gagal agar kode tidak error saat inisialisasi
# Dummy objects are no longer needed as we rely on BERT_AVAILABLE checks
# and setting instance attributes to None if initialization fails.
# if not BERT_AVAILABLE:
#     class AutoTokenizer:
# ... (dummy classes removed)


class BertLogregDetector:
    """
    Fake News Detection Model BERT Embeddings + Logistic Regression.
    Membutuhkan file model Logistic Regression (.joblib) dan optional embeddings BERT (.npz)
    serta library transformers dan torch untuk vectorisasi teks baru.
    """
    
    def __init__(self, model_path: str, embeddings_path: str | None = None):
        """
        Inisialisasi model deteksi fake news BERT+LogReg.
        
        Args:
            model_path: Path ke file model Logistic Regression (.joblib).
            embeddings_path: Optional path ke file embeddings BERT pre-komputasi (.npz).
        """
        self.logreg_model: BaseEstimator = None
        self.embeddings: np.ndarray | None = None
        self.model_loaded = False
        
        # Atribut instance untuk komponen BERT
        self._bert_tokenizer = None
        self._bert_model = None
        self._bert_components_initialized = False # Status inisialisasi komponen BERT instance

        self.model_info = {
            'name': 'BERT + Logistic Regression',
            'type': 'bert+logreg',
            'logreg_path': model_path,
            'embeddings_path': embeddings_path,
            'status': 'Not Loaded'
        }

        # Inisialisasi tokenizer dan model BERT default hanya jika library tersedia di level modul
        if BERT_AVAILABLE:
            try:
                self._bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self._bert_model = AutoModel.from_pretrained("bert-base-uncased")
                self._bert_components_initialized = True # Set True jika berhasil
                print("✓ BERT Tokenizer and Model loaded successfully for instance.")
            except Exception as e:
                print(f"⚠️ Gagal memuat BERT Tokenizer atau Model untuk instance: {e}")
                self._bert_tokenizer = None # Pastikan None jika gagal
                self._bert_model = None # Pastikan None jika gagal
                self._bert_components_initialized = False # Set False jika gagal
        else:
            print("⚠️ BERT libraries not available globally. BERT components not initialized for instance.")
            self._bert_tokenizer = None
            self._bert_model = None
            self._bert_components_initialized = False

        # Load Logistic Regression model
        if model_path:
            self._load_logreg_model(model_path)
        
        # Load BERT embeddings jika path diberikan
        if embeddings_path:
            self._load_embeddings(embeddings_path)
            
        # Perbarui status model secara keseluruhan
        # Model dianggap loaded jika LogReg model berhasil dimuat.
        # Ketersediaan BERT components dicatat di _bert_components_initialized.
        if self.logreg_model is not None:
             self.model_loaded = True
             # Anda bisa memperbarui status info lebih detail jika embeddings atau BERT components dimuat
             status_detail = "Loaded (LogReg)"
             if self.embeddings is not None:
                 status_detail += " + Embeddings"
             if self._bert_components_initialized:
                 status_detail += " + BERT Components"
             self.model_info['status'] = status_detail
        else:
             self.model_loaded = False
             self.model_info['status'] = 'Error loading LogReg model'

    def _load_logreg_model(self, model_path: str) -> bool:
        try:
            loaded_model = joblib.load(model_path)
            if not isinstance(loaded_model, BaseEstimator): # Cek jika ini objek scikit-learn
                 raise TypeError("Model yang dimuat bukan scikit-learn BaseEstimator.")
            self.logreg_model = loaded_model
            print(f"Model Logistic Regression berhasil dimuat dari: {model_path}")
            return True
        except Exception as e:
            print(f"Gagal memuat Model Logistic Regression: {e}")
            self.logreg_model = None
            return False

    def _load_embeddings(self, embeddings_path: str) -> bool:
         try:
            if str(embeddings_path).endswith('.npz'):
                data = np.load(embeddings_path)
                # Coba arr_0, jika tidak ada ambil array pertama
                if 'arr_0' in data:
                    embeddings = data['arr_0']
                elif data.files:
                     # Ambil array pertama dari file .npz
                    first_key = data.files[0]
                    embeddings = data[first_key]
                else:
                     raise ValueError("File .npz kosong atau tidak memiliki array.")
            elif str(embeddings_path).endswith('.npy'):
                embeddings = np.load(embeddings_path)
            else:
                raise RuntimeError(f"Format file embeddings tidak dikenali: {embeddings_path}")
            
            self.embeddings = embeddings
            print(f"Embeddings BERT berhasil dimuat dari: {embeddings_path}. Shape: {self.embeddings.shape}")
            return True
         except Exception as e:
            print(f"Gagal memuat Embeddings BERT: {e}")
            self.embeddings = None
            return False

    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model
        
        Returns:
            dict: Dictionary containing model information
        """
        # Update status berdasarkan logreg_model
        self.model_info['status'] = 'Loaded' if self.logreg_model is not None else 'Error loading LogReg model'
        return self.model_info

    def bert_vectorize(self, text: str) -> np.ndarray:
        """
        Mengubah teks menjadi vektor embedding menggunakan model BERT default.
        Membutuhkan library transformers dan torch terinstall.
        """
        if not BERT_AVAILABLE or self._bert_tokenizer is None or self._bert_model is None:
            raise RuntimeError("Library transformers atau torch belum terinstall atau gagal dimuat.")

        try:
            inputs = self._bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = self._bert_model(**inputs)
                # Ambil pooled output (representasi [CLS])
                pooled = outputs.last_hidden_state[:, 0, :]  # shape: (batch, hidden_size)
            return pooled.numpy().squeeze()
        except Exception as e:
             raise RuntimeError(f"Error saat vectorisasi teks dengan BERT: {str(e)}") from e
    
    def predict(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Melakukan prediksi teks menggunakan vectorizer BERT dan model Logistic Regression.
        
        Args:
            text: Teks atau list teks yang akan diprediksi (setelah preprocessing).
                  Jika list, diasumsikan batch processing.
            
        Returns:
            Dict berisi hasil prediksi untuk single text, atau List[Dict] untuk batch.
        """
        if not self.model_loaded or self.logreg_model is None:
            raise RuntimeError("Model Logistic Regression belum dimuat dengan benar")
            
        # Periksa apakah komponen BERT untuk vektorisasi teks baru tersedia
        if not self._bert_components_initialized:
             # Jika BERT tidak tersedia di instance ini, tidak bisa melakukan prediksi teks baru
             error_msg = "Komponen BERT (tokenizer/model) tidak berhasil diinisialisasi. Tidak bisa melakukan prediksi dengan model BERT+LogReg untuk teks baru."
             print(f"✗ {error_msg}")
             # Kembalikan hasil dummy atau error
             if isinstance(text, str):
                 return {
                    'prediction': 'ERROR', 'confidence': 0.0,
                    'probabilities': {'fake': 0.0, 'real': 0.0},
                    'model_info': self.get_model_info(), 'error': error_msg
                }
             else:
                 return [{
                    # 'text': t, # Tidak perlu menampilkan teks di hasil error batch
                    'prediction': 'ERROR', 'confidence': 0.0,
                    'probabilities': {'fake': 0.0, 'real': 0.0},
                    'model_info': self.get_model_info(), 'error': error_msg
                } for t in text]

        texts = [text] if isinstance(text, str) else text
        results = []
        
        try:
            # Vectorisasi teks menggunakan BERT instance components
            # Vectorisasi batch ditangani oleh tokenizer/model BERT secara langsung
            inputs = self._bert_tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
            
            # Handle kasus ketika inputs['input_ids'] kosong (misal teks sangat pendek/kosong)
            if inputs['input_ids'].shape[0] == 0:
                 print("Warning: BERT tokenizer returned empty input_ids. Cannot vectorize.")
                 # Kembalikan hasil dummy atau error untuk teks kosong/invalid
                 dummy_result = {
                      'prediction': 'UNKNOWN', # Atau 'ERROR'
                      'confidence': 0.0,
                      'probabilities': {'fake': 0.5, 'real': 0.5},
                      'model_info': self.get_model_info(),
                      'warning': "BERT tokenizer returned empty input_ids."
                 }
                 if isinstance(text, str):
                      return dummy_result
                 else:
                      return [dummy_result] * len(texts) # Apply dummy result to all items in batch

            with torch.no_grad():
                 # Gunakan model BERT instance components
                outputs = self._bert_model(**inputs)
                # Ambil pooled output (representasi [CLS])
                # Pastikan outputs memiliki atribut last_hidden_state
                if hasattr(outputs, 'last_hidden_state'):
                    pooled = outputs.last_hidden_state[:, 0, :]
                elif hasattr(outputs, 'pooler_output'):
                     # Beberapa model BERT mungkin menggunakan pooler_output
                     pooled = outputs.pooler_output
                else:
                     raise AttributeError("BERT model output has neither 'last_hidden_state' nor 'pooler_output'")

            text_vectorized = pooled.numpy()

            # Lakukan prediksi dengan model LogReg
            probabilities = self.logreg_model.predict_proba(text_vectorized)
            predictions = self.logreg_model.predict(text_vectorized)

            # Format hasil
            # Asumsi indeks 1 adalah kelas 'FAKE' dan 0 adalah 'REAL' (sesuai urutan kelas LogReg)
            # Pastikan classes_ ada sebelum diakses
            if not hasattr(self.logreg_model, 'classes_'):
                 raise AttributeError("Logistic Regression model has no 'classes_' attribute.")

            fake_class_index = 1 if self.logreg_model.classes_[1] == 1 else 0
            real_class_index = 1 - fake_class_index

            if isinstance(text, str): # Input single text
                fake_prob = probabilities[0][fake_class_index]
                real_prob = probabilities[0][real_class_index]
                return {
                    'prediction': 'FAKE' if predictions[0] == self.logreg_model.classes_[fake_class_index] else 'REAL',
                    'confidence': float(max(fake_prob, real_prob)),
                    'probabilities': {
                        'fake': float(fake_prob),
                        'real': float(real_prob)
                    },
                    'model_info': self.get_model_info()
                }
            else: # Input list of texts (batch)
                 for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                    fake_prob = prob[fake_class_index]
                    real_prob = prob[real_class_index]
                    results.append({
                        'text': texts[i], # Sertakan teks di hasil batch untuk identifikasi
                        'prediction': 'FAKE' if pred == self.logreg_model.classes_[fake_class_index] else 'REAL',
                        'confidence': float(max(fake_prob, real_prob)),
                        'probabilities': {
                            'fake': float(fake_prob),
                            'real': float(real_prob)
                        },
                        'model_info': self.get_model_info()
                    })
                 return results

        except Exception as e:
            error_msg = f"Error saat melakukan prediksi BERT+LogReg: {str(e)}"
            print(f"✗ {error_msg}")
            # Jika prediksi gagal, kembalikan hasil error atau nilai default
            if isinstance(text, str):
                return {
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'probabilities': {'fake': 0.0, 'real': 0.0},
                    'model_info': self.get_model_info(),
                    'error': error_msg
                }
            else:
                texts = [text] if isinstance(text, str) else text
                return [{
                    'text': t,
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'probabilities': {'fake': 0.0, 'real': 0.0},
                    'model_info': self.get_model_info(),
                    'error': error_msg
                } for t in texts]

    def explain_prediction(self, text: str, num_features: int = 10) -> Dict[str, Any]:
         """
         Penjelasan fitur tidak didukung secara langsung untuk model BERT+LogReg dengan metode vectorisasi saat ini.
         Diperlukan implementasi penjelasan yang lebih canggih (seperti LIME, SHAP, atau attention weights).
         """
         print("Info: Feature explanation not available for BERT+LogReg model with current implementation.")
         return {
             'important_words': [], # Tidak ada kata-kata penting yang bisa langsung diekstrak seperti TF-IDF
             'explanation_method': 'Not available for BERT+LogReg'
         }

    def _load_bert_logreg_embeddings(self, embeddings_path: str) -> np.ndarray:
         """
         Memuat embeddings BERT dari file .npz atau .npy.
         Menangani format .npz yang mungkin berbeda (menggunakan array pertama yang tersedia).
         """
         try:
             if embeddings_path.endswith('.npz'):
                 with np.load(embeddings_path) as data:
                     # Coba ambil 'arr_0' dulu, jika tidak ada, ambil array pertama yang tersedia
                     if 'arr_0' in data:
                         embeddings = data['arr_0']
                     else:
                         # Ambil kunci pertama yang tersedia
                         first_key = list(data.keys())[0]
                         embeddings = data[first_key]
             elif embeddings_path.endswith('.npy'):
                 embeddings = np.load(embeddings_path)
             else:
                 raise ValueError(f"Format file embedding tidak didukung: {embeddings_path}. Gunakan .npz atau .npy")

             # Lakukan normalisasi jika diperlukan (sesuai dengan preprocessing saat training logreg)
             # Contoh: Normalisasi L2 (jika model LogReg dilatih dengan embeddings yang dinormalisasi)
             # embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
             
             return embeddings
         except FileNotFoundError:
             raise FileNotFoundError(f"File embeddings tidak ditemukan di: {embeddings_path}")
         except Exception as e:
             raise RuntimeError(f"Error loading BERT+LogReg embeddings: {str(e)}") 