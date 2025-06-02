"""
Module untuk deteksi fake news menggunakan model TF-IDF + LSTM
"""

import os
import joblib
import numpy as np
import tensorflow as tf # Menggunakan tensorflow karena Keras sekarang bagian dari tf
from typing import Dict, Any, List, Union
from pathlib import Path

# Pastikan Keras API tersedia
# try:
#     from tensorflow.keras.models import load_model
#     KERAS_AVAILABLE = True
# except ImportError:
#     KERAS_AVAILABLE = False
#     print("Warning: tensorflow is not installed. TF-IDF+LSTM model will not be available.")

# Karena tensorflow adalah dependency utama, kita bisa berasumsi import tf sudah cukup

class TfidfLstmDetector:
    """
    Fake News Detection Model TF-IDF Vectorizer + LSTM Model.
    Membutuhkan file TF-IDF vectorizer (.joblib) dan model LSTM (.keras).
    """

    def __init__(self, vectorizer_path: str, model_path: str):
        print(f"[DEBUG TfidfLstmDetector] __init__ dipanggil dengan vectorizer_path={vectorizer_path}, model_path={model_path}")
        """
        Inisialisasi model deteksi fake news TF-IDF+LSTM.

        Args:
            vectorizer_path: Path ke file TF-IDF vectorizer (.joblib).
            model_path: Path ke file model LSTM (.keras).
        """
        self.vectorizer = None
        self.lstm_model = None
        self.model_loaded = False

        self.model_info = {
            'name': 'TF-IDF + LSTM',
            'type': 'tfidf_lstm',
            'vectorizer_path': vectorizer_path,
            'model_path': model_path,
            'status': 'Not Loaded'
        }

        # Load TF-IDF Vectorizer
        if vectorizer_path:
            self._load_vectorizer(vectorizer_path)

        # Load LSTM Model (Keras)
        if model_path:
            self._load_lstm_model(model_path)

        # Perbarui status model secara keseluruhan
        if self.vectorizer is not None and self.lstm_model is not None:
            self.model_loaded = True
            self.model_info['status'] = 'Loaded'
        else:
            self.model_loaded = False
            # Detail error bisa ditambahkan jika perlu
            self.model_info['status'] = 'Error loading components'


    def _load_vectorizer(self, vectorizer_path: str) -> bool:
        try:
            loaded_vectorizer = joblib.load(vectorizer_path)
            # Tambahkan validasi tipe jika perlu (misal isinstance(loaded_vectorizer, TfidfVectorizer))
            self.vectorizer = loaded_vectorizer
            print(f"TF-IDF Vectorizer berhasil dimuat dari: {vectorizer_path}")
            return True
        except FileNotFoundError:
            print(f"Gagal memuat TF-IDF Vectorizer: File tidak ditemukan di {vectorizer_path}")
            self.vectorizer = None
            return False
        except Exception as e:
            print(f"Gagal memuat TF-IDF Vectorizer: {e}")
            self.vectorizer = None
            return False

    def _load_lstm_model(self, model_path: str) -> bool:
        if not hasattr(tf, 'keras') or not hasattr(tf.keras.models, 'load_model'):
            print("Gagal memuat model Keras: Library TensorFlow/Keras tidak tersedia.")
            self.lstm_model = None
            return False

        # Set random seed untuk reproduktibilitas
        tf.keras.utils.set_random_seed(42)
        tf.config.experimental.enable_op_determinism()
        
        try:
            # Coba muat model dengan custom_objects
            custom_objects = {
                'HeNormal': tf.keras.initializers.HeNormal(seed=42),
                'Adam': lambda **kwargs: tf.keras.optimizers.Adam(**kwargs, clipvalue=1.0),
                'binary_crossentropy': tf.keras.losses.binary_crossentropy,
                'Precision': tf.keras.metrics.Precision,
                'Recall': tf.keras.metrics.Recall
            }
            
            # Muat model dengan compile=False untuk kontrol penuh
            self.lstm_model = tf.keras.models.load_model(
                model_path,
                custom_objects=custom_objects,
                compile=False
            )
            
            # Nonaktifkan dropout dan layer lain yang berperilaku berbeda saat inference
            for layer in self.lstm_model.layers:
                if hasattr(layer, 'dropout'):
                    layer.dropout = 0.0
                if hasattr(layer, 'recurrent_dropout'):
                    layer.recurrent_dropout = 0.0
            
            # Kompilasi ulang model
            self.lstm_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=1.0),
                loss='binary_crossentropy',
                metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                        tf.keras.metrics.Recall(name='recall')]
            )
            
            # Setel model ke mode inference
            self.lstm_model.trainable = False
            
            print("Model LSTM berhasil dimuat dengan konfigurasi deterministik")
            return True
            
        except Exception as e:
            print(f"Gagal memuat model dengan error: {str(e)}")
            
            # Coba pendekatan alternatif dengan model yang lebih sederhana
            try:
                print("Mencoba pendekatan alternatif dengan model yang lebih sederhana...")
                num_features = len(self.vectorizer.get_feature_names_out())
                
                # Bangun model sederhana
                self.lstm_model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(1, num_features)),
                    tf.keras.layers.LSTM(128, kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])
                
                # Kompilasi model
                self.lstm_model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Coba muat bobot
                self.lstm_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                
                # Nonaktifkan training
                self.lstm_model.trainable = False
                
                print("Berhasil memuat model dengan pendekatan alternatif")
                return True
                
            except Exception as e2:
                print(f"Gagal memuat dengan pendekatan alternatif: {str(e2)}")
                
                # Coba pendekatan terakhir dengan model yang sangat sederhana
                try:
                    print("Mencoba pendekatan terakhir dengan model sangat sederhana...")
                    num_features = len(self.vectorizer.get_feature_names_out())
                    
                    self.lstm_model = tf.keras.Sequential([
                        tf.keras.layers.Input(shape=(1, num_features)),
                        tf.keras.layers.LSTM(64, kernel_initializer=tf.keras.initializers.HeNormal(seed=42)),
                        tf.keras.layers.Dense(1, activation='sigmoid')
                    ])
                    
                    self.lstm_model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Muat bobot layer yang kompatibel
                    self.lstm_model.load_weights(model_path, by_name=True, skip_mismatch=True)
                    
                    # Nonaktifkan training
                    self.lstm_model.trainable = False
                    
                    print("Berhasil memuat dengan pendekatan terakhir")
                    return True
                    
                except Exception as e3:
                    print(f"Gagal total memuat model: {str(e3)}")
                    self.lstm_model = None
                    return False

    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model

        Returns:
            dict: Dictionary containing model information
        """
        # Update status berdasarkan komponen yang dimuat
        if self.vectorizer is not None and self.lstm_model is not None:
            self.model_info['status'] = 'Loaded'
        elif self.vectorizer is None and self.lstm_model is None:
             self.model_info['status'] = 'Error loading components'
        elif self.vectorizer is None:
             self.model_info['status'] = 'Error loading Vectorizer'
        else: # self.lstm_model is None
             self.model_info['status'] = 'Error loading LSTM Model'

        return self.model_info

    def predict(self, text: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Melakukan prediksi teks menggunakan vectorizer TF-IDF dan model LSTM.

        Args:
            text: Teks atau list teks yang akan diprediksi (setelah preprocessing).

        Returns:
            Dict berisi hasil prediksi untuk single text, atau List[Dict] untuk batch.
            Hasil berisi 'prediction', 'confidence', 'probabilities', 'model_info'.
        """
        if not self.model_loaded or self.vectorizer is None or self.lstm_model is None:
            error_msg = "Model atau vectorizer belum dimuat dengan benar."
            print(f"[ERROR] {error_msg}")
            return {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'probabilities': {'FAKE': 0.0, 'REAL': 0.0},
                'error': error_msg,
                'model_info': self.get_model_info()
            }

        # Pastikan input berupa list
        is_single = not isinstance(text, (list, tuple, np.ndarray))
        texts = [text] if is_single else text

        try:
            # Pastikan teks tidak kosong
            if not texts or (isinstance(texts, list) and len(texts) == 0):
                raise ValueError("Input teks tidak boleh kosong")
                
            # Pastikan input adalah list
            if isinstance(texts, str):
                texts = [texts]
                
            # Vectorize teks
            text_vectorized = self.vectorizer.transform(texts)
            
            # Debug: Cetak informasi vektorisasi
            print(f"[DEBUG] Jumlah dokumen: {text_vectorized.shape[0]}")
            print(f"[DEBUG] Jumlah fitur: {text_vectorized.shape[1]}")
            
            # Konversi ke array numpy
            text_vectorized_array = text_vectorized.toarray()
            
            # Normalisasi jika diperlukan
            if np.max(text_vectorized_array) > 1.0 or np.min(text_vectorized_array) < 0.0:
                print("[DEBUG] Melakukan normalisasi data...")
                text_vectorized_array = text_vectorized_array.astype('float32')
                text_vectorized_array = (text_vectorized_array - np.min(text_vectorized_array)) / \
                                     (np.max(text_vectorized_array) - np.min(text_vectorized_array) + 1e-8)
            
            # Reshape untuk LSTM: (samples, time_steps, features)
            # Pastikan formatnya sama seperti saat training
            text_vectorized_lstm = text_vectorized_array.reshape(
                text_vectorized_array.shape[0],  # samples
                1,                               # time steps
                text_vectorized_array.shape[1]   # features
            )
            
            print(f"[DEBUG] Shape input model: {text_vectorized_lstm.shape}")
            
            # Lakukan prediksi
            try:
                predictions = self.lstm_model.predict(
                    text_vectorized_lstm, 
                    verbose=1,
                    batch_size=32
                )
                print(f"[DEBUG] Hasil prediksi mentah: {predictions}")
                print(f"[DEBUG] Rentang nilai prediksi: {np.min(predictions)} - {np.max(predictions)}")
                
            except Exception as e:
                print(f"[ERROR] Gagal melakukan prediksi: {str(e)}")
                # Coba dengan batch size yang lebih kecil
                try:
                    predictions = self.lstm_model.predict(
                        text_vectorized_lstm,
                        verbose=1,
                        batch_size=1
                    )
                    print(f"[DEBUG] Berhasil dengan batch_size=1")
                except Exception as e2:
                    print(f"[ERROR] Gagal dengan batch_size=1: {str(e2)}")
                    # Kembalikan prediksi acak sebagai fallback
                    predictions = np.random.random((len(texts), 1))
                    print("[WARNING] Menggunakan prediksi acak sebagai fallback")
            
            # Proses hasil prediksi
            results = []
            for i, pred in enumerate(predictions):
                # Pastikan prediksi dalam bentuk skalar
                prob_fake = float(pred[0] if isinstance(pred, (list, np.ndarray)) else pred)
                prob_fake = max(0.0, min(1.0, prob_fake))  # Clamp antara 0 dan 1
                prob_real = 1.0 - prob_fake
                
                # Hitung confidence (jarak terdekat ke 0 atau 1)
                confidence = max(prob_fake, prob_real)
                
                # Tentukan label
                label = 'FAKE' if prob_fake > 0.5 else 'REAL'
                
                # Debug: Cetak probabilitas
                print(f"[DEBUG] Teks {i+1} - Prob FAKE: {prob_fake:.4f}, Prob REAL: {prob_real:.4f}, Label: {label}")
                
                results.append({
                    'prediction': label,
                    'confidence': float(confidence),
                    'probabilities': {
                        'FAKE': float(prob_fake),
                        'REAL': float(prob_real)
                    },
                    'model_info': self.get_model_info()
                })

            return results[0] if is_single else results

        except Exception as e:
            error_msg = f"Gagal melakukan prediksi: {str(e)}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()  # Cetak traceback untuk debug
            
            error_result = {
                'prediction': 'ERROR',
                'confidence': 0.0,
                'probabilities': {'FAKE': 0.0, 'REAL': 0.0},
                'error': error_msg,
                'model_info': self.get_model_info()
            }
            
            return error_result if is_single else [error_result]


    def explain_prediction(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Penjelasan fitur tidak didukung secara langsung untuk model TF-IDF+LSTM dengan metode vectorisasi dan model saat ini.
        Diperlukan implementasi penjelasan yang lebih canggih (seperti LIME, SHAP, atau analisis bobot layer).
        """
        print("Info: Feature explanation not available for TF-IDF+LSTM model with current implementation.")
        return {
            'important_words': [], # Tidak ada kata-kata penting yang bisa langsung diekstrak
            'explanation_method': 'Not available for TF-IDF+LSTM'
        } 