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
        # Cek ketersediaan TensorFlow/Keras sebelum memuat
        if not hasattr(tf, 'keras') or not hasattr(tf.keras.models, 'load_model'):
            print("Gagal memuat model Keras: Library TensorFlow/Keras tidak tersedia.")
            self.lstm_model = None
            return False

        try:
            # Coba muat model dengan opsi compile=False
            try:
                # Coba muat model langsung
                self.lstm_model = tf.keras.models.load_model(
                    model_path,
                    compile=False
                )
                
                # Kompilasi model dengan konfigurasi sederhana
                self.lstm_model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                print(f"Model LSTM berhasil dimuat dari: {model_path}")
                return True
                
            except Exception as e:
                print(f"Gagal memuat model dengan cara standar, mencoba pendekatan alternatif...")
                print(f"Error: {str(e)}")
                
                # Dapatkan jumlah fitur dari TF-IDF vectorizer
                num_features = len(self.vectorizer.get_feature_names_out()) if hasattr(self.vectorizer, 'get_feature_names_out') else 5000
                
                # Coba buat model dengan arsitektur yang sesuai dengan training
                try:
                    model = tf.keras.Sequential([
                        # Input shape: (batch_size, 1, num_features)
                        tf.keras.layers.InputLayer(input_shape=(1, num_features), name='input_layer'),
                        
                        # LSTM layer dengan 64 unit
                        tf.keras.layers.LSTM(64, return_sequences=False, name='lstm'),
                        
                        # Output layer untuk binary classification
                        tf.keras.layers.Dense(1, activation='sigmoid', name='output')
                    ])
                    
                    # Kompilasi model
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                        loss='binary_crossentropy',
                        metrics=['accuracy']
                    )
                    
                    # Coba muat bobot
                    model.load_weights(model_path)
                    self.lstm_model = model
                    print(f"Model LSTM berhasil dimuat dengan arsitektur custom (1, {num_features}) dari: {model_path}")
                    return True
                    
                except Exception as e2:
                    print(f"Gagal memuat model dengan arsitektur custom: {e2}")
                    
                    # Coba pendekatan terakhir: muat model dengan custom_objects
                    try:
                        custom_objects = {
                            'InputLayer': tf.keras.layers.InputLayer,
                            'LSTM': tf.keras.layers.LSTM,
                            'Dense': tf.keras.layers.Dense,
                            'Adam': tf.keras.optimizers.Adam
                        }
                        
                        self.lstm_model = tf.keras.models.load_model(
                            model_path,
                            custom_objects=custom_objects,
                            compile=False
                        )
                        
                        # Kompilasi model
                        self.lstm_model.compile(
                            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                            loss='binary_crossentropy',
                            metrics=['accuracy']
                        )
                        
                        print(f"Model LSTM berhasil dimuat dengan custom_objects dari: {model_path}")
                        return True
                        
                    except Exception as e3:
                        print(f"Gagal memuat model dengan custom_objects: {e3}")
                        
                        # Jika semua gagal, coba pendekatan terakhir dengan eksplisit shape
                        try:
                            # Coba dengan shape yang umum digunakan (5000 fitur)
                            model = tf.keras.Sequential([
                                tf.keras.layers.InputLayer(input_shape=(1, 5000), name='input_layer'),
                                tf.keras.layers.LSTM(64, return_sequences=False, name='lstm'),
                                tf.keras.layers.Dense(1, activation='sigmoid', name='output')
                            ])
                            
                            model.compile(
                                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                                loss='binary_crossentropy',
                                metrics=['accuracy']
                            )
                            
                            model.load_weights(model_path, by_name=True, skip_mismatch=True)
                            self.lstm_model = model
                            print(f"Model LSTM berhasil dimuat dengan shape default (1, 5000) dari: {model_path}")
                            return True
                            
                        except Exception as e4:
                            print(f"Gagal memuat model dengan shape default: {e4}")
                            raise e4
                    
        except FileNotFoundError:
            print(f"Gagal memuat Model LSTM: File tidak ditemukan di {model_path}")
            self.lstm_model = None
            return False
        except Exception as e:
            print(f"Gagal memuat Model LSTM: {e}")
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
        if not self.model_loaded:
            error_msg = "Model TF-IDF atau LSTM belum dimuat dengan benar."
            print(f"✗ {error_msg}")
            # Kembalikan hasil error
            if isinstance(text, str):
                 return {
                    'prediction': 'ERROR', 'confidence': 0.0,
                    'probabilities': {'fake': 0.0, 'real': 0.0},
                    'model_info': self.get_model_info(), 'error': error_msg
                 }
            else:
                 return [{
                    'prediction': 'ERROR', 'confidence': 0.0,
                    'probabilities': {'fake': 0.0, 'real': 0.0},
                    'model_info': self.get_model_info(), 'error': error_msg
                 } for _ in text] # Gunakan _ jika teks asli tidak diperlukan di sini

        try:
            # Teks harus dalam bentuk list untuk vectorizer
            texts = [text] if isinstance(text, str) else text

            # Vectorisasi teks
            # Perhatikan bahwa TF-IDF vectorizer membutuhkan input berupa list of strings
            # Model LSTM mungkin memerlukan input dalam bentuk sequence, tidak sparse matrix
            # Anda perlu menyesuaikan ini jika model LSTM Anda dilatih pada representasi sequence
            # Namun, jika model LSTM Anda dilatih pada output TF-IDF (Dense layer pertama),
            # maka transformasi TF-IDF sparse perlu diubah menjadi dense array.
            # Asumsi model LSTM dilatih pada dense representation dari TF-IDF.
            #text_vectorized = self.vectorizer.transform(texts).toarray() # Convert sparse to dense

            # Tambahkan dimensi 'time_steps' dengan ukuran 1 untuk input LSTM
            #text_vectorized_lstm = np.expand_dims(text_vectorized, axis=1)

            text_vectorized = self.vectorizer.transform(texts)
            text_vectorized_lstm = text_vectorized.toarray().reshape(text_vectorized.shape[0], 1, text_vectorized.shape[1])

            # Lakukan prediksi dengan model LSTM
            # Output model Keras predict_proba biasanya probability untuk setiap kelas
            # Asumsi output adalah array 2D, dengan sumbu 1 adalah probabilitas per kelas
            predictions_proba = self.lstm_model.predict(text_vectorized_lstm)

            # Asumsi model output 1 nilai (sigmoid untuk binary classification)
            # Jika output 1 nilai (sigmoid), ubah menjadi 2 nilai probabilitas
            if predictions_proba.shape[-1] == 1:
                 fake_probabilities = predictions_proba.flatten() # Probabilitas kelas positif (misal FAKE)
                 real_probabilities = 1 - fake_probabilities # Probabilitas kelas negatif (misal REAL)
                 probabilities = np.vstack((real_probabilities, fake_probabilities)).T # Shape (n_samples, 2)
            else: # Asumsi output 2 nilai (softmax untuk binary classification)
                 probabilities = predictions_proba # Shape (n_samples, 2)

            # Tentukan prediksi (kelas dengan probabilitas tertinggi)
            # Asumsi kelas_id 1 adalah FAKE, 0 adalah REAL
            predicted_classes_idx = np.argmax(probabilities, axis=1)
            # Mapping index ke nama kelas (ini perlu disesuaikan dengan urutan kelas saat training)
            # Umumnya Keras/TensorFlow output index 0 untuk kelas pertama, 1 untuk kedua, dst.
            # Jika model dilatih dengan label 0=REAL, 1=FAKE, maka index 1 adalah FAKE
            class_mapping = {0: 'REAL', 1: 'FAKE'} # Sesuaikan jika urutan kelas berbeda
            predicted_classes = [class_mapping[idx] for idx in predicted_classes_idx]

            # Format hasil
            results = []
            for i, txt in enumerate(texts):
                fake_prob = float(probabilities[i, 1]) # Probabilitas FAKE (sesuaikan index jika perlu)
                real_prob = float(probabilities[i, 0]) # Probabilitas REAL (sesuaikan index jika perlu)
                confidence = float(np.max(probabilities[i]))

                results.append({
                    # 'text': txt, # Tidak perlu menampilkan teks di sini
                    'prediction': predicted_classes[i],
                    'confidence': confidence,
                    'probabilities': {
                        'fake': fake_prob,
                        'real': real_prob
                    },
                    'model_info': self.get_model_info()
                })

            return results[0] if isinstance(text, str) else results # Return single dict or list of dicts

        except Exception as e:
            error_msg = f"Error saat melakukan prediksi TF-IDF+LSTM: {str(e)}"
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
                    # 'text': t, # Tidak perlu menampilkan teks di hasil error batch
                    'prediction': 'ERROR',
                    'confidence': 0.0,
                    'probabilities': {'fake': 0.0, 'real': 0.0},
                    'model_info': self.get_model_info(),
                    'error': error_msg
                } for t in texts]


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