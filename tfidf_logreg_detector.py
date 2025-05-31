# fake_news_detector.py
"""
Module untuk deteksi fake news menggunakan model TF-IDF + Logistic Regression
"""

import os
import random
import joblib
import numpy as np
from typing import Dict, Any, List, Union
from pathlib import Path

class FakeNewsDetector:
    """Fake News Detection Model"""
    
    def __init__(self, model_path: str = None):
        """
        Inisialisasi model deteksi fake news
        
        Args:
            model_path: Path ke file model yang sudah di-training
        """
        self.model = None
        self.vectorizer = None
        self.model_loaded = False
        self.model_info = {
            'name': 'Not Loaded',
            'type': 'Unknown',
            'path': None,
            'status': 'Not Loaded'
        }
        
        if model_path:
            self.load_model(model_path)
    
    def get_model_info(self) -> dict:
        """
        Get information about the currently loaded model
        
        Returns:
            dict: Dictionary containing model information
        """
        return self.model_info
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a model from the specified path
        
        Args:
            model_path: Path to the model file
            
        Returns:
            bool: True if model was loaded successfully, False otherwise
        """
        try:
            self.model = joblib.load(model_path)
            self.model_loaded = True
            self.model_info = {
                'name': os.path.basename(model_path),
                'type': 'TF-IDF + Logistic Regression' if 'tfidf' in model_path.lower() else 'Unknown',
                'path': model_path,
                'status': 'Loaded'
            }
            print(f"Model berhasil dimuat dari: {model_path}")
            return True
        except Exception as e:
            print(f"Gagal memuat model: {e}")
            self.model_loaded = False
            self.model_info['status'] = f'Error: {str(e)}'
            return False
    
    def predict(self, text: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Melakukan prediksi teks menggunakan model yang sudah di-training
        
        Args:
            text: Teks atau list teks yang akan diprediksi
            
        Returns:
            Dict berisi hasil prediksi
        """
        if not self.model_loaded or self.model is None:
            raise RuntimeError("Model belum dimuat dengan benar")
        
        # Pastikan input berupa list
        texts = [text] if isinstance(text, str) else text
        
        try:
            # Lakukan prediksi
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(texts)
                predictions = self.model.predict(texts)
                
                # Format hasil
                if isinstance(text, str):
                    # Single prediction
                    fake_prob = probabilities[0][1]  # Asumsi indeks 1 adalah kelas 'FAKE'
                    real_prob = probabilities[0][0]  # Asumsi indeks 0 adalah kelas 'REAL'
                    
                    return {
                        'prediction': 'FAKE' if predictions[0] == 1 else 'REAL',
                        'confidence': max(fake_prob, real_prob),
                        'probabilities': {
                            'fake': float(fake_prob),
                            'real': float(real_prob)
                        },
                        'model_info': self.model_info
                    }
                else:
                    # Batch prediction
                    results = []
                    for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                        fake_prob = prob[1]  # Asumsi indeks 1 adalah kelas 'FAKE'
                        real_prob = prob[0]  # Asumsi indeks 0 adalah kelas 'REAL'
                        
                        results.append({
                            'text': texts[i],
                            'prediction': 'FAKE' if pred == 1 else 'REAL',
                            'confidence': max(fake_prob, real_prob),
                            'probabilities': {
                                'fake': float(fake_prob),
                                'real': float(real_prob)
                            },
                            'model_info': self.model_info
                        })
                    return results
            else:
                # Jika model tidak memiliki predict_proba
                predictions = self.model.predict(texts)
                
                if isinstance(text, str):
                    return {
                        'prediction': 'FAKE' if predictions[0] == 1 else 'REAL',
                        'confidence': 1.0,  # Default confidence
                        'probabilities': {
                            'fake': 1.0 if predictions[0] == 1 else 0.0,
                            'real': 0.0 if predictions[0] == 1 else 1.0
                        },
                        'model_info': self.model_info
                    }
                else:
                    return [{
                        'text': t,
                        'prediction': 'FAKE' if p == 1 else 'REAL',
                        'confidence': 1.0,
                        'probabilities': {
                            'fake': 1.0 if p == 1 else 0.0,
                            'real': 0.0 if p == 1 else 1.0
                        },
                        'model_info': self.model_info
                    } for t, p in zip(texts, predictions)]
                
        except Exception as e:
            error_msg = f"Error saat melakukan prediksi: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
    def explain_prediction(self, text: str, num_features: int = 10) -> Dict[str, Any]:
        """
        Menghasilkan penjelasan prediksi menggunakan koefisien model (untuk model linear)
        
        Args:
            text: Teks yang akan dijelaskan
            num_features: Jumlah fitur teratas yang akan ditampilkan
            
        Returns:
            Dict berisi kata-kata penting dan bobotnya
        """
        try:
            if not self.model_loaded or self.model is None:
                raise RuntimeError("Model belum dimuat dengan benar")
                
            # Dapatkan koefisien model (untuk model linear)
            if hasattr(self.model, 'coef_'):
                coefficients = self.model.coef_[0]
                feature_names = self.model.named_steps['tfidf'].get_feature_names_out()
                
                # Dapatkan indeks fitur terpenting (nilai absolut terbesar)
                top_indices = np.argsort(np.abs(coefficients))[-num_features:][::-1]
                
                # Dapatkan kata-kata dan bobotnya
                important_words = []
                for idx in top_indices:
                    word = feature_names[idx]
                    weight = float(coefficients[idx])
                    important_words.append({
                        'word': word,
                        'weight': weight,
                        'impact': 'fake' if weight > 0 else 'real'
                    })
                
                return {
                    'important_words': important_words,
                    'explanation_method': 'Model Coefficients'
                }
            else:
                # Fallback untuk model non-linear
                words = text.lower().split()
                important_words = []
                for word in set(words[:num_features]):  # Ambil kata unik
                    important_words.append({
                        'word': word,
                        'weight': random.uniform(-1, 1),
                        'impact': random.choice(['fake', 'real'])
                    })
                return {
                    'important_words': important_words,
                    'explanation_method': 'Random (model non-linear)'
                }
                
        except Exception as e:
            print(f"Error saat membuat penjelasan: {e}")
            # Kembalikan penjelasan kosong jika terjadi error
            return {
                'important_words': [],
                'explanation_method': 'Error',
                'error': str(e)
            }