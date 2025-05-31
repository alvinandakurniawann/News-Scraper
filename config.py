# config.py
"""
Configuration module for application settings
"""

import os
import streamlit as st
from typing import Dict, Any


class Config:
    """Application configuration"""
    
    @staticmethod
    def get_supabase_config() -> Dict[str, str]:
        """
        Get Supabase configuration from environment or Streamlit secrets
        
        Returns:
            Dict with supabase_url and supabase_key
        """
        # Try to get from Streamlit secrets first
        if hasattr(st, 'secrets'):
            supabase_url = st.secrets.get("SUPABASE_URL", "")
            supabase_key = st.secrets.get("SUPABASE_KEY", "")
        else:
            # Fallback to environment variables
            supabase_url = os.getenv("SUPABASE_URL", "")
            supabase_key = os.getenv("SUPABASE_KEY", "")
        
        return {
            "supabase_url": supabase_url,
            "supabase_key": supabase_key
        }
    
    @staticmethod
    def validate_supabase_config() -> bool:
        """Validate if Supabase configuration is properly set"""
        config = Config.get_supabase_config()
        return bool(config["supabase_url"] and config["supabase_key"])
    
    @staticmethod
    def get_app_settings() -> Dict[str, Any]:
        """Get general application settings"""
        return {
            "app_name": "News Scraper & Fake News Detector",
            "version": "1.0.0",
            "max_batch_urls": 20,
            "history_limit": 1000,
            "cache_ttl": 3600,  # 1 hour
            "supported_domains": [
                "detik.com",
                "kompas.com",
                "tribunnews.com",
                "cnnindonesia.com",
                "liputan6.com"
            ],
            "preprocessing_defaults": [
                "clean",
                "punctuation",
                "tokenize",
                "stopwords"
            ]
        }
    
    @staticmethod
    def get_model_config() -> Dict[str, Any]:
        """Get model configuration"""
        # Use the directory where config.py is located as base
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        return {
            "available_models": [
                {
                    "name": "TF-IDF + Logistic Regression",
                    "path": os.path.join(base_dir, "models", "tfidf", "tfidf_logreg_tuned_pipeline.joblib"),
                    "type": "tfidf"
                },
                {
                    "name": "BERT + Logistic Regression",
                    "path": os.path.join(base_dir, "models", "bert+logreg", "logreg_tuned_bert_maxlen512.joblib"),
                    "type": "bert+logreg"
                },
                {
                    "name": "TF-IDF + LSTM",
                    "vectorizer_path": os.path.join(base_dir, "models", "tfidf_lstm", "tfidf_vectorizer_for_lstm.joblib"),
                    "model_path": os.path.join(base_dir, "models", "tfidf_lstm", "tfidf_lstm_model.keras"),
                    "type": "tfidf_lstm"
                }
            ],
            "confidence_threshold": 0.7,
            "max_text_length": 5000
        }