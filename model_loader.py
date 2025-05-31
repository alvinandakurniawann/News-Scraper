import os
import joblib
import torch
import numpy as np
from typing import Tuple, Any, Dict, Union
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from sklearn.base import BaseEstimator
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

class ModelLoader:
    """
    A class to handle loading of pre-trained models for fake news detection.
    Supports both Logistic Regression and BERT embeddings.
    """
    
    def __init__(self):
        """Initialize the ModelLoader with local model paths."""
        self.models_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Model file paths
        self.model_files = {
            'logistic_regression': 'logreg_tuned_bert_maxlen512.joblib',
            'bert_embeddings': 'bert_embeddings_maxlen512_batch32.npz'
        }

    def _get_model_path(self, model_name: str) -> Path:
        """
        Get the path to a local model file.
        
        Args:
            model_name: Name of the model ('logistic_regression' or 'bert_embeddings')
            
        Returns:
            Path: Path to the model file
        """
        if model_name not in self.model_files:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.model_files.keys())}")
            
        model_path = Path(self.models_dir) / self.model_files[model_name]
        
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                f"Please make sure to place the file in the models directory."
            )
            
        return model_path
    
    def load_logistic_regression_model(self) -> BaseEstimator:
        """
        Load the logistic regression model with BERT embeddings.
        
        Returns:
            BaseEstimator: The loaded scikit-learn model
        """
        try:
            model_path = self._get_model_path('logistic_regression')
            print(f"Loading model from: {model_path}")
            
            # Load the model
            model = joblib.load(model_path)
            
            print("✓ Logistic Regression model loaded successfully")
            return model
            
        except Exception as e:
            error_msg = f"Error loading Logistic Regression model: {str(e)}"
            print(f"✗ {error_msg}")
            import traceback
            print(traceback.format_exc())
            raise RuntimeError(error_msg) from e
    
    def load_bert_embeddings(self) -> np.ndarray:
        """
        Load pre-computed BERT embeddings.
        
        Returns:
            np.ndarray: The pre-computed BERT embeddings
        """
        try:
            embeddings_path = self._get_model_path('bert_embeddings')
            print(f"Loading BERT embeddings from: {embeddings_path}")
            
            # Load the embeddings
            embeddings = np.load(embeddings_path)['arr_0']
            
            print(f"✓ BERT embeddings loaded successfully. Shape: {embeddings.shape}")
            return embeddings
            
        except Exception as e:
            error_msg = f"Error loading BERT embeddings: {str(e)}"
            print(f"✗ {error_msg}")
            import traceback
            print(traceback.format_exc())
            raise RuntimeError(error_msg) from e
    
    def load_tfidf_vectorizer(self) -> Any:
        """Load only the TF-IDF vectorizer."""
        _, vectorizer = self.load_logistic_regression_model()
        return vectorizer
    
    def load_bert_tokenizer(self) -> Any:
        """Load only the BERT tokenizer."""
        _, tokenizer = self.load_bert_model()
        return tokenizer
