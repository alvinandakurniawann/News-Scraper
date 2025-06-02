# visualizations.py
"""
Module untuk membuat visualisasi data
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import re


def create_confidence_gauge(confidence: float) -> go.Figure:
    """Create a gauge chart for confidence score"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence Score"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def create_probability_chart(probabilities: Dict[str, float]) -> go.Figure:
    """Create a bar chart for fake/real probabilities"""
    # Normalisasi key menjadi uppercase untuk konsistensi
    prob_dict = {k.upper(): v for k, v in probabilities.items()}
    
    # Pastikan key yang diharapkan ada
    fake_key = 'FAKE' if 'FAKE' in prob_dict else 'fake'.upper()
    real_key = 'REAL' if 'REAL' in prob_dict else 'real'.upper()
    
    # Gunakan nilai default 0.0 jika key tidak ditemukan
    fake_prob = prob_dict.get(fake_key, 0.0)
    real_prob = prob_dict.get(real_key, 0.0)
    
    # Pastikan total probabilitas = 1.0
    total = fake_prob + real_prob
    if total > 0:
        fake_prob = fake_prob / total
        real_prob = real_prob / total
    
    fig = go.Figure([go.Bar(
        x=['FAKE', 'REAL'],
        y=[fake_prob, real_prob],
        marker_color=['red', 'green']
    )])
    
    fig.update_layout(
        title="Probability Distribution",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=300
    )
    
    return fig


def highlight_important_words(text: str, important_words: List[Dict], preprocessor=None, preprocessing_steps=None) -> str:
    """Highlight important words in text using HTML
    
    Args:
        text (str): The original text to highlight
        important_words (List[Dict]): List of dicts with 'word' and 'weight' keys
        preprocessor: Optional preprocessor instance with preprocess_pipeline method
        preprocessing_steps: List of preprocessing steps to apply (if preprocessor is provided)
    """
    if preprocessor and preprocessing_steps:
        # Preprocess the text in the same way as the model input
        processed_text = preprocessor.preprocess_pipeline(text, preprocessing_steps)
        # Use the preprocessed text for highlighting
        highlighted_text = processed_text
        print(f"[DEBUG] Using preprocessed text for highlighting (length: {len(processed_text)})")
    else:
        # Fallback to original text if no preprocessor provided
        highlighted_text = text
        print("[DEBUG] Using original text for highlighting")
    
    # Sort by word length (descending) to avoid partial replacements
    sorted_words = sorted(important_words, key=lambda x: len(x['word']), reverse=True)
    
    for word_info in sorted_words:
        word = word_info['word']
        weight = word_info['weight']
        
        # Skip empty words
        if not word.strip():
            continue
            
        # Determine color based on weight
        if weight > 0:
            # Reddish highlight for fake news indicators
            color = f"rgba(255, 100, 100, {min(abs(weight), 0.9)})"  # Brighter red
            text_color = "#fff"  # White text for better contrast
        else:
            # Greenish highlight for real news indicators
            color = f"rgba(100, 255, 100, {min(abs(weight), 0.9)})"  # Brighter green
            text_color = "#000"  # Black text for better contrast
        
        # Create highlighted version with better contrast
        highlighted = (
            f'<span style="'
            f'background-color: {color}; '
            f'color: {text_color}; '
            f'padding: 2px 5px; '
            f'border-radius: 3px; '
            f'margin: 0 1px; '
            f'display: inline-block; '
            f'line-height: 1.5; '
            f'font-weight: 500; '
            f'box-shadow: 0 1px 2px rgba(0,0,0,0.3);'
            f'" class="highlight-word">{word}</span>'
        )
        
        try:
            # Replace all occurrences (case-sensitive to match preprocessed text)
            pattern = re.compile(re.escape(word))
            highlighted_text = pattern.sub(highlighted, highlighted_text)
        except re.error as e:
            print(f"[WARNING] Error highlighting word '{word}': {e}")
    
    return highlighted_text