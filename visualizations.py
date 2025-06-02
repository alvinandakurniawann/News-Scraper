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
    if not important_words or not text:
        return text
    
    # Check if the text already contains HTML spans to avoid double-highlighting
    if '<span style="background-color:' in text:
        return text
        
    # Create a mapping of words to their highlighting info (case-insensitive)
    word_highlights = {}
    for word_info in important_words:
        word = word_info.get('word', '').strip()
        if not word or len(word) < 2:  # Skip empty or very short words
            continue
            
        # Determine color based on weight
        weight = word_info.get('weight', 0)
        if weight > 0:
            # Traffic light red for fake news indicators
            opacity = max(0.3, min(abs(weight) * 2.0, 0.9))
            color = f"rgba(255, 0, 0, {opacity})"  # Red for fake indicators
            text_color = "#ffffff"  # White text for better contrast
        else:
            # Traffic light green for real news indicators
            opacity = max(0.3, min(abs(weight) * 2.0, 0.9))
            color = f"rgba(0, 255, 0, {opacity})"  # Green for real indicators
            text_color = "#000000"  # Black text for better contrast
        
        # Store word info with original case for exact matching
        word_highlights[word.lower()] = {
            'original': word,
            'color': color,
            'text_color': text_color,
            'weight': weight
        }
    
    # If no valid words to highlight, return original text
    if not word_highlights:
        return text
    
    # Create a regex pattern to match any of the important words (case-insensitive)
    # Sort by length in descending order to match longer words first (e.g., 'new york' before 'new')
    words_sorted = sorted(word_highlights.keys(), key=len, reverse=True)
    
    # Create a pattern that matches whole words only (using word boundaries)
    pattern_str = r'(?<![\w-])(' + '|'.join(map(re.escape, words_sorted)) + r')(?![\w-])'
    pattern = re.compile(pattern_str, re.IGNORECASE)
    
    # Track which parts of the text we've already processed
    result_parts = []
    last_end = 0
    
    # Find all matches and build the result
    for match in pattern.finditer(text):
        # Add text before the match
        result_parts.append(text[last_end:match.start()])
        
        # Get the matched word and its lowercase version
        matched_word = match.group(1)
        lower_word = matched_word.lower()
        
        # Add the highlighted word
        if lower_word in word_highlights:
            highlight = word_highlights[lower_word]
            # Use the original word's case from the text
            display_word = matched_word
            highlighted = (
                f'<span style="background-color: {highlight["color"]}; '
                f'color: {highlight["text_color"]}; padding: 1px 3px; border-radius: 3px; '
                f'margin: 0 1px; display: inline-block; line-height: 1.5; font-weight: 500; '
                f'box-shadow: 0 1px 2px rgba(0,0,0,0.3);" class="highlight-word">'
                f'{display_word}</span>'
            )
            result_parts.append(highlighted)
        else:
            result_parts.append(matched_word)
            
        last_end = match.end()
    
    # Add any remaining text
    result_parts.append(text[last_end:])
    
    # Join all parts together
    return ''.join(result_parts)