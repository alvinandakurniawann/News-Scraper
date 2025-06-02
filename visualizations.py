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
    # Keep track of original text for highlighting
    original_text = text
    
    # Preprocess the text if preprocessor is available
    if preprocessor and preprocessing_steps:
        # Preprocess the text in the same way as the model input
        processed_text = preprocessor.preprocess_pipeline(text, preprocessing_steps)
        print(f"[DEBUG] Using preprocessed text for highlighting (length: {len(processed_text)})")
    else:
        # Fallback to original text if no preprocessor provided
        processed_text = original_text
        print("[DEBUG] Using original text for highlighting")
    
    # Create a mapping of words to their highlighting info (case-insensitive)
    word_highlights = {}
    for word_info in important_words:
        word = word_info['word'].strip()
        if not word:  # Skip empty words
            continue
            
        # Determine color based on weight
        weight = word_info.get('weight', 0)
        if weight > 0:
            # Traffic light red for fake news indicators
            # Use a minimum opacity of 0.3 for better visibility
            opacity = max(0.3, min(abs(weight) * 2.0, 0.9))
            color = f"rgba(255, 0, 0, {opacity})"  # Bright traffic light red
            text_color = "#ffffff"  # White text for better contrast
        else:
            # Traffic light green for real news indicators
            # Use a minimum opacity of 0.3 for better visibility
            opacity = max(0.3, min(abs(weight) * 2.0, 0.9))
            color = f"rgba(0, 255, 0, {opacity})"  # Bright traffic light green
            text_color = "#000000"  # Black text for better contrast
        
        # Store both original and lowercase versions for matching
        word_highlights[word.lower()] = {
            'original': word,
            'color': color,
            'text_color': text_color,
            'weight': weight
        }
    
    # Split text into tokens while keeping track of their positions
    tokens = []
    word_pattern = re.compile(r'\b(\w+)\b')
    pos = 0
    
    while pos < len(processed_text):
        # Find the next word boundary
        match = word_pattern.search(processed_text, pos)
        if not match:
            # No more words, add the remaining text
            tokens.append((processed_text[pos:], False))
            break
            
        # Add text before the match
        if match.start() > pos:
            tokens.append((processed_text[pos:match.start()], False))
            
        # Add the matched word
        word = match.group(1)
        tokens.append((word, True))
        pos = match.end()
    
    # Process each token
    result = []
    debug_words = {'real', 'business'}
    
    # Print all important words for debugging
    print("\n[DEBUG] Important words to highlight:", [word for word in word_highlights.keys()])
    
    for token, is_word in tokens:
        if not is_word:
            result.append(token)
            continue
            
        # Check if this word (case-insensitive) is in our important words
        lower_word = token.lower()
        
        # Debug logging for specific words
        if lower_word in debug_words:
            print(f"[DEBUG] Checking word: '{token}' (lower: '{lower_word}')")
            print(f"[DEBUG] Word in highlights: {lower_word in word_highlights}")
            if lower_word in word_highlights:
                print(f"[DEBUG] Word info: {word_highlights[lower_word]}")
        
        if lower_word in word_highlights:
            highlight = word_highlights[lower_word]
            # Use the original casing from the word_highlights if available
            display_word = highlight.get('original', token)
            # Force the color to be fully opaque for better visibility
            color = highlight["color"]
            if 'rgba' in color:
                # Extract RGB values and force alpha to 0.8 for better visibility
                parts = color[5:-1].split(',')
                if len(parts) == 4:  # If it's an rgba color
                    r, g, b, _ = parts
                    color = f'rgba({r},{g},{b},0.8)'  # Fixed high opacity for visibility
            
            highlighted = (
                f'<span style="'
                f'background-color: {color}; '
                f'color: {highlight["text_color"]}; '
                f'padding: 1px 3px; '
                f'border-radius: 3px; '
                f'margin: 0 1px; '
                f'display: inline-block; '
                f'line-height: 1.5; '
                f'font-weight: 500; '
                f'box-shadow: 0 1px 2px rgba(0,0,0,0.3);'
                f'" class="highlight-word">{display_word}</span>'
            )
            result.append(highlighted)
        else:
            result.append(token)
    
    # Join all tokens back together
    return ''.join(result)