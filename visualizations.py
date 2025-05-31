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
    fig = go.Figure([go.Bar(
        x=['FAKE', 'REAL'],
        y=[probabilities['fake'], probabilities['real']],
        marker_color=['red', 'green']
    )])
    
    fig.update_layout(
        title="Probability Distribution",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        height=300
    )
    
    return fig


def highlight_important_words(text: str, important_words: List[Dict]) -> str:
    """Highlight important words in text using HTML"""
    highlighted_text = text
    
    # Sort by word length (descending) to avoid partial replacements
    sorted_words = sorted(important_words, key=lambda x: len(x['word']), reverse=True)
    
    for word_info in sorted_words:
        word = word_info['word']
        weight = word_info['weight']
        
        # Determine color based on weight
        if weight > 0:
            color = f"rgba(255, 0, 0, {min(abs(weight), 1)})"  # Red for fake
        else:
            color = f"rgba(0, 255, 0, {min(abs(weight), 1)})"  # Green for real
        
        # Create highlighted version
        highlighted = f'<span style="background-color: {color}; padding: 2px 4px; border-radius: 3px;">{word}</span>'
        
        # Replace all occurrences (case-insensitive)
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted_text = pattern.sub(highlighted, highlighted_text)
    
    return highlighted_text