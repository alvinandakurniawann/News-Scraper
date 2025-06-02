# main.py
"""
Main Streamlit application untuk News Scraper & Fake News Detector
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import custom modules
from news_extractor import NewsExtractor
from text_preprocessor import TextPreprocessor
from tfidf_logreg_detector import TfidfLogregDetector
from database_supabase import HistoryDatabase
from config import Config
from visualizations import (
    create_confidence_gauge, 
    create_probability_chart, 
    highlight_important_words
)
from utils import process_batch_urls


def main():
    """Main application function"""
    st.set_page_config(
        page_title="News Scraper & Fake News Detector",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 News Scraper & Fake News Detector")
    st.markdown("---")
    
    # Validate Supabase configuration
    if not Config.validate_supabase_config():
        st.error("⚠️ Supabase configuration not found!")
        st.info("""
        Please set your Supabase credentials:
        1. Create a file `.streamlit/secrets.toml` in your project
        2. Add your Supabase URL and Key:
        ```toml
        SUPABASE_URL = "https://your-project.supabase.co"
        SUPABASE_KEY = "your-anon-public-key"
        ```
        """)
        st.stop()
    
    # Initialize session state for preprocessing steps if not exists
    if 'preprocessing_steps' not in st.session_state:
        st.session_state.preprocessing_steps = ['clean', 'punctuation', 'tokenize', 'stopwords', 'stem']
    
    # Initialize components
    extractor = NewsExtractor()
    preprocessor = TextPreprocessor()
    st.session_state.preprocessor = preprocessor  # Store in session state
    
    # Get model configuration
    model_config = Config.get_model_config()
    
    # Initialize detector with the default model
    default_model = model_config["available_models"][0]
    
    # Helper function to create detector instance
    def create_detector(model_info):
        if model_info["type"] == "tfidf":
            return TfidfLogregDetector(model_path=model_info["path"])
        else:
            st.error(f"Tipe model tidak didukung: {model_info['type']}")
            return None
    
    # Initialize detector with the default model
    detector = create_detector(default_model)
    if detector is None or not detector.model_loaded:
         st.error("Gagal menginisialisasi detector default.")
         st.stop()
    
    # Store detector and model config in session state
    st.session_state.detector = detector
    st.session_state.model_config = model_config
    st.session_state.current_model = default_model
    print(f"[DEBUG Main] Detector default diinisialisasi: {type(st.session_state.detector).__name__}, Loaded: {st.session_state.detector.model_loaded}, Info: {st.session_state.detector.get_model_info()['name']}")
    print(f"[DEBUG Main] Instance Detector di Session State: {id(st.session_state.detector)}")
    
    # Initialize Supabase database with config
    config = Config.get_supabase_config()
    history_db = HistoryDatabase(
        supabase_url=config["supabase_url"],
        supabase_key=config["supabase_key"]
    )
    
    # Initialize session state
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = None
    if 'preprocessed_text' not in st.session_state:
        st.session_state.preprocessed_text = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'explanation' not in st.session_state:
        st.session_state.explanation = None
    
    # Sidebar configuration
    setup_sidebar()
    preprocessing_steps = st.session_state.preprocessing_steps
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Check", "📊 Batch Processing", "📜 History", "📈 Analytics"])
    
    with tab1:
        single_check_tab(extractor, preprocessor, history_db, preprocessing_steps)
    
    with tab2:
        batch_processing_tab(extractor, preprocessor, history_db, preprocessing_steps)
    
    with tab3:
        history_tab(history_db)
    
    with tab4:
        analytics_tab(history_db)


def setup_sidebar():
    """Setup sidebar configuration"""
    # Initialize session state variables if they don't exist
    if 'preprocessing_steps' not in st.session_state:
        st.session_state.preprocessing_steps = Config.get_app_settings().get("preprocessing_defaults", 
                                                                          ['clean', 'punctuation', 'tokenize', 'stopwords'])
    
    if 'current_model' not in st.session_state:
        st.session_state.current_model = {}
    
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Display model info - only TF-IDF model is available
        selected_model_info = st.session_state.model_config["available_models"][0]  # Get first (and only) model
        st.session_state.current_model = selected_model_info
        
        # Display model information
        with st.expander("ℹ️ Informasi Model"):
            if 'detector' in st.session_state and st.session_state.detector:
                model_info = st.session_state.detector.get_model_info()
                st.write(f"**Nama:** {model_info.get('name', 'TF-IDF + Logistic Regression')}")
                st.write(f"**Tipe:** TF-IDF + Logistic Regression")
                st.write("**Status:** ✅ Loaded")
                st.write(f"**Path Model:** `{model_info.get('path', 'N/A')}`")
            else:
                st.warning("Model belum diinisialisasi")
        
        st.markdown("---")
        
        # Preprocessing options
        st.header("⚙️ Opsi Preprocessing")
        
        # Get available preprocessing steps from config or use defaults
        available_steps = Config.get_app_settings().get("available_preprocessing_steps", 
                                                      ['clean', 'punctuation', 'tokenize', 'stopwords', 'stem'])
        
        # Update preprocessing steps in session state
        selected_steps = st.multiselect(
            "Langkah Preprocessing:",
            available_steps,
            default=st.session_state.preprocessing_steps
        )
        
        # Update session state if selection changed
        if selected_steps != st.session_state.preprocessing_steps:
            st.session_state.preprocessing_steps = selected_steps
        
        st.markdown("### 📝 Keterangan:")
        st.markdown("""        
        - **clean**: Membersihkan teks (lowercase, hapus URL, email, dll)
        - **punctuation**: Menghapus tanda baca
        - **tokenize**: Memecah teks menjadi kata-kata
        - **stopwords**: Menghapus kata-kata umum
        - **stem**: Mengubah kata ke bentuk dasar
        """)
    


def single_check_tab(extractor, preprocessor, history_db, preprocessing_steps):
    """Single URL checking tab"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🔗 Input URL")
        
        # Check if URL exists in history
        url_input = st.text_input(
            "Enter news URL:",
            placeholder="https://www.detik.com/..."
        )
        
        # Check history
        if url_input:
            existing_record = history_db.check_url_exists(url_input)
            if existing_record:
                st.info(f"ℹ️ This URL was checked before on {existing_record['checked_at']}")
        
        # Extract and predict buttons
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🔍 Extract Only", type="secondary"):
                extract_only(url_input, extractor)
        
        with col_btn2:
            if st.button("🎯 Extract & Detect", type="primary"):
                # Ambil detector dari session_state
                detector = st.session_state.get('detector')
                print(f"[DEBUG SingleCheck] Mengambil detector dari session_state. Instance ID: {id(detector) if detector else 'None'}")
                if detector:
                    extract_and_detect(url_input, extractor, preprocessor, detector, 
                                     history_db, preprocessing_steps)
                else:
                    st.error("Detector belum diinisialisasi.")
    
    with col2:
        display_results()
    
    # Feature explanation section
    if st.session_state.explanation and st.session_state.extracted_data:
        display_explanation()


def extract_only(url_input, extractor):
    """Extract news without detection"""
    if url_input:
        with st.spinner("Extracting news..."):
            result = extractor.extract_from_url(url_input)
            
            if result['success']:
                st.session_state.extracted_data = result
                st.success("✅ Successfully extracted!")
            else:
                st.error(f"❌ Error: {result['error']}")
    else:
        st.warning("⚠️ Please enter URL first!")


def extract_and_detect(url_input, extractor, preprocessor, detector, history_db, preprocessing_steps):
    """Extract news and detect fake news"""
    print("--- Memulai Deteksi ---")
    print(f"[DEBUG ExtractDetect] Detector diterima: {type(detector).__name__}, Loaded: {detector.model_loaded}")
    print(f"[DEBUG ExtractDetect] Instance Detector diterima: {id(detector)}")
    print("[DEBUG ExtractDetect] Detector Info:", detector.get_model_info())
    if url_input:
        with st.spinner("Processing..."):
            # Extract
            result = extractor.extract_from_url(url_input)
            if result['success']:
                st.session_state.extracted_data = result
                # Preprocess
                full_text = f"{result['title']} {result['content']}"
                processed_text = preprocessor.preprocess_pipeline(full_text, preprocessing_steps)
                st.session_state.preprocessed_text = processed_text
                print(f"[DEBUG Main] Teks setelah preprocessing: {processed_text[:500]}...")
                # Gunakan objek detector dari parameter
                if detector.model_loaded:
                    prediction = detector.predict(processed_text)
                    st.session_state.prediction_result = prediction
                    explanation = detector.explain_prediction(processed_text)
                    st.session_state.explanation = explanation
                    
                    # Normalisasi format prediksi
                    pred = st.session_state.prediction_result
                    probs = pred['probabilities']
                    
                    # Handle perbedaan format (FAKE/REAL vs fake/real)
                    if 'FAKE' in probs and 'REAL' in probs:
                        fake_prob = probs['FAKE']
                        real_prob = probs['REAL']
                    elif 'fake' in probs and 'real' in probs:
                        fake_prob = probs['fake']
                        real_prob = probs['real']
                    else:
                        # Fallback jika format tidak dikenali
                        fake_prob = 0.5
                        real_prob = 0.5
                    
                    # Pastikan prediction dalam format yang konsisten
                    prediction_label = pred['prediction'].upper()
                    
                    # Save to history
                    history_record = {
                        'url': url_input,
                        'domain': result['domain'],
                        'title': result['title'],
                        'content': result['content'],
                        'prediction': prediction_label,
                        'confidence': pred['confidence'],
                        'fake_probability': fake_prob,
                        'real_probability': real_prob,
                        'checked_at': datetime.now()
                    }
                    history_db.add_record(history_record)
                    st.success("✅ Analysis complete!")
                else:
                    st.error("Model detector belum dimuat.")
            else:
                st.error(f"❌ Error: {result['error']}")
    else:
        st.warning("⚠️ Please enter URL first!")


def display_results():
    """Display extraction and prediction results"""
    if 'extracted_data' not in st.session_state or not st.session_state.extracted_data:
        st.warning("No data to display. Please extract content from a URL first.")
        return
        
    data = st.session_state.extracted_data
    
    # Check if data is valid
    if not isinstance(data, dict) or 'title' not in data or 'content' not in data:
        st.error("❌ Invalid data format. Failed to display results.")
        return
    
    try:
        # Display basic info
        st.subheader("📰 Extracted News")
        st.write(f"**Title:** {data.get('title', 'No Title')}")
        st.write(f"**Source:** {data.get('domain', 'Unknown')}")
        if data.get('url'):
            st.write(f"**URL:** [{data['url']}]({data['url']})")
        st.write(f"**Publish Date:** {data.get('publish_date', 'N/A')}")
        
        # Add export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # CSV Export
            csv = pd.DataFrame([{
                'title': data.get('title', ''),
                'source': data.get('domain', ''),
                'publish_date': data.get('publish_date', ''),
                'content': data.get('content', ''),
                'url': data.get('url', '')
            }]).to_csv(index=False)
            st.download_button(
                label="💾 Export as CSV",
                data=csv,
                file_name=f"news_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        with col2:
            # JSON Export
            json_data = {
                'title': data.get('title', ''),
                'source': data.get('domain', ''),
                'publish_date': data.get('publish_date', ''),
                'content': data.get('content', ''),
                'url': data.get('url', '')
            }
            st.download_button(
                label="📄 Export as JSON",
                data=json.dumps(json_data, indent=2, ensure_ascii=False),
                file_name=f"news_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        st.divider()
        
        # Create tabs for original and preprocessed content
        tab1, tab2 = st.tabs(["📝 Original Content", "🔧 Preprocessed Content"])
        
        with tab1:
            st.subheader("Original Content")
            if data.get('content'):
                st.markdown(
                    f"""
                    <div style="
                        max-height: 300px;
                        overflow-y: auto;
                        padding: 15px;
                        border: 1px solid #e0e0e0;
                        border-radius: 8px;
                        margin-bottom: 20px;
                        white-space: pre-wrap;
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        line-height: 1.6;
                        color: #333;
                        background-color: #f9f9f9;
                    ">
                        {data['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.warning("No content available to display.")
        
        with tab2:
            st.subheader("Preprocessed Content")
            if data.get('content'):
                # Get preprocessed text from session state if available
                preprocessed_text = st.session_state.get('preprocessed_text')
                
                if preprocessed_text:
                    st.markdown(
                        f"""
                        <div style="
                            max-height: 300px;
                            overflow-y: auto;
                            padding: 15px;
                            border: 1px solid #e0e0e0;
                            border-radius: 8px;
                            margin-bottom: 20px;
                            white-space: pre-wrap;
                            font-family: 'Courier New', monospace;
                            line-height: 1.6;
                            color: #333;
                            background-color: #f0f8ff;
                        ">
                            {preprocessed_text}
                        </div>
                        <div style="margin-top: 10px; font-size: 0.9em; color: #666; font-style: italic;">
                            This text has been preprocessed with: {', '.join(st.session_state.get('preprocessing_steps', []))}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Add copy button for preprocessed text
                    st.download_button(
                        label="📋 Copy Preprocessed Text",
                        data=preprocessed_text,
                        file_name="preprocessed_text.txt",
                        mime="text/plain"
                    )
                else:
                    st.info("Preprocessed text not available. Please run the detection to see preprocessed content.")
            else:
                st.warning("No content available to preprocess.")
        
        # Display prediction if available
        if 'prediction_result' in st.session_state and st.session_state.prediction_result:
            prediction = st.session_state.prediction_result
            
            st.subheader("🔍 Prediction Result")
            
            # Validate prediction data
            if isinstance(prediction, dict) and 'prediction' in prediction and 'confidence' in prediction:
                # Display prediction with color coding
                if prediction['prediction'] == 'FAKE':
                    st.error(f"❌ Prediction: **{prediction['prediction']}** (Confidence: {prediction['confidence']:.1%})")
                else:
                    st.success(f"✅ Prediction: **{prediction['prediction']}** (Confidence: {prediction['confidence']:.1%})")
                
                # Confidence gauge
                if 'confidence' in prediction and prediction['confidence'] is not None:
                    st.plotly_chart(
                        create_confidence_gauge(prediction['confidence']),
                        use_container_width=True
                    )
                
                # Probability distribution
                if 'probabilities' in prediction and prediction['probabilities'] is not None:
                    st.plotly_chart(
                        create_probability_chart(prediction['probabilities']),
                        use_container_width=True
                    )
            else:
                st.warning("⚠️ Prediction data is incomplete or invalid.")
    except Exception as e:
        st.error(f"❌ An error occurred while displaying results: {str(e)}")
        st.exception(e)  # This will show the full traceback in the app


def display_explanation():
    """Display feature explanation"""
    st.markdown("---")
    st.header("🔍 Feature Explanation")
    print("[DEBUG Explanation] Data Explanation:", st.session_state.explanation)
    
    # Get preprocessing steps from session state or use default
    preprocessing_steps = st.session_state.get('preprocessing_steps', 
        ['clean', 'normalize', 'remove_stopwords', 'stem'])
    
    col3, col4 = st.columns([2, 1])
    
    with col3:
        st.subheader("Important Words Highlighted")
        full_text = f"{st.session_state.extracted_data['title']} {st.session_state.extracted_data['content']}"
        
        # Get preprocessor from session state
        preprocessor = st.session_state.get('preprocessor')
        
        # Show first 2000 characters by default, with option to show more
        show_full_text = st.checkbox("Show full text", value=False, key="show_full_text")
        
        if show_full_text:
            text_to_show = full_text
            show_less = "(Showing full text)"
        else:
            text_to_show = full_text[:2000] + ("..." if len(full_text) > 2000 else "")
            show_less = ""
        
        # Highlight important words
        highlighted_html = highlight_important_words(
            text_to_show,
            st.session_state.explanation['important_words'],
            preprocessor=preprocessor,
            preprocessing_steps=preprocessing_steps
        )
        
        # Display the text with highlighting
        st.markdown(
            f"""
            <div style="
                max-height: 400px;
                overflow-y: auto;
                padding: 20px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-bottom: 10px;
                white-space: pre-wrap;
                line-height: 1.8;
                background-color: #2d2d2d;  /* Darker background */
                color: #f0f0f0;  /* Light text color */
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 15px;
            ">
                {highlighted_html}
            </div>
            <div style="color: #aaa; font-size: 0.9em; margin: 5px 0 15px 5px;">
                {show_less}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Show a note about preprocessing
        st.caption("💡 Highlighting shows words that influenced the model's decision. "
                 "The text is preprocessed in the same way as the model input.")
        
        # Show word count info
        word_count = len(full_text.split())
        st.caption(f"📊 Total words: {word_count} | Showing: {'all' if show_full_text or len(full_text) <= 2000 else 'first 2000 characters'}")
    
    with col4:
        st.subheader("Word Importance")
        words = st.session_state.explanation['important_words']
        if words:
            words_df = pd.DataFrame(words)
            if 'weight' in words_df.columns:
                words_df['weight'] = words_df['weight'].round(3)
                # Sort by absolute weight for better visualization
                words_df = words_df.iloc[words_df['weight'].abs().argsort()[::-1]]
            st.dataframe(words_df, hide_index=True)
            
            # Add color legend
            st.markdown("""
                <div style="margin-top: 10px; font-size: 0.9em; color: #666;">
                    <span style="color: #f00;">🔴</span> Words suggesting fake news<br>
                    <span style="color: #0a0;">🟢</span> Words suggesting real news
                </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Penjelasan fitur tidak tersedia untuk model ini.")


def batch_processing_tab(extractor, preprocessor, history_db, preprocessing_steps):
    """Batch processing tab"""
    st.header("📊 Batch URL Processing")
    
    urls_input = st.text_area(
        "Enter multiple URLs (one per line):",
        height=150,
        placeholder="https://www.detik.com/...\nhttps://www.kompas.com/...\nhttps://www.tribunnews.com/..."
    )
    
    if st.button("🚀 Process Batch", type="primary"):
        if urls_input:
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            
            if urls:
                # Ambil detector dari session_state
                detector = st.session_state.get('detector')
                if detector and detector.model_loaded:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    with st.spinner(f"Processing {len(urls)} URLs..."):
                        # Process batch menggunakan detector dari session_state
                        results = process_batch_urls(
                            urls, extractor, preprocessor, detector, preprocessing_steps
                        )
                        
                        # Update progress
                        progress_bar.progress(1.0)
                        status_text.text(f"Processed {len(results)} URLs")
                    
                    # Display results
                    display_batch_results(results, history_db)
                elif detector:
                     st.error("Model detector belum dimuat.")
                else:
                     st.error("Detector belum diinisialisasi.")
        else:
            st.warning("⚠️ Please enter at least one URL!")


def display_batch_results(results, history_db):
    """Display batch processing results"""
    st.subheader("📊 Batch Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    success_count = sum(1 for r in results if r.get('status') == 'success')
    fake_count = sum(1 for r in results if r.get('prediction') == 'FAKE')
    real_count = sum(1 for r in results if r.get('prediction') == 'REAL')
    avg_confidence = np.mean([r.get('confidence', 0) for r in results if r.get('confidence')])
    
    with col1:
        st.metric("Total URLs", len(results))
    with col2:
        st.metric("Successfully Processed", success_count)
    with col3:
        st.metric("Fake News Detected", fake_count)
    with col4:
        st.metric("Average Confidence", f"{avg_confidence:.2%}")
    
    # Display detailed results for each URL
    for idx, result in enumerate(results, 1):
        with st.expander(f"{idx}. {result.get('title', 'No title')}", expanded=False):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display URL and domain
                st.markdown(f"**URL:** [{result.get('url', 'N/A')}]({result.get('url', '')})")
                st.markdown(f"**Domain:** {result.get('domain', 'N/A')}")
                
                # Display prediction and confidence
                if result.get('status') == 'success':
                    prediction = result.get('prediction', 'UNKNOWN')
                    confidence = result.get('confidence', 0)
                    
                    # Prediction badge
                    pred_color = "red" if prediction == "FAKE" else "green"
                    st.markdown(
                        f"<div style='background-color:{pred_color}20; padding: 10px; border-radius: 5px; margin: 10px 0;'>"
                        f"<h3 style='color:{pred_color}; margin:0;'>Prediction: {prediction}</h3>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    
                    # Confidence gauge
                    st.plotly_chart(
                        create_confidence_gauge(confidence),
                        use_container_width=True
                    )
                    
                    # Probability distribution
                    st.plotly_chart(
                        create_probability_chart({
                            'FAKE': result.get('fake_probability', 0),
                            'REAL': result.get('real_probability', 0)
                        }),
                        use_container_width=True
                    )
                else:
                    st.error(f"Error processing URL: {result.get('error', 'Unknown error')}")
            
            with col2:
                if result.get('status') == 'success' and 'content' in result:
                    # Display important words if available
                    if 'important_words' in result:
                        st.markdown("### Important Words")
                        important_words = result['important_words']
                        
                        # Display word importance table
                        words_df = pd.DataFrame(important_words)
                        words_df['weight'] = words_df['weight'].round(4)
                        st.dataframe(
                            words_df,
                            column_config={
                                "word": "Word",
                                "weight": st.column_config.NumberColumn(
                                    "Importance",
                                    format="%.4f"
                                )
                            },
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        # Display highlighted text
                        st.markdown("### Text with Important Words Highlighted")
                        highlighted = highlight_important_words(
                            result['content'],
                            important_words
                        )
                        st.markdown(highlighted, unsafe_allow_html=True)
    
    # Save successful results to history
    results_df = pd.DataFrame([r for r in results if r.get('status') == 'success'])
    if not results_df.empty:
        st.subheader("Summary Table")
        st.dataframe(results_df[['url', 'domain', 'prediction', 'confidence']], hide_index=True)
        
        # Download results
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
        
        # Save to history database
        for _, result in results_df.iterrows():
            history_record = {
                'url': result['url'],
                'domain': result['domain'],
                'title': result['title'],
                'content': result.get('content', ''),
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'fake_probability': result['fake_probability'],
                'real_probability': result['real_probability'],
                'checked_at': datetime.now()
            }
            history_db.add_record(history_record)


def history_tab(history_db):
    """History tab"""
    st.header("📜 Checking History")
    
    # Get history
    history_df = history_db.get_history(limit=100)
    
    if not history_df.empty:
        display_history(history_df)
    else:
        st.info("No checking history yet. Start by checking some news URLs!")


def display_history(history_df):
    """Display history with filters"""
    # History metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Checks", len(history_df))
    with col2:
        fake_pct = (history_df['prediction'] == 'FAKE').sum() / len(history_df) * 100
        st.metric("Fake News %", f"{fake_pct:.1f}%")
    with col3:
        avg_conf = history_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_conf:.2%}")
    with col4:
        unique_domains = history_df['domain'].nunique()
        st.metric("Unique Domains", unique_domains)
    
    # Filter options
    col5, col6 = st.columns([1, 3])
    
    with col5:
        filter_prediction = st.selectbox(
            "Filter by Prediction:",
            ["All", "FAKE", "REAL"]
        )
    
    with col6:
        search_term = st.text_input("Search in titles:", "")
    
    # Apply filters
    filtered_df = history_df.copy()
    
    if filter_prediction != "All":
        filtered_df = filtered_df[filtered_df['prediction'] == filter_prediction]
    
    if search_term:
        filtered_df = filtered_df[
            filtered_df['title'].str.contains(search_term, case=False, na=False)
        ]
    
    # Display filtered history
    st.dataframe(filtered_df, hide_index=True)
    
    # Export history
    if st.button("📥 Export Full History"):
        csv = history_df.to_csv(index=False)
        st.download_button(
            label="Download History CSV",
            data=csv,
            file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def analytics_tab(history_db):
    """Analytics tab"""
    st.header("📈 Analytics Dashboard")
    
    history_df = history_db.get_history(limit=1000)
    
    if not history_df.empty:
        display_analytics(history_df)
    else:
        st.info("No data available for analytics. Start checking some news URLs!")


def display_analytics(history_df):
    """Display analytics charts"""
    # Convert checked_at to datetime
    history_df['checked_at'] = pd.to_datetime(history_df['checked_at'])
    
    # Prediction distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Distribution")
        pred_counts = history_df['prediction'].value_counts()
        fig_pie = px.pie(
            values=pred_counts.values,
            names=pred_counts.index,
            color_discrete_map={'FAKE': 'red', 'REAL': 'green'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("Confidence Distribution")
        fig_hist = px.histogram(
            history_df,
            x='confidence',
            nbins=20,
            title='Confidence Score Distribution'
        )
        # Fixed: Using update_xaxes instead of update_xaxis
        fig_hist.update_xaxes(title='Confidence Score')
        fig_hist.update_yaxes(title='Count')
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Domain analysis
    st.subheader("Top Domains Checked")
    domain_counts = history_df['domain'].value_counts().head(10)
    fig_domains = px.bar(
        x=domain_counts.values,
        y=domain_counts.index,
        orientation='h',
        labels={'x': 'Number of Checks', 'y': 'Domain'},
        title='Top 10 Most Checked Domains'
    )
    st.plotly_chart(fig_domains, use_container_width=True)
    
    # Fake news by domain
    st.subheader("Fake News Rate by Domain")
    domain_fake_rate = history_df.groupby('domain').agg({
        'prediction': lambda x: (x == 'FAKE').sum() / len(x) * 100
    }).round(1)
    domain_fake_rate = domain_fake_rate.sort_values('prediction', ascending=False).head(10)
    
    fig_fake_rate = px.bar(
        x=domain_fake_rate['prediction'],
        y=domain_fake_rate.index,
        orientation='h',
        labels={'x': 'Fake News Rate (%)', 'y': 'Domain'},
        title='Top 10 Domains by Fake News Rate'
    )
    st.plotly_chart(fig_fake_rate, use_container_width=True)


if __name__ == "__main__":
    main()