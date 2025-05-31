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

# Import custom modules
from news_extractor import NewsExtractor
from text_preprocessor import TextPreprocessor
from fake_news_detector import FakeNewsDetector
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
    
    # Initialize components
    extractor = NewsExtractor()
    preprocessor = TextPreprocessor()
    
    # Get model configuration
    model_config = Config.get_model_config()
    
    # Initialize detector with the first available model by default
    default_model = model_config["available_models"][0]
    
    # Verify the model file exists
    model_path = default_model["path"]
    print(f"Looking for model at: {model_path}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Directory exists: {os.path.exists(os.path.dirname(model_path))}")
    print(f"Model file exists: {os.path.exists(model_path)}")
    
    if not os.path.exists(model_path):
        st.error(f"Model file not found at: {model_path}")
        st.stop()
    
    # Initialize the detector
    detector = FakeNewsDetector(model_path=model_path)
    
    # Store model config in session state
    if 'model_config' not in st.session_state:
        st.session_state.model_config = model_config
        st.session_state.current_model = default_model
    
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
    setup_sidebar(detector)
    preprocessing_steps = st.session_state.preprocessing_steps
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Single Check", "📊 Batch Processing", "📜 History", "📈 Analytics"])
    
    with tab1:
        single_check_tab(extractor, preprocessor, detector, history_db, preprocessing_steps)
    
    with tab2:
        batch_processing_tab(extractor, preprocessor, detector, history_db, preprocessing_steps)
    
    with tab3:
        history_tab(history_db)
    
    with tab4:
        analytics_tab(history_db)


def setup_sidebar(detector):
    """Setup sidebar configuration"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        model_options = [model["name"] for model in st.session_state.model_config["available_models"]]
        selected_model_name = st.selectbox(
            "Select Model:",
            model_options,
            index=0
        )
        
        # Update model if selection changed
        if (hasattr(st.session_state, 'current_model') and 
            st.session_state.current_model["name"] != selected_model_name):
            selected_model = next(m for m in st.session_state.model_config["available_models"] 
                               if m["name"] == selected_model_name)
            if detector.load_model(selected_model["path"]):
                st.session_state.current_model = selected_model
                st.success(f"Switched to model: {selected_model['name']}")
            else:
                st.error(f"Failed to load model: {selected_model['name']}")
        
        st.markdown("---")
        
        # Display current model info
        model_info = detector.get_model_info()
        with st.expander("ℹ️ Model Info"):
            st.write(f"**Name:** {model_info.get('name', 'N/A')}")
            st.write(f"**Type:** {model_info.get('type', 'N/A')}")
            st.write(f"**Status:** {'✅ ' if detector.model_loaded else '❌ '}{model_info.get('status', 'Unknown')}")
            st.write(f"**Path:** `{model_info.get('path', 'N/A')}`")
        
        st.markdown("---")
        
    # Initialize preprocessing_steps in session state if not exists
    if 'preprocessing_steps' not in st.session_state:
        st.session_state.preprocessing_steps = Config.get_app_settings()["preprocessing_defaults"]
    
    # Sidebar for preprocessing options
    with st.sidebar:
        st.header("⚙️ Preprocessing Options")
        
        # Update preprocessing steps in session state when changed
        st.session_state.preprocessing_steps = st.multiselect(
            "Preprocessing Steps:",
            ['clean', 'punctuation', 'tokenize', 'stopwords', 'stem'],
            default=st.session_state.preprocessing_steps
        )
        
        st.markdown("---")
        st.markdown("### 📝 Keterangan:")
        st.markdown("""        
        - **clean**: Membersihkan teks (lowercase, hapus URL, email, dll)
        - **punctuation**: Menghapus tanda baca
        - **tokenize**: Memecah teks menjadi kata-kata
        - **stopwords**: Menghapus kata-kata umum
        - **stem**: Mengubah kata ke bentuk dasar
        """)
    


def single_check_tab(extractor, preprocessor, detector, history_db, preprocessing_steps):
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
                extract_and_detect(url_input, extractor, preprocessor, detector, 
                                 history_db, preprocessing_steps)
    
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
    if url_input:
        with st.spinner("Processing..."):
            # Extract
            result = extractor.extract_from_url(url_input)
            
            if result['success']:
                st.session_state.extracted_data = result
                
                # Preprocess
                full_text = f"{result['title']} {result['content']}"
                processed_text = preprocessor.preprocess_pipeline(
                    full_text, preprocessing_steps
                )
                st.session_state.preprocessed_text = processed_text
                
                # Predict
                prediction = detector.predict(processed_text)
                st.session_state.prediction_result = prediction
                
                # Get explanation
                explanation = detector.explain_prediction(processed_text)
                st.session_state.explanation = explanation
                
                # Save to history
                history_record = {
                    'url': url_input,
                    'domain': result['domain'],
                    'title': result['title'],
                    'content': result['content'],
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'fake_probability': prediction['probabilities']['fake'],
                    'real_probability': prediction['probabilities']['real'],
                    'checked_at': datetime.now()
                }
                history_db.add_record(history_record)
                
                st.success("✅ Analysis complete!")
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
            
        with col3:
            # Copy to Clipboard
            if st.button("📋 Copy to Clipboard"):
                text_to_copy = f"""Title: {data.get('title', '')}
Source: {data.get('domain', '')}
Publish Date: {data.get('publish_date', 'N/A')}

Content:
{data.get('content', '')}

URL: {data.get('url', '')}"""
                st.session_state.copied_text = text_to_copy
                st.experimental_rerun()
        
        if 'copied_text' in st.session_state:
            st.success("✅ Text copied to clipboard!")
            del st.session_state.copied_text
        
        st.markdown("---")
        
        # Display content with max height and scroll
        st.subheader("📝 Content")
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
                    color: #ffffff;  /* White text color */
                    background-color: #f9f9f9;
                ">
                    {data['content']}
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.warning("No content available to display.")
        
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
    
    col3, col4 = st.columns([2, 1])
    
    with col3:
        st.subheader("Important Words Highlighted")
        
        # Get full text
        full_text = f"{st.session_state.extracted_data['title']} {st.session_state.extracted_data['content']}"
        
        # Highlight important words
        highlighted_html = highlight_important_words(
            full_text[:1000] + "..." if len(full_text) > 1000 else full_text,
            st.session_state.explanation['important_words']
        )
        
        st.markdown(highlighted_html, unsafe_allow_html=True)
    
    with col4:
        st.subheader("Word Importance")
        
        # Display important words table
        words_df = pd.DataFrame(st.session_state.explanation['important_words'])
        words_df['weight'] = words_df['weight'].round(3)
        st.dataframe(words_df, hide_index=True)


def batch_processing_tab(extractor, preprocessor, detector, history_db, preprocessing_steps):
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
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                with st.spinner(f"Processing {len(urls)} URLs..."):
                    # Process batch
                    results = process_batch_urls(
                        urls, extractor, preprocessor, detector, preprocessing_steps
                    )
                    
                    # Update progress
                    progress_bar.progress(1.0)
                    status_text.text(f"Processed {len(results)} URLs")
                
                # Display results
                display_batch_results(results, history_db)
        else:
            st.warning("⚠️ Please enter at least one URL!")


def display_batch_results(results, history_db):
    """Display batch processing results"""
    st.subheader("Batch Results")
    
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
    
    # Results table
    results_df = pd.DataFrame(results)
    st.dataframe(results_df, hide_index=True)
    
    # Save successful results to history
    for result in results:
        if result.get('status') == 'success':
            history_record = {
                'url': result['url'],
                'domain': result['domain'],
                'title': result['title'],
                'content': '',  # Not storing full content for batch
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'fake_probability': result['fake_probability'],
                'real_probability': result['real_probability'],
                'checked_at': datetime.now()
            }
            history_db.add_record(history_record)
    
    # Download results
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


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
    
    # Time series analysis
    st.subheader("Checks Over Time")
    
    daily_checks = history_df.groupby(history_df['checked_at'].dt.date).size()
    fig_timeline = px.line(
        x=daily_checks.index,
        y=daily_checks.values,
        labels={'x': 'Date', 'y': 'Number of Checks'},
        title='Daily News Checks'
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
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