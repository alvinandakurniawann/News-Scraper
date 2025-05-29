# News Scraper

A Streamlit-based web application for extracting and analyzing news from various Indonesian online news sources.

## Features

- News extraction from various popular Indonesian news websites
- Automatic text cleaning and preprocessing
- Basic text analysis (tokenization, stopword removal, stemming)
- Export extraction results in JSON format
- User-friendly interface

## Supported News Websites

- Detik.com
- Kompas.com
- Tribunnews.com

## System Requirements

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone this repository:
   ```bash
   git clone [REPOSITORY_URL]
   cd text-scraper-complete
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If requirements.txt is not available, install packages manually:
   ```bash
   pip install streamlit beautifulsoup4 requests nltk sastrawi pandas
   ```

4. Download required NLTK data:
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run news_scraper.py
   ```

2. Open your browser and navigate to the displayed URL (usually http://localhost:8501)

3. Enter the news URL you want to extract

4. Select desired preprocessing options:
   - Text cleaning
   - Punctuation removal
   - Tokenization
   - Stopword removal
   - Stemming

5. Click the "Extract News" button to process the URL

6. Use the "Download JSON" button to save the extraction results

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue to report bugs or request new features.

## License

[MIT License](LICENSE)
   