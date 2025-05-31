#!/bin/bash
# setup.sh - Quick setup script for News Scraper & Fake News Detector

echo "🚀 Setting up News Scraper & Fake News Detector"
echo "=================================================="

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found!"
    echo "   Please run this script from the project root directory"
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p .streamlit
echo "✅ Directories created"

# Create secrets file
echo ""
echo "🔐 Setting up secrets file..."
if [ -f ".streamlit/secrets.toml" ]; then
    echo "⚠️  secrets.toml already exists, skipping..."
else
    cat > .streamlit/secrets.toml << EOF
# Supabase Configuration
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-public-key-here"
EOF
    echo "✅ Created .streamlit/secrets.toml"
    echo "⚠️  Please edit .streamlit/secrets.toml with your Supabase credentials!"
fi

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Download NLTK data
echo ""
echo "📚 Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"
echo "✅ NLTK data downloaded"

# Print next steps
echo ""
echo "=================================================="
echo "🎉 Setup completed!"
echo "=================================================="
echo ""
echo "📋 Next steps:"
echo "1. Create a Supabase account at https://supabase.com"
echo "2. Create a new project and get your credentials"
echo "3. Edit .streamlit/secrets.toml with your credentials"
echo "4. Run the SQL queries from SUPABASE_SETUP.md in your Supabase SQL editor"
echo "5. Run the app with: streamlit run main.py"
echo ""
echo "📖 See SUPABASE_SETUP.md for detailed instructions"