# setup.py
"""
Quick setup script for News Scraper & Fake News Detector
"""

import os
import subprocess
import sys


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    os.makedirs(".streamlit", exist_ok=True)
    print("✅ Directories created")


def create_secrets_file():
    """Create secrets.toml from example"""
    print("\n🔐 Setting up secrets file...")
    
    example_file = ".streamlit/secrets.toml.example"
    secrets_file = ".streamlit/secrets.toml"
    
    if os.path.exists(secrets_file):
        print("⚠️  secrets.toml already exists, skipping...")
        return
    
    if os.path.exists(example_file):
        with open(example_file, 'r') as f:
            content = f.read()
        
        with open(secrets_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .streamlit/secrets.toml")
        print("⚠️  Please edit .streamlit/secrets.toml with your Supabase credentials!")
    else:
        # Create from scratch
        content = """# Supabase Configuration
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-public-key-here"
"""
        with open(secrets_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .streamlit/secrets.toml")
        print("⚠️  Please edit .streamlit/secrets.toml with your Supabase credentials!")


def install_dependencies():
    """Install required packages"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed")
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        sys.exit(1)


def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  Failed to download NLTK data: {e}")
        print("   You can download it manually later")


def print_next_steps():
    """Print next steps for user"""
    print("\n" + "="*50)
    print("🎉 Setup completed!")
    print("="*50)
    print("\n📋 Next steps:")
    print("1. Create a Supabase account at https://supabase.com")
    print("2. Create a new project and get your credentials")
    print("3. Edit .streamlit/secrets.toml with your credentials")
    print("4. Run the SQL queries from SUPABASE_SETUP.md in your Supabase SQL editor")
    print("5. Run the app with: streamlit run main.py")
    print("\n📖 See SUPABASE_SETUP.md for detailed instructions")


def main():
    """Main setup function"""
    print("🚀 Setting up News Scraper & Fake News Detector")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists("main.py"):
        print("❌ Error: main.py not found!")
        print("   Please run this script from the project root directory")
        sys.exit(1)
    
    # Run setup steps
    create_directories()
    create_secrets_file()
    install_dependencies()
    download_nltk_data()
    print_next_steps()


if __name__ == "__main__":
    main()