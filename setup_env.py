#!/usr/bin/env python3
"""
Setup script for Voice Browser environment
"""
import os
import sys

def create_env_file():
    """Create .env file with necessary environment variables"""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Browser Use Configuration
BROWSER_USE_LOGGING_LEVEL=debug

# Optional: Other LLM providers
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
# GOOGLE_API_KEY=your_google_api_key_here

# Server Configuration
HOST=localhost
PORT=8000
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(env_content)
        print("✅ Created .env file")
        print("⚠️  Please edit .env and add your OPENAI_API_KEY")
        return True
    except Exception as e:
        print(f"❌ Could not create .env file: {e}")
        return False

def check_environment():
    """Check if environment is properly set up"""
    print("🔍 Checking environment setup...")
    
    # Check Python version
    if sys.version_info < (3, 11):
        print(f"❌ Python 3.11+ required, found {sys.version}")
        return False
    else:
        print(f"✅ Python {sys.version.split()[0]}")
    
    # Check required packages
    package_imports = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn', 
        'websockets': 'websockets',
        'langchain_openai': 'langchain_openai',
        'browser_use': 'browser_use',
        'python_dotenv': 'dotenv',
        'speech_recognition': 'speech_recognition',
        'pyaudio': 'pyaudio',
        'playwright': 'playwright',
        'openai': 'openai'
    }
    
    missing_packages = []
    for package_name, import_name in package_imports.items():
        try:
            __import__(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name}")
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r backend/requirements.txt")
        return False
    
    # Check .env file
    if not os.path.exists('.env'):
        print("❌ .env file not found")
        return False
    else:
        print("✅ .env file exists")
    
    # Check OpenAI API key
    from dotenv import load_dotenv
    load_dotenv()
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key or openai_key == 'your_openai_api_key_here':
        print("❌ OPENAI_API_KEY not set in .env file")
        return False
    else:
        print("✅ OPENAI_API_KEY configured")
    
    return True

def test_browser_use():
    """Test browser-use installation"""
    print("\n🧪 Testing browser-use...")
    try:
        from browser_use import Browser, BrowserConfig
        browser = Browser(config=BrowserConfig(headless=True))
        print("✅ browser-use working")
        return True
    except Exception as e:
        print(f"❌ browser-use test failed: {e}")
        return False

def main():
    print("🚀 Voice Browser Environment Setup")
    print("=" * 40)
    
    # Create .env if it doesn't exist
    if not os.path.exists('.env'):
        create_env_file()
    
    # Check environment
    if check_environment():
        print("\n✅ Environment setup complete!")
        print("\nNext steps:")
        print("1. Make sure your OPENAI_API_KEY is set in .env")
        print("2. Run the voice browser: python workingScript.py")
        print("3. Or run the web server: cd backend && python app.py")
    else:
        print("\n❌ Environment setup incomplete")
        print("Please fix the issues above and run this script again")

if __name__ == '__main__':
    main() 