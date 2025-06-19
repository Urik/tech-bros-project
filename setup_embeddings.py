#!/usr/bin/env python3
"""
Setup script for codebase embedding generation
Installs dependencies and sets up the environment
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {command}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install required Python packages."""
    print("📦 Installing Python dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True


def check_openai_api_key():
    """Check if OpenAI API key is configured."""
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found in environment variables")
        print("   Please set your OpenAI API key:")
        print("   export OPENAI_API_KEY='your-api-key-here'")
        print("   Or add it to your .env file:")
        print("   echo 'OPENAI_API_KEY=your-api-key-here' >> .env")
        return False
    
    print("✅ OPENAI_API_KEY found in environment")
    return True


def check_maintainx_repo():
    """Check if MaintainX repo exists."""
    maintainx_path = Path("~/repos/maintainx").expanduser()
    if not maintainx_path.exists():
        print(f"⚠️  MaintainX repository not found at {maintainx_path}")
        print("   Please ensure the repository exists or update the path in simple_embeddings.py")
        return False
    
    print(f"✅ MaintainX repository found at {maintainx_path}")
    return True


def create_env_template():
    """Create a .env template file if it doesn't exist."""
    env_file = Path(".env")
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    template_content = """# OpenAI API Key (required for embeddings)
OPENAI_API_KEY=your-openai-api-key-here

# Anthropic API Key (for Claude integration)
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Notion Integration (optional)
NOTION_TOKEN=your-notion-token-here
NOTION_PAGE_ID=your-notion-page-id-here

# Codebase path (optional, defaults to ~/repos/maintainx)
CODEBASE_PATH=~/repos/maintainx
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(template_content)
        print("✅ Created .env template file")
        print("   Please edit .env and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env template: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up codebase embedding generation environment...")
    print("=" * 60)
    
    success = True
    
    # Check Python version
    if not check_python_version():
        success = False
    
    # Create .env template
    if not create_env_template():
        success = False
    
    # Install dependencies
    if not install_dependencies():
        success = False
    
    # Check API key (after dependencies are installed for dotenv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        if not check_openai_api_key():
            success = False
    except ImportError:
        print("⚠️  Could not load dotenv to check API key")
    
    # Check MaintainX repo
    if not check_maintainx_repo():
        success = False
    
    print("=" * 60)
    
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Make sure your OPENAI_API_KEY is set in .env")
        print("2. Run: python simple_embeddings.py")
        print("3. Once embeddings are created, run: python query_embeddings.py")
    else:
        print("❌ Setup completed with errors")
        print("Please fix the issues above before proceeding")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 