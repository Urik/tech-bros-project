#!/usr/bin/env python3
"""
Startup script for Notion + Claude + Codebase Integration

This script checks the environment and dependencies before starting the main application.
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'anthropic', 
        'notion_client',
        'claude_code_sdk',
        'python_dotenv',
        'nest_asyncio',
        'gitpython',
        'anyio'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - Not installed")
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    
    return True

def check_system_dependencies():
    """Check for Node.js and Claude Code CLI."""
    import subprocess
    import shutil
    
    print("\n📋 System Dependencies:")
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Node.js: {result.stdout.strip()}")
            node_ok = True
        else:
            print("❌ Node.js - Not working properly")
            node_ok = False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Node.js - Not installed")
        node_ok = False
    
    # Check Claude Code CLI
    try:
        result = subprocess.run(['claude', '--version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ Claude Code CLI: {result.stdout.strip()}")
            claude_ok = True
        else:
            print("❌ Claude Code CLI - Not working properly")
            claude_ok = False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ Claude Code CLI - Not installed")
        claude_ok = False
    
    if not node_ok or not claude_ok:
        print("\n⚠️  Missing system dependencies:")
        if not node_ok:
            print("   • Install Node.js: https://nodejs.org/")
        if not claude_ok:
            print("   • Install Claude Code CLI: npm install -g @anthropic-ai/claude-code")
        return False
    
    return True

def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = {
        'NOTION_TOKEN': 'Notion API integration token',
        'ANTHROPIC_API_KEY': 'Claude API key from Anthropic'
    }
    
    optional_vars = {
        'NOTION_PAGE_ID': 'Default Notion page ID',
        'CODEBASE_PATH': 'Path to your codebase'
    }
    
    missing_required = []
    
    # Check required variables
    for var, description in required_vars.items():
        if os.getenv(var):
            print(f"✅ {var}")
        else:
            missing_required.append(f"{var} ({description})")
            print(f"❌ {var} - {description}")
    
    # Check optional variables
    for var, description in optional_vars.items():
        if os.getenv(var):
            print(f"✅ {var} (optional)")
        else:
            print(f"⚪ {var} - {description} (optional)")
    
    if missing_required:
        print(f"\n⚠️  Missing required environment variables:")
        for var in missing_required:
            print(f"   • {var}")
        print("\nCreate a .env file with these variables or set them in your environment.")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists and provide guidance."""
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if env_file.exists():
        print("✅ .env file found")
        return True
    elif env_example.exists():
        print("⚠️  .env file not found, but .env.example exists")
        print("   Copy .env.example to .env and fill in your API keys:")
        print("   cp .env.example .env")
        return False
    else:
        print("⚠️  No .env file found. Creating a template...")
        create_env_template()
        return False

def create_env_template():
    """Create a basic .env template file."""
    template_content = """# Notion API Configuration
NOTION_TOKEN=your_notion_integration_token_here
NOTION_PAGE_ID=your_notion_page_id_here

# Claude API Configuration  
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Codebase Configuration
CODEBASE_PATH=./your_codebase_path

# Application Settings
LOG_LEVEL=INFO
"""
    
    try:
        with open('.env', 'w') as f:
            f.write(template_content)
        print("✅ Created .env template file")
        print("   Please edit .env and add your API keys")
    except Exception as e:
        print(f"❌ Could not create .env file: {e}")

def main():
    """Main startup checks."""
    print("🚀 Notion + Claude + Codebase Integration")
    print("=" * 50)
    print("Checking environment setup...\n")
    
    checks = [
        ("Python Version", check_python_version),
        ("System Dependencies", check_system_dependencies),
        ("Python Dependencies", check_dependencies),
        ("Environment File", check_env_file),
        ("Environment Variables", check_environment_variables)
    ]
    
    all_passed = True
    
    for check_name, check_func in checks:
        print(f"\n📋 {check_name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "=" * 50)
    
    if all_passed:
        print("🎉 All checks passed! Starting the application...")
        print("-" * 50)
        
        # Import and run the main application
        try:
            from main import main as run_main
            import asyncio
            asyncio.run(run_main())
        except KeyboardInterrupt:
            print("\n👋 Application stopped by user")
        except Exception as e:
            print(f"❌ Error running application: {e}")
            sys.exit(1)
    else:
        print("❌ Some checks failed. Please fix the issues above and try again.")
        print("\n📚 Quick Setup Guide:")
        print("1. Install Node.js: https://nodejs.org/")
        print("2. Install Claude Code CLI: npm install -g @anthropic-ai/claude-code")
        print("3. Install Python dependencies: pip install -r requirements.txt")
        print("4. Get a Notion API token: https://www.notion.so/my-integrations")
        print("5. Get a Claude API key: https://console.anthropic.com")
        print("6. Edit the .env file with your API keys")
        print("7. Run: python start.py")
        sys.exit(1)

if __name__ == "__main__":
    main() 