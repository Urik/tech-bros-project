#!/usr/bin/env python3
"""
Re-index Script for Source Files Only

Cleans existing embeddings and rebuilds with improved filtering (src/ directories only).
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def main():
    """Clean and re-index with src-only filtering."""
    print("🧹 Cleaning existing embeddings and re-indexing with src-only filtering...")
    print("=" * 60)
    
    # Check if vector storage exists
    storage_path = Path("./vector_storage")
    if storage_path.exists():
        print(f"📁 Removing existing index at {storage_path}")
        shutil.rmtree(storage_path)
        print("✅ Existing index removed")
    else:
        print("📁 No existing index found")
    
    # Get codebase path
    codebase_path = input("\n📂 Enter codebase path (default: ~/repos/maintainx): ").strip()
    if not codebase_path:
        codebase_path = "~/repos/maintainx"
    
    # Confirm src/ directories exist
    expanded_path = Path(codebase_path).expanduser()
    if not expanded_path.exists():
        print(f"❌ Codebase path does not exist: {expanded_path}")
        return 1
    
    # Find src directories
    src_dirs = list(expanded_path.rglob("src"))
    if not src_dirs:
        print(f"⚠️  No 'src' directories found in {expanded_path}")
        print("   This script only indexes files under src/ directories")
        proceed = input("   Continue anyway? (y/N): ").strip().lower()
        if proceed != 'y':
            print("   Aborted")
            return 0
    else:
        print(f"📂 Found {len(src_dirs)} src/ directories:")
        for src_dir in src_dirs[:5]:  # Show first 5
            print(f"   • {src_dir.relative_to(expanded_path)}")
        if len(src_dirs) > 5:
            print(f"   ... and {len(src_dirs) - 5} more")
    
    # Choose indexing method
    print("\n🚀 Choose indexing method:")
    print("1. Fast (recommended) - Async processing with rate limiting")
    print("2. Simple - Basic processing")
    
    choice = input("Enter choice (1-2, default: 1): ").strip()
    if choice == "2":
        script = "simple_embeddings.py"
    else:
        script = "fast_embeddings.py"
    
    # Run the indexing
    print(f"\n🔄 Starting indexing with {script}...")
    print("-" * 40)
    
    try:
        result = subprocess.run([
            sys.executable, script,
            codebase_path,
            "--force-reprocess"
        ], check=True)
        
        print("-" * 40)
        print("🎉 Re-indexing completed successfully!")
        print("\n📝 What changed:")
        print("✅ Only files under src/ directories are indexed")
        print("✅ Test files, reports, and build artifacts excluded")
        print("✅ Better filtering for relevant source code")
        print("\n🔍 Try querying now - you should get much better results!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Indexing failed with exit code {e.returncode}")
        return 1
    except KeyboardInterrupt:
        print("\n⏹️  Indexing interrupted by user")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 