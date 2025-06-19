#!/usr/bin/env python3
"""
Simplified Codebase Embedding Generator

Creates embeddings for all files in a codebase using OpenAI API directly
and stores them in ChromaDB.
"""

import os
import time
import logging
import json
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv
import openai
import chromadb
from chromadb.config import Settings as ChromaSettings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleCodebaseEmbedder:
    """Simple codebase embedder using OpenAI API directly."""
    
    def __init__(
        self,
        codebase_path: str,
        storage_path: str = "./vector_storage",
        collection_name: str = "maintainx_codebase"
    ):
        self.codebase_path = Path(codebase_path).expanduser()
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        
        # Validate codebase path
        if not self.codebase_path.exists():
            raise ValueError(f"Codebase path does not exist: {self.codebase_path}")
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize OpenAI client
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.model_name = "text-embedding-3-large"
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.storage_path / "chroma_db")
        )
        
        # File extensions to process
        self.supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.sql', '.html', '.css', '.scss', '.less', '.json', '.xml', '.yaml', '.yml',
            '.md', '.txt', '.sh', '.bash', '.zsh', '.fish', '.ps1', '.bat',
            '.dockerfile', '.makefile', '.cmake', '.gradle', '.pom'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '.git', '.svn', '.hg', '__pycache__', 'node_modules', '.next',
            'dist', 'build', 'target', 'bin', 'obj', '.idea', '.vscode',
            'venv', 'env', '.env', 'virtualenv', '.tox', 'coverage',
            '.pytest_cache', '.mypy_cache', '.coverage', 'htmlcov',
            # Test and report directories
            'e2e', 'allure-results', 'test-results', 'cypress', 'playwright',
            'reports', 'screenshots', 'videos', 'artifacts', 'logs',
            'temp', 'tmp', 'cache'
        }
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed for embeddings."""
        # Check extension
        if file_path.suffix.lower() not in self.supported_extensions:
            # Also check for files without extensions that might be code
            if file_path.suffix == '' and file_path.name.lower() in {
                'makefile', 'dockerfile', 'rakefile', 'gemfile', 'procfile'
            }:
                return True
            return False
        
        # Check if in skip directory
        for part in file_path.parts:
            if part in self.skip_dirs:
                return False
        
        # Check file size (skip very large files > 1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:  # 1MB
                logger.warning(f"Skipping large file: {file_path}")
                return False
        except OSError:
            return False
        
        return True
    
    def get_files_to_process(self) -> List[Path]:
        """Get all files that should be processed for embeddings."""
        files = []
        
        logger.info(f"Scanning {self.codebase_path} for processable files...")
        
        for file_path in self.codebase_path.rglob('*'):
            if file_path.is_file() and self.should_process_file(file_path):
                files.append(file_path)
        
        logger.info(f"Found {len(files)} files to process")
        return files
    
    def read_file_content(self, file_path: Path) -> str:
        """Read file content safely."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            return content
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, chunk_size: int = 6000, overlap: int = 300) -> List[str]:
        """Split text into overlapping chunks."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at a reasonable point (newline, space, etc.)
            if end < len(text):
                # Look for a good break point
                for break_char in ['\n\n', '\n', ' ', '.', ';', '}', ')']:
                    break_point = text.rfind(break_char, start + chunk_size - overlap, end)
                    if break_point > start:
                        end = break_point + len(break_char)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap
        
        return chunks
    
    def get_existing_files(self, collection) -> set:
        """Get set of files that already have embeddings."""
        try:
            # Get all existing documents
            results = collection.get(include=['metadatas'])
            existing_files = set()
            if results and results['metadatas']:
                for metadata in results['metadatas']:
                    if 'file_path' in metadata:
                        existing_files.add(metadata['file_path'])
            logger.info(f"Found {len(existing_files)} files already processed")
            return existing_files
        except Exception as e:
            logger.warning(f"Could not get existing files: {e}")
            return set()

    def process_files(self, files: List[Path], batch_size: int = 50, skip_existing: bool = True) -> bool:
        """Process files and create embeddings."""
        logger.info(f"Processing {len(files)} files in batches of {batch_size}")
        
        # Get or create collection
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"Embeddings for {self.codebase_path.name}"}
            )
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
        
        # Skip already processed files if requested
        if skip_existing:
            existing_files = self.get_existing_files(collection)
            original_count = len(files)
            files = [f for f in files if str(f.relative_to(self.codebase_path)) not in existing_files]
            logger.info(f"Skipping {original_count - len(files)} already processed files")
            
            if not files:
                logger.info("All files already processed!")
                return True
        
        total_processed = 0
        total_chunks = 0
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(files) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
            
            # Prepare batch data
            texts = []
            metadatas = []
            ids = []
            
            for file_path in batch_files:
                try:
                    content = self.read_file_content(file_path)
                    if not content.strip():
                        continue
                    
                    # Split into chunks
                    chunks = self.chunk_text(content)
                    
                    for chunk_idx, chunk in enumerate(chunks):
                        relative_path = str(file_path.relative_to(self.codebase_path))
                        chunk_id = f"{relative_path}:chunk_{chunk_idx}"
                        
                        texts.append(chunk)
                        metadatas.append({
                            "file_path": relative_path,
                            "full_path": str(file_path),
                            "chunk_index": chunk_idx,
                            "file_extension": file_path.suffix,
                            "file_size": file_path.stat().st_size
                        })
                        ids.append(chunk_id)
                        total_chunks += 1
                    
                    total_processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
            
            if not texts:
                logger.warning(f"No valid texts in batch {batch_num}")
                continue
            
            try:
                # Generate embeddings using OpenAI API with retry logic
                logger.info(f"Generating embeddings for {len(texts)} chunks...")
                
                # Retry logic for rate limiting
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        response = self.openai_client.embeddings.create(
                            input=texts,
                            model=self.model_name
                        )
                        
                        if not response or not response.data:
                            logger.error(f"No embeddings returned for batch {batch_num}")
                            break
                        
                        embeddings = [embedding.embedding for embedding in response.data]
                        logger.info(f"Generated {len(embeddings)} embeddings")
                        break  # Success, exit retry loop
                        
                    except Exception as e:
                        if "rate limit" in str(e).lower() or "429" in str(e):
                            wait_time = (2 ** retry) + 1  # Exponential backoff: 2s, 5s, 9s
                            logger.warning(f"Rate limit hit (batch {batch_num}, attempt {retry+1}/{max_retries}). Waiting {wait_time}s...")
                            time.sleep(wait_time)
                            if retry == max_retries - 1:
                                raise e
                        else:
                            raise e
                else:
                    continue  # Skip this batch if all retries failed
                
                # Add to ChromaDB
                collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                logger.info(f"Added {len(texts)} chunks to ChromaDB")
                
                # Small delay to avoid rate limiting (conservative)
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                continue
        
        logger.info(f"✅ Processing complete: {total_processed}/{len(files)} files, {total_chunks} chunks")
        return True
    
    def generate_embeddings(self, batch_size: int = 50, skip_existing: bool = True) -> bool:
        """Main method to generate embeddings for the entire codebase."""
        start_time = time.time()
        
        logger.info(f"Starting embedding generation for {self.codebase_path}")
        
        # Get files to process
        files = self.get_files_to_process()
        
        if not files:
            logger.error("No files found to process")
            return False
        
        # Process files and create embeddings
        success = self.process_files(files, batch_size, skip_existing)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            logger.info(f"✅ Embedding generation completed in {duration:.2f} seconds")
            logger.info(f"📊 Processed {len(files)} files")
            logger.info(f"💾 Index stored at: {self.storage_path}")
        else:
            logger.error(f"❌ Embedding generation failed after {duration:.2f} seconds")
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored embeddings."""
        try:
            collection = self.chroma_client.get_collection(name=self.collection_name)
            count = collection.count()
            
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "storage_path": str(self.storage_path)
            }
        except Exception as e:
            return {"error": str(e)}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for a codebase using OpenAI")
    parser.add_argument(
        "codebase_path",
        nargs="?",
        default="~/repos/maintainx",
        help="Path to the codebase (default: ~/repos/maintainx)"
    )
    parser.add_argument(
        "--storage-path",
        default="./vector_storage",
        help="Path to store the vector index (default: ./vector_storage)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of files to process per batch (default: 50)"
    )
    parser.add_argument(
        "--collection-name",
        default="maintainx_codebase",
        help="Name for the ChromaDB collection (default: maintainx_codebase)"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show stats for existing collection"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip files that already have embeddings (default: True)"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess all files, even if they already have embeddings"
    )
    
    args = parser.parse_args()
    
    try:
        # Create embedder
        embedder = SimpleCodebaseEmbedder(
            codebase_path=args.codebase_path,
            storage_path=args.storage_path,
            collection_name=args.collection_name
        )
        
        if args.stats_only:
            # Just show stats
            stats = embedder.get_stats()
            print("\n📊 Collection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return 0
        
        # Generate embeddings
        skip_existing = args.skip_existing and not args.force_reprocess
        success = embedder.generate_embeddings(batch_size=args.batch_size, skip_existing=skip_existing)
        
        if success:
            # Show final stats
            stats = embedder.get_stats()
            print("\n📊 Final Collection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            print(f"\n🎉 Success! Embeddings created and stored at: {args.storage_path}")
            return 0
        else:
            print("❌ Embedding generation failed")
            return 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 