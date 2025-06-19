#!/usr/bin/env python3
"""
Codebase Embedding Generator

Creates embeddings for all files in a codebase using VoyageCode and stores them
in a persistent vector index using LlamaIndex and ChromaDB.
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Optional, Set
from dotenv import load_dotenv

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.node_parser import CodeSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import voyageai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodebaseEmbeddingGenerator:
    """Generates and stores embeddings for a codebase using VoyageCode."""
    
    def __init__(
        self,
        codebase_path: str,
        storage_path: str = "./vector_storage",
        collection_name: str = "codebase_embeddings"
    ):
        self.codebase_path = Path(codebase_path).expanduser()
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        
        # Validate codebase path
        if not self.codebase_path.exists():
            raise ValueError(f"Codebase path does not exist: {self.codebase_path}")
        
        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize VoyageCode embedding model
        voyage_api_key = os.getenv('VOYAGE_API_KEY')
        if not voyage_api_key:
            raise ValueError("VOYAGE_API_KEY environment variable is required")
        
        # Use Voyage API directly
        self.voyage_client = voyageai.Client(api_key=voyage_api_key)
        self.model_name = "voyage-code-2"
        
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
            '.pytest_cache', '.mypy_cache', '.coverage', 'htmlcov'
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
    
    def create_vector_store(self) -> ChromaVectorStore:
        """Create or load ChromaDB vector store."""
        # Initialize ChromaDB client
        chroma_client = chromadb.PersistentClient(path=str(self.storage_path / "chroma_db"))
        
        # Get or create collection
        chroma_collection = chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": f"Embeddings for {self.codebase_path.name}"}
        )
        
        # Create vector store
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        
        return vector_store
    
    def create_storage_context(self, vector_store: ChromaVectorStore) -> StorageContext:
        """Create storage context for persistent storage."""
        # Create document and index stores
        docstore = SimpleDocumentStore()
        index_store = SimpleIndexStore()
        
        # Create storage context
        storage_context = StorageContext.from_defaults(
            vector_store=vector_store,
            docstore=docstore,
            index_store=index_store
        )
        
        return storage_context
    
    def process_files_batch(self, files: List[Path], batch_size: int = 50) -> VectorStoreIndex:
        """Process files in batches to create embeddings."""
        logger.info(f"Processing {len(files)} files in batches of {batch_size}")
        
        # Create vector store and storage context
        vector_store = self.create_vector_store()
        storage_context = self.create_storage_context(vector_store)
        
        # Initialize index
        index = None
        total_processed = 0
        
        for i in range(0, len(files), batch_size):
            batch_files = files[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            total_batches = (len(files) + batch_size - 1) // batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch_files)} files)")
            
            try:
                # Create temporary directory with batch files
                batch_paths = [str(f) for f in batch_files]
                
                # Use SimpleDirectoryReader with input_files parameter
                reader = SimpleDirectoryReader(
                    input_files=batch_paths,
                    recursive=False
                )
                
                # Load documents
                documents = reader.load_data()
                
                if not documents:
                    logger.warning(f"No documents loaded from batch {batch_num}")
                    continue
                
                # Parse documents into nodes with code-aware splitting
                parser = CodeSplitter(
                    language="python",  # Default, will be auto-detected per file
                    chunk_lines=100,    # Lines per chunk
                    chunk_lines_overlap=20,  # Overlap between chunks
                    max_chars=4000      # Max characters per chunk
                )
                
                nodes = parser.get_nodes_from_documents(documents)
                
                if not nodes:
                    logger.warning(f"No nodes created from batch {batch_num}")
                    continue
                
                logger.info(f"Created {len(nodes)} nodes from {len(documents)} documents")
                
                # Create or update index
                if index is None:
                    # Create new index
                    index = VectorStoreIndex(
                        nodes=nodes,
                        storage_context=storage_context
                    )
                else:
                    # Add nodes to existing index
                    index.insert_nodes(nodes)
                
                total_processed += len(batch_files)
                logger.info(f"Processed {total_processed}/{len(files)} files")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                continue
        
        if index is None:
            raise RuntimeError("No index was created - all batches failed")
        
        # Persist the index
        index.storage_context.persist(persist_dir=str(self.storage_path))
        logger.info(f"Index persisted to {self.storage_path}")
        
        return index
    
    def generate_embeddings(self, batch_size: int = 50) -> VectorStoreIndex:
        """Main method to generate embeddings for the entire codebase."""
        start_time = time.time()
        
        logger.info(f"Starting embedding generation for {self.codebase_path}")
        
        # Get files to process
        files = self.get_files_to_process()
        
        if not files:
            raise ValueError("No files found to process")
        
        # Process files and create embeddings
        index = self.process_files_batch(files, batch_size)
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"✅ Embedding generation completed in {duration:.2f} seconds")
        logger.info(f"📊 Processed {len(files)} files")
        logger.info(f"💾 Index stored at: {self.storage_path}")
        
        return index
    
    def load_existing_index(self) -> Optional[VectorStoreIndex]:
        """Load existing index from storage."""
        try:
            vector_store = self.create_vector_store()
            storage_context = self.create_storage_context(vector_store)
            
            # Try to load existing index
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            
            logger.info(f"✅ Loaded existing index from {self.storage_path}")
            return index
            
        except Exception as e:
            logger.info(f"No existing index found: {e}")
            return None
    
    def get_index_stats(self, index: VectorStoreIndex) -> dict:
        """Get statistics about the index."""
        try:
            # Get collection info from ChromaDB
            vector_store = index.vector_store
            collection = vector_store._collection
            
            stats = {
                "collection_name": collection.name,
                "document_count": collection.count(),
                "storage_path": str(self.storage_path)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting index stats: {e}")
            return {"error": str(e)}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for a codebase")
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
        "--load-existing",
        action="store_true",
        help="Try to load existing index instead of creating new one"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show stats for existing index"
    )
    
    args = parser.parse_args()
    
    try:
        # Create generator
        generator = CodebaseEmbeddingGenerator(
            codebase_path=args.codebase_path,
            storage_path=args.storage_path,
            collection_name=args.collection_name
        )
        
        if args.stats_only:
            # Just show stats
            index = generator.load_existing_index()
            if index:
                stats = generator.get_index_stats(index)
                print("\n📊 Index Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            else:
                print("❌ No existing index found")
            return
        
        if args.load_existing:
            # Try to load existing index
            index = generator.load_existing_index()
            if index:
                stats = generator.get_index_stats(index)
                print("\n📊 Loaded Index Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                return
            else:
                print("⚠️  No existing index found, creating new one...")
        
        # Generate embeddings
        index = generator.generate_embeddings(batch_size=args.batch_size)
        
        # Show final stats
        stats = generator.get_index_stats(index)
        print("\n📊 Final Index Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\n🎉 Success! Embeddings created and stored at: {args.storage_path}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 