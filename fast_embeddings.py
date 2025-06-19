#!/usr/bin/env python3
"""
Fast Codebase Embedding Generator

High-performance version with async processing, rate limiting, and optimized batching.
Creates embeddings using OpenAI API and stores them in ChromaDB.
"""

import os
import time
import logging
import asyncio
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Set
from concurrent.futures import ThreadPoolExecutor
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


class ProgressTracker:
    """Tracks progress and calculates ETA for embedding generation."""
    
    def __init__(self, total_items: int, item_name: str = "items"):
        self.total_items = total_items
        self.item_name = item_name
        self.completed_items = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.update_interval = 10  # Update every 10 seconds
        
    def update(self, completed: int, force_update: bool = False):
        """Update progress and optionally display ETA."""
        self.completed_items = completed
        current_time = time.time()
        
        if force_update or (current_time - self.last_update_time) >= self.update_interval:
            self._display_progress()
            self.last_update_time = current_time
    
    def _display_progress(self):
        """Display current progress and ETA."""
        if self.completed_items == 0:
            return
            
        elapsed_time = time.time() - self.start_time
        progress_percent = (self.completed_items / self.total_items) * 100
        
        # Calculate ETA
        if self.completed_items > 0:
            avg_time_per_item = elapsed_time / self.completed_items
            remaining_items = self.total_items - self.completed_items
            eta_seconds = avg_time_per_item * remaining_items
            eta_str = self._format_duration(eta_seconds)
        else:
            eta_str = "calculating..."
        
        elapsed_str = self._format_duration(elapsed_time)
        
        logger.info(f"📈 Progress: {self.completed_items}/{self.total_items} {self.item_name} "
                   f"({progress_percent:.1f}%) | Elapsed: {elapsed_str} | ETA: {eta_str}")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def complete(self):
        """Mark progress as complete and show final stats."""
        total_time = time.time() - self.start_time
        total_time_str = self._format_duration(total_time)
        avg_time = total_time / self.total_items if self.total_items > 0 else 0
        
        logger.info(f"✅ Completed {self.total_items} {self.item_name} in {total_time_str} "
                   f"(avg: {avg_time:.2f}s per {self.item_name.rstrip('s')})")


class RateLimitedEmbedder:
    """Rate-limited embedding generator with exponential backoff."""
    
    def __init__(self, openai_client, async_client, model_name: str):
        self.openai_client = openai_client
        self.async_client = async_client
        self.model_name = model_name
        self.token_count = 0
        self.last_reset_time = time.time()
        
        # Rate limiting parameters
        self.max_tokens_per_minute = 8_000_000  # Conservative limit (80% of 10M)
        self.max_retries = 5
        self.base_delay = 1.0  # Base delay in seconds
        
    def estimate_tokens(self, texts: List[str]) -> int:
        """Estimate token count for a batch of texts."""
        # Rough estimate: ~4 characters per token for embeddings
        total_chars = sum(len(text) for text in texts)
        return total_chars // 4
    
    async def generate_embeddings_with_backoff(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with exponential backoff on rate limits."""
        for attempt in range(self.max_retries):
            try:
                # Check if we need to wait for rate limit reset
                await self._check_rate_limit(texts)
                
                response = await self.async_client.embeddings.create(
                    input=texts,
                    model=self.model_name
                )
                
                if not response or not response.data:
                    raise RuntimeError("No embeddings returned from OpenAI")
                
                # Update token tracking
                estimated_tokens = self.estimate_tokens(texts)
                self.token_count += estimated_tokens
                
                return [embedding.embedding for embedding in response.data]
                
            except openai.RateLimitError as e:
                wait_time = self._parse_rate_limit_error(e, attempt)
                logger.warning(f"Rate limit hit (attempt {attempt + 1}/{self.max_retries}). Waiting {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
                continue
                
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed after {self.max_retries} attempts: {e}")
                    raise
                
                # Exponential backoff for other errors
                wait_time = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"API error (attempt {attempt + 1}/{self.max_retries}): {e}. Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)
        
        raise RuntimeError(f"Failed to generate embeddings after {self.max_retries} attempts")
    
    def _parse_rate_limit_error(self, error, attempt: int) -> float:
        """Parse rate limit error and return appropriate wait time."""
        try:
            error_str = str(error)
            # Try to extract wait time from error message
            if "Please try again in" in error_str:
                # Extract milliseconds from error message
                import re
                match = re.search(r'Please try again in (\d+)ms', error_str)
                if match:
                    wait_ms = int(match.group(1))
                    return (wait_ms / 1000.0) + 0.5  # Add buffer
        except:
            pass
        
        # Fallback to exponential backoff
        wait_time = self.base_delay * (2 ** attempt) + random.uniform(1, 3)
        return min(wait_time, 60)  # Cap at 60 seconds
    
    async def _check_rate_limit(self, texts: List[str]):
        """Check if we're approaching rate limits and wait if necessary."""
        current_time = time.time()
        time_since_reset = current_time - self.last_reset_time
        
        # Reset token count every minute
        if time_since_reset >= 60:
            self.token_count = 0
            self.last_reset_time = current_time
            time_since_reset = 0
        
        # Estimate tokens for this batch
        estimated_batch_tokens = self.estimate_tokens(texts)
        
        # If we're close to the limit, wait until the next minute
        if self.token_count + estimated_batch_tokens > self.max_tokens_per_minute:
            wait_time = 60 - time_since_reset + 1  # Wait until next minute + buffer
            logger.info(f"Approaching rate limit. Waiting {wait_time:.1f}s for reset...")
            await asyncio.sleep(wait_time)
            self.token_count = 0
            self.last_reset_time = time.time()


class FastCodebaseEmbedder:
    """High-performance codebase embedder with rate limiting."""
    
    def __init__(
        self,
        codebase_path: str,
        storage_path: str = "./vector_storage",
        collection_name: str = "maintainx_codebase",
        max_concurrent: int = 3  # Reduced default for rate limiting
    ):
        self.codebase_path = Path(codebase_path).expanduser()
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        self.max_concurrent = max_concurrent
        
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
        self.async_client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.model_name = "text-embedding-3-large"
        
        # Initialize rate-limited embedder
        self.embedder = RateLimitedEmbedder(
            self.openai_client, 
            self.async_client, 
            self.model_name
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.storage_path / "chroma_db")
        )
        
        # File extensions to process (optimized list)
        self.supported_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h',
            '.cs', '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala',
            '.sql', '.html', '.css', '.scss', '.json', '.xml', '.yaml', '.yml',
            '.md', '.txt', '.sh', '.bash', '.dockerfile', '.makefile'
        }
        
        # Directories to skip
        self.skip_dirs = {
            '.git', '.svn', '__pycache__', 'node_modules', '.next',
            'dist', 'build', 'target', 'bin', 'obj', '.idea', '.vscode',
            'venv', 'env', 'virtualenv', '.pytest_cache', '.mypy_cache',
            # Test and report directories
            'e2e', 'allure-results', 'test-results', 'cypress', 'playwright',
            'reports', 'screenshots', 'videos', 'artifacts', 'logs',
            'temp', 'tmp', 'cache'
        }
    
    def should_process_file(self, file_path: Path) -> bool:
        """Check if a file should be processed for embeddings."""
        # FIRST: Only process files under src/ directories
        relative_path = file_path.relative_to(self.codebase_path)
        path_parts = relative_path.parts
        
        # Check if file is under any src/ directory at any level
        has_src_in_path = any(part.lower() == 'src' for part in path_parts)
        if not has_src_in_path:
            return False
        
        # Check extension
        if file_path.suffix.lower() not in self.supported_extensions:
            if file_path.suffix == '' and file_path.name.lower() in {
                'makefile', 'dockerfile', 'rakefile', 'gemfile'
            }:
                return True
            return False
        
        # Check if in skip directory
        for part in path_parts:
            if part.lower() in self.skip_dirs:
                return False
        
        # Additional filtering for test files even in src/
        file_name_lower = file_path.name.lower()
        if any(test_pattern in file_name_lower for test_pattern in [
            'test_', '_test.', '.test.', '.spec.', '_spec.', 'test.', 'spec.',
            'mock_', '_mock.', '.mock.', 'fixture'
        ]):
            return False
        
        # Skip certain file types even in src/
        if any(file_name_lower.endswith(ext) for ext in [
            '.log', '.lock', '.cache', '.tmp', '.temp', '.bak', 
            '.swp', '.swo', '.orig', '.rej'
        ]):
            return False
        
        # Check file size (skip very large files > 500KB)
        try:
            if file_path.stat().st_size > 512 * 1024:  # 512KB
                logger.warning(f"Skipping large file: {file_path}")
                return False
        except OSError:
            return False
        
        return True
    
    def get_files_to_process(self) -> List[Path]:
        """Get all files that should be processed for embeddings."""
        files = []
        total_files = 0
        src_files = 0
        
        logger.info(f"Scanning {self.codebase_path} for processable files...")
        logger.info("📂 Only indexing files under src/ directories")
        
        for file_path in self.codebase_path.rglob('*'):
            if file_path.is_file():
                total_files += 1
                
                # Check if under src/
                relative_path = file_path.relative_to(self.codebase_path)
                if any(part.lower() == 'src' for part in relative_path.parts):
                    src_files += 1
                    
                    if self.should_process_file(file_path):
                        files.append(file_path)
        
        logger.info(f"📊 Scan results: {total_files:,} total files, {src_files:,} in src/, {len(files):,} selected for processing")
        
        if len(files) == 0:
            logger.warning("⚠️  No files found to process! Check if your codebase has src/ directories")
        
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
        """Split text into overlapping chunks (optimized for rate limits)."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at a reasonable point
            if end < len(text):
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
    
    def get_existing_files(self, collection) -> Set[str]:
        """Get set of files that already have embeddings."""
        try:
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
    
    async def process_files_async(
        self, 
        files: List[Path], 
        batch_size: int = 50,  # Reduced default batch size
        skip_existing: bool = True
    ) -> bool:
        """Process files asynchronously with rate limiting."""
        logger.info(f"Processing {len(files)} files with rate-limited batching (batch_size: {batch_size})")
        
        # Get or create collection
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": f"Embeddings for {self.codebase_path.name}"}
            )
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
        
        # Skip already processed files
        if skip_existing:
            existing_files = self.get_existing_files(collection)
            original_count = len(files)
            files = [f for f in files if str(f.relative_to(self.codebase_path)) not in existing_files]
            logger.info(f"Skipping {original_count - len(files)} already processed files")
            
            if not files:
                logger.info("All files already processed!")
                return True
        
        # Process files in parallel batches
        total_processed = 0
        total_chunks = 0
        
        # Use ThreadPoolExecutor for file I/O
        logger.info(f"📖 Reading {len(files)} files in parallel...")
        file_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Read all files in parallel
            file_contents = await asyncio.gather(*[
                asyncio.get_event_loop().run_in_executor(
                    executor, self.read_file_content, file_path
                ) for file_path in files
            ])
        
        file_read_time = time.time() - file_start_time
        logger.info(f"✅ Files read in {file_read_time:.1f}s")
        
        # Prepare all chunks
        all_texts = []
        all_metadatas = []
        all_ids = []
        
        logger.info("🔪 Chunking texts...")
        chunk_start_time = time.time()
        
        for file_path, content in zip(files, file_contents):
            if not content.strip():
                continue
            
            chunks = self.chunk_text(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                relative_path = str(file_path.relative_to(self.codebase_path))
                chunk_id = f"{relative_path}:chunk_{chunk_idx}"
                
                all_texts.append(chunk)
                all_metadatas.append({
                    "file_path": relative_path,
                    "full_path": str(file_path),
                    "chunk_index": chunk_idx,
                    "file_extension": file_path.suffix,
                    "file_size": file_path.stat().st_size
                })
                all_ids.append(chunk_id)
                total_chunks += 1
            
            total_processed += 1
        
        if not all_texts:
            logger.warning("No valid texts to process")
            return True
        
        chunk_time = time.time() - chunk_start_time
        avg_chunks_per_file = total_chunks / total_processed if total_processed > 0 else 0
        logger.info(f"✅ Generated {total_chunks} chunks from {total_processed} files in {chunk_time:.1f}s "
                   f"(avg: {avg_chunks_per_file:.1f} chunks/file)")
        
        # Initialize progress tracking
        total_batches = (len(all_texts) + batch_size - 1) // batch_size
        progress_tracker = ProgressTracker(total_batches, "batches")
        
        # Process embeddings in batches with rate limiting
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_batch(batch_start: int, batch_end: int, batch_num: int):
            async with semaphore:
                batch_texts = all_texts[batch_start:batch_end]
                batch_metadatas = all_metadatas[batch_start:batch_end]
                batch_ids = all_ids[batch_start:batch_end]
                
                try:
                    logger.info(f"Processing batch {batch_num} ({len(batch_texts)} chunks)...")
                    
                    # Use rate-limited embedder
                    embeddings = await self.embedder.generate_embeddings_with_backoff(batch_texts)
                    
                    # Add to ChromaDB (this is sync, but fast)
                    collection.add(
                        embeddings=embeddings,
                        documents=batch_texts,
                        metadatas=batch_metadatas,
                        ids=batch_ids
                    )
                    
                    logger.info(f"✅ Batch {batch_num} completed ({len(batch_texts)} chunks)")
                    return len(batch_texts)
                    
                except Exception as e:
                    logger.error(f"❌ Batch {batch_num} failed: {e}")
                    return 0
        
        # Create batches and process them with controlled concurrency
        batch_tasks = []
        
        for i in range(0, len(all_texts), batch_size):
            batch_end = min(i + batch_size, len(all_texts))
            batch_num = (i // batch_size) + 1
            batch_tasks.append(process_batch(i, batch_end, batch_num))
        
        logger.info(f"🚀 Starting processing of {len(batch_tasks)} batches with max {self.max_concurrent} concurrent...")
        logger.info(f"📊 Total chunks to process: {len(all_texts)}")
        
        # Execute all batches with progress tracking
        completed = 0
        successful_chunks = 0
        failed_batches = 0
        failed_chunk_count = 0
        
        # Initial progress update
        progress_tracker.update(0, force_update=True)
        
        for batch_result in asyncio.as_completed(batch_tasks):
            try:
                result = await batch_result
                completed += 1
                if isinstance(result, int) and result > 0:
                    successful_chunks += result
                else:
                    failed_batches += 1
                    # Estimate failed chunks (assume average batch size)
                    avg_batch_size = len(all_texts) // len(batch_tasks)
                    failed_chunk_count += avg_batch_size
                
                # Update progress tracker
                progress_tracker.update(completed)
                
                # Log milestone progress
                if completed in [1, 5, 10] or completed % 25 == 0 or completed == len(batch_tasks):
                    success_rate = (successful_chunks / (successful_chunks + failed_chunk_count)) * 100 if (successful_chunks + failed_chunk_count) > 0 else 100
                    logger.info(f"🎯 Milestone: {completed}/{len(batch_tasks)} batches | "
                               f"Success rate: {success_rate:.1f}% | "
                               f"Chunks: {successful_chunks}/{len(all_texts)}")
                    
            except Exception as e:
                completed += 1
                failed_batches += 1
                failed_chunk_count += len(all_texts) // len(batch_tasks)  # Estimate
                progress_tracker.update(completed)
                logger.error(f"Batch failed: {e}")
        
        # Final progress update
        progress_tracker.complete()
        
        # Comprehensive final stats
        success_rate = (successful_chunks / total_chunks) * 100 if total_chunks > 0 else 0
        total_processing_time = time.time() - chunk_start_time
        chunks_per_second = successful_chunks / total_processing_time if total_processing_time > 0 else 0
        
        logger.info("=" * 60)
        logger.info("📊 FINAL PROCESSING SUMMARY")
        logger.info(f"✅ Successful chunks: {successful_chunks:,}/{total_chunks:,} ({success_rate:.1f}%)")
        logger.info(f"⚠️  Failed batches: {failed_batches}/{len(batch_tasks)}")
        logger.info(f"⏱️  Processing rate: {chunks_per_second:.1f} chunks/second")
        logger.info(f"🎯 Overall success rate: {success_rate:.1f}%")
        logger.info("=" * 60)
        
        if failed_batches > 0:
            logger.warning(f"⚠️  {failed_batches} batches failed - check logs for rate limiting issues")
        
        return successful_chunks > 0
    
    def generate_embeddings(
        self, 
        batch_size: int = 50,  # Reduced default
        skip_existing: bool = True,
        max_concurrent: int = None
    ) -> bool:
        """Main method to generate embeddings for the entire codebase."""
        if max_concurrent:
            self.max_concurrent = max_concurrent
            
        start_time = time.time()
        
        logger.info(f"Starting RATE-LIMITED embedding generation for {self.codebase_path}")
        logger.info(f"Config: batch_size={batch_size}, max_concurrent={self.max_concurrent}")
        logger.info("🎯 Focus: Only indexing files under src/ directories (excluding tests/reports/build artifacts)")
        
        # Get files to process
        files = self.get_files_to_process()
        
        if not files:
            logger.error("No files found to process")
            return False
        
        # Process files asynchronously with rate limiting
        success = asyncio.run(self.process_files_async(files, batch_size, skip_existing))
        
        end_time = time.time()
        duration = end_time - start_time
        
        if success:
            logger.info(f"✅ RATE-LIMITED embedding generation completed in {duration:.2f} seconds")
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
    
    parser = argparse.ArgumentParser(description="Rate-limited fast embedding generation using OpenAI")
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
        default=50,  # Reduced default
        help="Number of chunks per embedding batch (default: 50)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,  # Reduced default
        help="Maximum concurrent API calls (default: 3)"
    )
    parser.add_argument(
        "--collection-name",
        default="maintainx_codebase",
        help="Name for the ChromaDB collection (default: maintainx_codebase)"
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Reprocess all files, even if they already have embeddings"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only show stats for existing collection"
    )
    
    args = parser.parse_args()
    
    try:
        # Create embedder
        embedder = FastCodebaseEmbedder(
            codebase_path=args.codebase_path,
            storage_path=args.storage_path,
            collection_name=args.collection_name,
            max_concurrent=args.max_concurrent
        )
        
        if args.stats_only:
            stats = embedder.get_stats()
            print("\n📊 Collection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            return 0
        
        # Generate embeddings
        skip_existing = not args.force_reprocess
        success = embedder.generate_embeddings(
            batch_size=args.batch_size,
            skip_existing=skip_existing,
            max_concurrent=args.max_concurrent
        )
        
        if success:
            stats = embedder.get_stats()
            print("\n📊 Final Collection Statistics:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            print(f"\n🎉 Success! Rate-limited embeddings created at: {args.storage_path}")
            return 0
        else:
            print("❌ Rate-limited embedding generation failed")
            return 1
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main()) 