#!/usr/bin/env python3
"""
Simple Claude Code Bot with Vector Search

A simplified tool to query codebases using vector search and Claude AI with optional Notion context.
"""

import asyncio
import logging
import os
import time
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv
from llama_index.readers.notion import NotionPageReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Document
from llama_index.embeddings.openai import OpenAIEmbedding
import chromadb
import openai
import anthropic

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class SimpleClaudeBot:
    """Simplified bot for Claude queries with vector search and optional Notion context."""
    
    def __init__(self):
        print("🔧 Initializing SimpleClaudeBot...")
        
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.notion_token = os.getenv('NOTION_TOKEN')
        self.prd_page_id = os.getenv('PRD_PAGE_ID')
        self.tech_spec_page_id = os.getenv('TECH_SPEC_PAGE_ID')
        self.storage_path = Path("./vector_storage")
        
        print(f"📁 Storage path: {self.storage_path}")
        print(f"🔑 Anthropic API key: {'✅ Found' if self.anthropic_api_key else '❌ Missing'}")
        print(f"🔑 OpenAI API key: {'✅ Found' if self.openai_api_key else '❌ Missing'}")
        print(f"🔑 Notion token: {'✅ Found' if self.notion_token else '❌ Missing'}")
        print(f"📄 PRD page ID: {self.prd_page_id if self.prd_page_id else '❌ Missing'}")
        print(f"📄 Tech spec page ID: {self.tech_spec_page_id if self.tech_spec_page_id else '❌ Missing'}")
        
        if not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize clients
        print("🤖 Initializing API clients...")
        self.claude_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        print("✅ API clients initialized")
        
        # Initialize vector store
        print("🗃️ Loading vector index...")
        self.vector_index = self._load_vector_index()
        
        # Initialize Notion cache vector store
        print("💾 Initializing Notion page cache...")
        self.notion_cache_index = self._load_notion_cache_index()
        
        # Initialize Notion reader if token is available
        if self.notion_token:
            print("📚 Initializing Notion reader...")
            self.notion_reader = NotionPageReader(integration_token=self.notion_token)
            print("✅ Notion reader initialized")
        else:
            print("⚠️ Notion reader not initialized (no token)")
            self.notion_reader = None
        
        self.prd_context = None
        print("🎉 SimpleClaudeBot initialization complete!\n")
    
    def _load_vector_index(self) -> VectorStoreIndex:
        """Load the vector index from ChromaDB."""
        try:
            chroma_path = self.storage_path / "chroma_db"
            print(f"📂 ChromaDB path: {chroma_path}")
            print(f"📂 ChromaDB exists: {chroma_path.exists()}")
            
            # Initialize ChromaDB client
            print("🔌 Connecting to ChromaDB...")
            chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            
            print("📋 Getting collection 'maintainx_codebase'...")
            chroma_collection = chroma_client.get_collection(name="maintainx_codebase")
            
            # Get collection info
            collection_count = chroma_collection.count()
            print(f"📊 Collection contains {collection_count} documents")
            
            # Create vector store
            print("🏗️ Setting up vector store...")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Load index
            print("⚡ Loading vector index...")
            index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            
            print("✅ Vector index loaded successfully")
            return index
            
        except Exception as e:
            print(f"❌ Failed to load vector index: {e}")
            raise RuntimeError(f"Failed to load vector index: {e}")
    
    def _load_notion_cache_index(self) -> Optional[VectorStoreIndex]:
        """Load or create the Notion page cache vector index."""
        try:
            notion_cache_path = self.storage_path / "notion_cache"
            print(f"📂 Notion cache path: {notion_cache_path}")
            
            # Create directory if it doesn't exist
            notion_cache_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client for Notion cache
            print("🔌 Connecting to Notion cache ChromaDB...")
            chroma_client = chromadb.PersistentClient(path=str(notion_cache_path))
            
            # Try to get existing collection or create new one
            try:
                print("📋 Getting collection 'notion_pages'...")
                chroma_collection = chroma_client.get_collection(name="notion_pages")
                collection_count = chroma_collection.count()
                print(f"📊 Notion cache contains {collection_count} cached pages")
            except Exception:
                print("📋 Creating new collection 'notion_pages'...")
                chroma_collection = chroma_client.create_collection(name="notion_pages")
                print("✅ New Notion cache collection created")
            
            # Create vector store
            print("🏗️ Setting up Notion cache vector store...")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create or load index
            if chroma_collection.count() > 0:
                print("⚡ Loading existing Notion cache index...")
                index = VectorStoreIndex.from_vector_store(
                    vector_store=vector_store,
                    storage_context=storage_context
                )
            else:
                print("🆕 Creating new Notion cache index...")
                # Create empty index with OpenAI embeddings
                embed_model = OpenAIEmbedding(api_key=self.openai_api_key)
                index = VectorStoreIndex.from_documents(
                    [],
                    storage_context=storage_context,
                    embed_model=embed_model
                )
            
            print("✅ Notion cache index ready")
            return index
            
        except Exception as e:
            print(f"⚠️ Failed to load Notion cache index: {e}")
            print("📝 Continuing without Notion page caching...")
            return None
    
    def _get_page_cache_key(self, page_id: str) -> str:
        """Generate a cache key for a Notion page."""
        return f"notion_page_{page_id}"
    
    def _is_page_cached(self, page_id: str) -> bool:
        """Check if a Notion page is cached."""
        if not self.notion_cache_index:
            return False
        
        try:
            cache_key = self._get_page_cache_key(page_id)
            # Query the cache index to see if the page exists
            retriever = self.notion_cache_index.as_retriever(similarity_top_k=1)
            results = retriever.retrieve(cache_key)
            
            # Check if we found a matching document
            for result in results:
                if hasattr(result, 'metadata') and result.metadata.get('page_id') == page_id:
                    return True
            return False
        except Exception as e:
            print(f"⚠️ Error checking cache for page {page_id}: {e}")
            return False
    
    def _save_page_to_cache(self, page_id: str, content: str, page_name: str = "page"):
        """Save a Notion page to cache."""
        if not self.notion_cache_index:
            return
        
        try:
            print(f"💾 Caching {page_name} (ID: {page_id[:8]}...)")
            
            # Create document with metadata
            doc = Document(
                text=content,
                metadata={
                    'page_id': page_id,
                    'page_name': page_name,
                    'cached_at': time.time(),
                    'content_hash': hashlib.md5(content.encode()).hexdigest()
                }
            )
            
            # Add to index
            self.notion_cache_index.insert(doc)
            print(f"✅ {page_name} cached successfully")
            
        except Exception as e:
            print(f"⚠️ Error caching page {page_id}: {e}")
    
    def _load_page_from_cache(self, page_id: str, page_name: str = "page") -> Optional[str]:
        """Load a Notion page from cache."""
        if not self.notion_cache_index:
            return None
        
        try:
            print(f"🔍 Checking cache for {page_name} (ID: {page_id[:8]}...)")
            
            # Query the cache index
            retriever = self.notion_cache_index.as_retriever(similarity_top_k=5)
            cache_key = self._get_page_cache_key(page_id)
            results = retriever.retrieve(cache_key)
            
            # Find exact match by page_id
            for result in results:
                if hasattr(result, 'metadata') and result.metadata.get('page_id') == page_id:
                    cached_at = result.metadata.get('cached_at', 0)
                    cached_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cached_at))
                    print(f"✅ Found cached {page_name} from {cached_time}")
                    return result.text
            
            print(f"❌ {page_name} not found in cache")
            return None
            
        except Exception as e:
            print(f"⚠️ Error loading page from cache {page_id}: {e}")
            return None
    
    def read_file_content(self, file_path: str) -> Optional[str]:
        """Read the full content of a file."""
        try:
            print(f"📖 Attempting to read file: {file_path}")
            
            # Handle both absolute and relative paths
            if not os.path.isabs(file_path):
                print(f"🔍 Searching for relative path: {file_path}")
                # Try to find the file in common locations
                possible_paths = [
                    Path(file_path),
                    Path.cwd() / file_path,
                    Path.cwd().parent / file_path
                ]
                
                print(f"🔍 Trying {len(possible_paths)} possible paths:")
                for i, path in enumerate(possible_paths, 1):
                    print(f"  {i}. {path} - {'✅ Found' if path.exists() and path.is_file() else '❌ Not found'}")
                    if path.exists() and path.is_file():
                        file_path = str(path)
                        print(f"✅ Using path: {file_path}")
                        break
                else:
                    print(f"❌ File not found in any location: {file_path}")
                    logger.warning(f"File not found: {file_path}")
                    return None
            
            print(f"📂 Reading file: {file_path}")
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            file_size = len(content)
            line_count = content.count('\n') + 1
            print(f"✅ File read successfully: {file_size} chars, {line_count} lines")
            return content
                
        except Exception as e:
            print(f"❌ Error reading file {file_path}: {e}")
            logger.error(f"Error reading file {file_path}: {e}")
            return None
    
    def search_relevant_files(self, query: str, top_k: int = 10) -> List[Dict]:
        """Search for relevant code chunks using vector similarity with smart filtering."""
        try:
            print(f"🔍 Starting vector search for query (length: {len(query)} chars)")
            print(f"🎯 Target results: {top_k}")
            
            # Generate query embedding
            print("🧠 Generating query embedding...")
            embedding_start = time.time()
            response = self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-3-large"
            )
            query_embedding = response.data[0].embedding
            embedding_end = time.time()
            print(f"✅ Embedding generated in {embedding_end - embedding_start:.2f}s (dimension: {len(query_embedding)})")
            
            # Query ChromaDB directly for more control (get more results to filter)
            print("🗃️ Querying ChromaDB...")
            search_start = time.time()
            chroma_client = chromadb.PersistentClient(path=str(self.storage_path / "chroma_db"))
            collection = chroma_client.get_collection(name="maintainx_codebase")
            
            search_count = top_k * 3  # Get 3x more results to filter from
            print(f"📊 Requesting {search_count} results from ChromaDB...")
            
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=search_count,
                include=['documents', 'metadatas', 'distances']
            )
            search_end = time.time()
            print(f"✅ ChromaDB search completed in {search_end - search_start:.2f}s")
            
            if not results['documents'] or not results['documents'][0]:
                print("❌ No results returned from ChromaDB")
                return []
                
            raw_result_count = len(results['documents'][0])
            print(f"📊 ChromaDB returned {raw_result_count} raw results")
            
            # Smart filtering - exclude unwanted directories/files
            excluded_patterns = {
                'e2e', 'allure-results', 'test-results', 'coverage', 'reports',
                'logs', 'temp', 'tmp', 'cache', 'dist', 'build', 'node_modules',
                '.next', 'target', 'bin', 'obj', '.pytest_cache', '.mypy_cache',
                'screenshots', 'videos', 'artifacts'
            }
            
            # Preferred file types (boost these)
            preferred_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.cpp', '.c'}
            
            print("🔧 Applying smart filtering and scoring...")
            print(f"🚫 Excluded patterns: {', '.join(sorted(excluded_patterns))}")
            print(f"⭐ Preferred extensions: {', '.join(sorted(preferred_extensions))}")
            
            # Format and filter results
            relevant_chunks = []
            filtered_count = 0
            boosted_count = 0
            
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                file_path = metadata.get('full_path', 'Unknown')
                
                # Skip files from excluded directories
                if any(excluded in file_path.lower() for excluded in excluded_patterns):
                    filtered_count += 1
                    continue
                
                # Skip specific unwanted files
                if any(file_path.lower().endswith(ext) for ext in ['.log', '.xml', '.json', '.lock', '.md']):
                    # Allow some .json and .md files if they're in important locations
                    if not any(important in file_path.lower() for important in ['config', 'package', 'readme', 'api']):
                        filtered_count += 1
                        continue
                
                base_similarity = 1.0 / (1.0 + distance)
                
                # Boost score for preferred file types
                file_ext = Path(file_path).suffix.lower()
                if file_ext in preferred_extensions:
                    boosted_similarity = base_similarity * 1.2  # 20% boost
                    boosted_count += 1
                else:
                    boosted_similarity = base_similarity
                
                relevant_chunks.append({
                    'file_path': file_path,
                    'content': doc,
                    'similarity_score': boosted_similarity,
                    'chunk_index': metadata.get('chunk_index', 0),
                    'file_extension': file_ext,
                    'original_distance': distance
                })
            
            print(f"📊 Filtering results:")
            print(f"  • Raw results: {raw_result_count}")
            print(f"  • Filtered out: {filtered_count}")
            print(f"  • Boosted (preferred types): {boosted_count}")
            print(f"  • Remaining: {len(relevant_chunks)}")
            
            # Sort by boosted similarity and return top_k
            relevant_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = relevant_chunks[:top_k]
            
            print(f"✅ Returning top {len(final_results)} results")
            
            # Show similarity score distribution
            if final_results:
                scores = [chunk['similarity_score'] for chunk in final_results]
                print(f"📊 Similarity scores: min={min(scores):.3f}, max={max(scores):.3f}, avg={sum(scores)/len(scores):.3f}")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            logger.error(f"Error searching relevant files: {e}")
            return []
    
    async def load_page(self, page_id: str, page_name: str = "page") -> str:
        """Load content from a Notion page, with caching support."""
        if not self.notion_reader:
            error_msg = "Notion token not configured. Set NOTION_TOKEN in .env file."
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        if not page_id:
            error_msg = f"No {page_name} page ID provided."
            print(f"❌ {error_msg}")
            raise ValueError(error_msg)
        
        try:
            # First, try to load from cache
            cached_content = self._load_page_from_cache(page_id, page_name)
            if cached_content:
                print(f"⚡ Using cached {page_name} content ({len(cached_content)} chars)")
                
                # Only update prd_context if this is the PRD page
                if page_name.lower() == "prd":
                    self.prd_context = cached_content
                
                return cached_content
            
            # If not cached, load from Notion API
            print(f"📖 Loading {page_name} from Notion API (ID: {page_id})...")
            load_start = time.time()
            
            # Load with timeout handling
            documents = self.notion_reader.load_data(page_ids=[page_id])
            load_end = time.time()
            
            print(f"⏱️ {page_name} API call completed in {load_end - load_start:.2f}s")
            print(f"📊 Retrieved {len(documents)} document(s)")
            
            if not documents:
                print(f"⚠️ No content found in {page_name}")
                return f"No content found in {page_name}."
            
            # Combine all document content
            print("🔧 Processing document content...")
            content_parts = []
            for i, doc in enumerate(documents):
                doc_text = doc.text.strip()
                if doc_text:
                    content_parts.append(doc_text)
                    print(f"  Document {i+1}: {len(doc_text)} chars")
                else:
                    print(f"  Document {i+1}: empty (skipped)")
            
            page_content = "\n\n".join(content_parts)
            
            # Cache the content for next time
            self._save_page_to_cache(page_id, page_content, page_name)
            
            # Only update prd_context if this is the PRD page
            if page_name.lower() == "prd":
                self.prd_context = page_content
            
            print(f"✅ {page_name} content loaded: {len(page_content)} chars total")
            print(f"📊 Content preview: {page_content[:200]}{'...' if len(page_content) > 200 else ''}")
            
            return page_content
            
        except Exception as e:
            error_msg = f"Error loading {page_name}: {str(e)}"
            print(f"❌ {error_msg}")
            logger.error(error_msg)
            return error_msg
    
    async def analyze_notion_with_codebase(self, page_id: Optional[str] = None) -> str:
        """Analyze PRD and Tech Spec against codebase for potential issues."""
        print("🚀 Starting comprehensive PRD-Codebase analysis...")
        analysis_start_time = time.time()
        
        # Get document page IDs from environment variables
        prd_page_id = self.prd_page_id
        tech_spec_page_id = self.tech_spec_page_id
        
        print(f"📋 Document configuration:")
        print(f"  • PRD page ID: {prd_page_id if prd_page_id else '❌ Not configured (PRD_PAGE_ID)'}")
        print(f"  • Tech spec page ID: {tech_spec_page_id if tech_spec_page_id else '❌ Not configured (TECH_SPEC_PAGE_ID)'}")
        
        if not prd_page_id:
            final_msg = "❌ PRD_PAGE_ID environment variable is required"
            print(final_msg)
            return final_msg
        
        # Step 1: Load both documents efficiently in parallel
        print("\n" + "="*50)
        print("🔄 STEP 1: Loading documents")
        print("="*50)
        step1_start = time.time()
        
        # Prepare loading tasks
        loading_tasks = []
        task_names = []
        
        # Always load PRD
        prd_task = self.load_page(prd_page_id, "PRD")
        loading_tasks.append(prd_task)
        task_names.append("PRD")
        
        # Load tech spec if configured
        tech_spec_task = None
        if tech_spec_page_id:
            tech_spec_task = self.load_page(tech_spec_page_id, "Tech Spec")
            loading_tasks.append(tech_spec_task)
            task_names.append("Tech Spec")
        else:
            print("⚠️ Tech spec page ID not configured (TECH_SPEC_PAGE_ID)")
            print("📝 Will continue with PRD only...")
        
        # Load all pages in parallel
        if len(loading_tasks) > 1:
            print(f"⚡ Loading {len(loading_tasks)} documents in parallel...")
        else:
            print("📖 Loading PRD document...")
        
        try:
            results = await asyncio.gather(*loading_tasks, return_exceptions=True)
        except Exception as e:
            final_msg = f"❌ Error during parallel loading: {str(e)}"
            print(final_msg)
            return final_msg
        
        # Process results
        prd_content = results[0]
        tech_spec_content = None
        
        # Check PRD result
        if isinstance(prd_content, Exception) or not prd_content or "Error" in str(prd_content):
            final_msg = f"❌ Failed to load PRD content: {prd_content}"
            print(final_msg)
            return final_msg
        
        # Check tech spec result if it was loaded
        if len(results) > 1:
            tech_spec_result = results[1]
            if isinstance(tech_spec_result, Exception) or not tech_spec_result or "Error" in str(tech_spec_result):
                print(f"⚠️ Failed to load tech spec: {tech_spec_result}")
                print("📝 Continuing with PRD only...")
                tech_spec_content = None
            else:
                tech_spec_content = tech_spec_result
                print("✅ Tech spec loaded successfully")
        
        # Combine contents for vector search
        if tech_spec_content:
            combined_content = f"{prd_content}\n\n--- TECH SPEC ---\n\n{tech_spec_content}"
        else:
            combined_content = prd_content
        
        step1_end = time.time()
        print(f"📊 Content summary:")
        print(f"  • PRD document: {len(prd_content):,} chars")
        if tech_spec_content:
            print(f"  • Tech spec: {len(tech_spec_content):,} chars")
            print(f"  • Combined: {len(combined_content):,} chars")
        else:
            print(f"  • Tech spec: Not loaded")
            print(f"  • Total: {len(combined_content):,} chars")
        print(f"⏱️ Step 1 completed in {step1_end - step1_start:.2f}s (parallel loading)")
        
        if len(loading_tasks) > 1:
            estimated_sequential_time = step1_end - step1_start  # This would be longer if sequential
            print(f"💨 Parallel loading saved significant time vs sequential loading")
        
        # Step 2: Find related files using combined content as query
        print("\n" + "="*50)
        print("🔄 STEP 2: Finding related files in codebase")
        print("="*50)
        step2_start = time.time()
        
        print("🎯 Using combined content as search query...")
        print(f"📊 Query length: {len(combined_content)} characters")
        
        # Adaptive file retrieval with retry logic
        max_attempts = 4
        top_k_values = [30, 20, 15, 10]  # Try progressively smaller values
        
        for attempt, top_k in enumerate(top_k_values, 1):
            print(f"\n🔄 Attempt {attempt}/{max_attempts}: Trying top_k={top_k}")
            
            related_chunks = self.search_relevant_files(combined_content, top_k=top_k)
            
            step2_end = time.time()
            print(f"⏱️ Step 2 completed in {step2_end - step2_start:.2f}s")
            
            # Step 3: Read full content of the most relevant files
            print("\n" + "="*50)
            print("🔄 STEP 3: Reading full file contents")
            print("="*50)
            step3_start = time.time()
            
            # Group chunks by file and select top files
            print("🔍 Consolidating unique files from chunks...")
            file_chunks = {}
            for chunk in related_chunks:
                file_path = chunk['file_path']
                if file_path not in file_chunks:
                    file_chunks[file_path] = []
                file_chunks[file_path].append(chunk)
            
            # Calculate file scores and select top files
            unique_files = {}
            for file_path, chunks in file_chunks.items():
                max_score = max(chunk['similarity_score'] for chunk in chunks)
                file_ext = chunks[0]['file_extension']
                unique_files[file_path] = {
                    'max_score': max_score,
                    'chunk_count': len(chunks),
                    'extension': file_ext
                }
            
            print(f"📊 Found {len(unique_files)} unique files from {len(related_chunks)} chunks")
            
            # Sort files by score and take top 8
            sorted_files = sorted(unique_files.items(), key=lambda x: x[1]['max_score'], reverse=True)
            top_files = sorted_files[:8]
            
            print(f"📁 Will read top {len(top_files)} files:")
            for i, (file_path, file_info) in enumerate(top_files, 1):
                score_indicator = "🔥" if file_info['max_score'] > 0.8 else "⭐" if file_info['max_score'] > 0.6 else "📄"
                print(f"  {i}. {score_indicator} {file_path}")
                print(f"     Score: {file_info['max_score']:.3f}, Chunks: {file_info['chunk_count']}, Type: {file_info['extension']}")
            
            print(f"\n📖 Reading file contents...")
            
            # Read file contents
            codebase_context = ""
            files_read = 0
            total_chars = 0
            
            for file_path, file_info in top_files:
                print(f"\n📂 Processing: {file_path}")
                file_content = self.read_file_content(file_path)
                
                if file_content:
                    codebase_context += f"\n\n--- FILE: {file_path} ---\n{file_content}"
                    files_read += 1
                    total_chars += len(file_content)
                    print("✅ Added to analysis context")
                else:
                    print("❌ Skipped (could not read)")
            
            print(f"\n📊 Step 3 summary:")
            print(f"  • Files successfully read: {files_read}")
            print(f"  • Total characters: {total_chars:,}")
            if files_read > 0:
                print(f"  • Average file size: {total_chars // files_read:,} chars")
            
            step3_end = time.time()
            print(f"⏱️ Step 3 completed in {step3_end - step3_start:.2f}s")
            
            # Step 4: Analyze with Claude (with retry logic)
            print("\n" + "="*50)
            print("🔄 STEP 4: Analyzing with Claude")
            print("="*50)
            step4_start = time.time()
            
            print("🏗️ Building analysis context...")
            print(f"📊 Analysis context (before token limiting):")
            print(f"  • PRD document: {len(prd_content):,} chars")
            if tech_spec_content:
                print(f"  • Tech spec: {len(tech_spec_content):,} chars")
            print(f"  • Codebase context: {len(codebase_context)+1:,} chars")
            print(f"  • Total context: {len(prd_content) + len(tech_spec_content or '') + len(codebase_context):,} chars")
            
            # Apply token limiting
            print(f"\n🔒 Applying token limits...")
            limited_prd, limited_tech_spec, limited_codebase = self.limit_context_tokens(
                prd_content, tech_spec_content, codebase_context
            )
            
            # Create analysis prompt
            print("✍️ Creating analysis prompt...")
            analysis_prompt = f"""You are an expert software architect and technical analyst. Please analyze the following Product Requirements Document (PRD) against the current codebase implementation to identify potential issues, gaps, and recommendations.

**ANALYSIS INSTRUCTIONS:**
1. **Consistency Check**: Compare PRD requirements with existing code patterns and implementations
2. **Gap Analysis**: Identify features mentioned in PRD that may not be implemented yet
3. **Implementation Review**: Check if current code aligns with PRD specifications
4. **Recommendations**: Suggest improvements, missing components, or potential issues

**PRD DOCUMENT:**
{limited_prd}

{"**TECH SPEC DOCUMENT:**" if limited_tech_spec else ""}
{limited_tech_spec or ""}

**RELEVANT CODEBASE CONTEXT:**
{limited_codebase}

**ANALYSIS REQUEST:**
Please provide a comprehensive analysis covering:
- **Alignment Issues**: Where the code doesn't match PRD requirements
- **Missing Features**: PRD features not found in the codebase
- **Implementation Gaps**: Incomplete or partial implementations
- **Code Quality**: Areas where implementation could be improved
- **Recommendations**: Specific actionable suggestions

Focus on actionable insights that would help developers implement the PRD requirements effectively."""
            
            print(f"📋 Final prompt length: {len(analysis_prompt):,} chars")
            
            try:
                print(f"🤖 Sending request to Claude (attempt {attempt} with top_k={top_k})...")
                claude_start = time.time()
                
                response = self.claude_client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4000,
                    temperature=0.1,
                    messages=[{"role": "user", "content": analysis_prompt}]
                )
                
                claude_end = time.time()
                claude_duration = claude_end - claude_start
                print(f"⚡ Claude responded in {claude_duration:.2f}s")
                
                step4_end = time.time()
                print(f"⏱️ Step 4 completed in {step4_end - step4_start:.2f}s")
                
                # Success! Break out of retry loop
                analysis_result = response.content[0].text
                
                analysis_end_time = time.time()
                total_duration = analysis_end_time - analysis_start_time
                
                print("\n" + "="*70)
                print("🎯 CLAUDE ANALYSIS RESULTS")
                print("="*70)
                print(analysis_result)
                print("="*70)
                
                print(f"\n📊 PERFORMANCE SUMMARY:")
                print(f"  • Step 1 (Loading): {step1_end - step1_start:.2f}s")
                print(f"  • Step 2 (Search): {step2_end - step2_start:.2f}s") 
                print(f"  • Step 3 (Reading): {step3_end - step3_start:.2f}s")
                print(f"  • Step 4 (Analysis): {step4_end - step4_start:.2f}s")
                print(f"    - Claude response time: {claude_duration:.2f}s")
                print(f"  • Total time: {total_duration:.2f}s")
                print(f"  • Successful with top_k={top_k} (attempt {attempt})")
                
                return analysis_result
                
            except Exception as e:
                error_msg = str(e)
                print(f"❌ Error during Claude analysis (attempt {attempt}): {error_msg}")
                
                # Check if it's a token limit error
                if "prompt is too long" in error_msg or "tokens >" in error_msg:
                    print(f"🔄 Token limit exceeded with top_k={top_k}, trying smaller value...")
                    if attempt < max_attempts:
                        continue  # Try next smaller top_k value
                    else:
                        print(f"❌ All retry attempts failed. Final error: {error_msg}")
                        return f"Analysis failed after {max_attempts} attempts. Final error: {error_msg}"
                else:
                    # Non-token-limit error, don't retry
                    print(f"❌ Non-recoverable error: {error_msg}")
                    return f"Analysis failed: {error_msg}"
        
        # This should never be reached due to the loop structure, but just in case
        return "Analysis failed: Unexpected error in retry logic"
    
    async def query_with_vector_context(self, query: str, include_prd_context: bool = True) -> str:
        """Query using vector search context and Claude API."""
        query_start_time = time.time()
        print(f"🔍 Searching for relevant code...")
        
        # Search for relevant code chunks
        relevant_chunks = self.search_relevant_files(query, top_k=8)
        
        if not relevant_chunks:
            print("⚠️  No relevant code found")
            return "No relevant code found for your query."
        
        # Show found files with types
        unique_files = {}
        for chunk in relevant_chunks:
            file_path = chunk['file_path']
            if file_path not in unique_files:
                unique_files[file_path] = {
                    'extension': chunk['file_extension'],
                    'max_score': chunk['similarity_score']
                }
            else:
                unique_files[file_path]['max_score'] = max(
                    unique_files[file_path]['max_score'], 
                    chunk['similarity_score']
                )
        
        print(f"📁 Found relevant code in {len(unique_files)} files:")
        for file_path in sorted(unique_files.keys()):
            file_info = unique_files[file_path]
            score_indicator = "🔥" if file_info['max_score'] > 0.8 else "⭐" if file_info['max_score'] > 0.6 else "📄"
            print(f"  {score_indicator} {file_path} ({file_info['extension']})")
        
        # Build context from relevant chunks
        code_context = "\n\n".join([
            f"File: {chunk['file_path']} (similarity: {chunk['similarity_score']:.3f})\n"
            f"```\n{chunk['content']}\n```"
            for chunk in relevant_chunks[:6]  # Limit to top 6 chunks
        ])
        
        # Build prompt with optional Notion context
        prompt_parts = []
        
        if include_prd_context and self.prd_context:
            prompt_parts.append(f"Context from PRD:\n{self.prd_context}\n")
        
        prompt_parts.extend([
            f"Relevant code from codebase:\n{code_context}\n",
            f"Based on the above context, please answer: {query}"
        ])
        
        full_prompt = "\n".join(prompt_parts)
        
        print(f"🤖 Asking Claude...")
        
        try:
            # Query Claude API
            response = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=2000,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ]
            )
            
            query_end_time = time.time()
            total_query_time = query_end_time - query_start_time
            
            answer = response.content[0].text
            
            print(f"\n💡 Claude's Response:")
            print("-" * 50)
            print(answer)
            print("-" * 50)
            print(f"🎯 Total query time: {total_query_time:.2f} seconds")
            
            return answer
            
        except Exception as e:
            error_msg = f"Error querying Claude: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            return error_msg
    
    async def analyze_codebase(self) -> str:
        """Get a high-level analysis of the codebase."""
        analysis_start_time = time.time()
        
        if not self.prd_context:
            print("⚠️  No PRD context available for analysis")
            return "No PRD context available for analysis"
        
        prompt = """
Based on the PRD context provided, what are the files in the codebase that are most relevant?
Return a JSON array of file paths that would be most important for understanding this project.
Be concise and focus on the most critical files.
"""
        
        result = await self.query_with_vector_context(prompt, include_prd_context=True)
        
        analysis_end_time = time.time()
        analysis_time = analysis_end_time - analysis_start_time
        print(f"📊 Analysis completed in {analysis_time:.2f} seconds")
        
        return result
    
    async def run_interactive_session(self):
        """Run an interactive session for querying."""
        print("🚀 Simple Claude Code Bot with Vector Search Started!")
        print("Commands:")
        print("  • Type any question about the codebase")
        print("  • 'analyze' - Get codebase overview")
        print("  • 'analyze-notion' - Analyze PRD + Tech Spec against codebase")
        print("  • 'load-prd' - Load PRD content")
        print("  • 'load-prd <page_id>' - Load specific PRD page")
        print("  • 'context' - Show current PRD context")
        print("  • 'quit' - Exit")
        print("-" * 50)
        
                # Try to load PRD content on startup if configured
        if self.notion_reader and self.prd_page_id:
            try:
                await self.load_page(self.prd_page_id, "PRD")
                print("✅ PRD content loaded on startup")
            except Exception as e:
                print(f"⚠️  Could not load PRD page on startup: {e}")
        
        while True:
            try:
                user_input = input("\n💭 Your question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'analyze':
                    print("📊 Analyzing codebase...")
                    await self.analyze_codebase()
                
                elif user_input.lower().startswith('analyze-notion'):
                    print("🔍 Analyzing PRD + Tech Spec against codebase...")
                    await self.analyze_notion_with_codebase()
                
                elif user_input.lower().startswith('load-prd'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        # Load specific page
                        page_id = parts[1]
                        response = await self.load_page(page_id, "PRD")
                    else:
                        # Load default PRD page
                        if self.prd_page_id:
                            response = await self.load_page(self.prd_page_id, "PRD")
                        else:
                            response = "❌ PRD_PAGE_ID not configured"
                    print(f"\n{response}")
                
                elif user_input.lower() == 'context':
                    if self.prd_context:
                        print(f"\n📝 Current PRD Context ({len(self.prd_context)} chars):")
                        print(f"{self.prd_context[:500]}{'...' if len(self.prd_context) > 500 else ''}")
                    else:
                        print("📝 No PRD context loaded. Use 'load-prd' command to load content.")
                
                else:
                    if not user_input:
                        continue
                    await self.query_with_vector_context(user_input, include_prd_context=True)
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (roughly 3.5 characters = 1 token for Claude)."""
        return int(len(text) / 3.5)
    
    def limit_context_tokens(self, prd_content: str, tech_spec_content: Optional[str], codebase_context: str, max_tokens: int = 200000) -> tuple[str, Optional[str], str]:
        """Limit context to fit within token limits, prioritizing PRD > Tech Spec > Codebase."""
        # Reserve tokens for prompt overhead (instructions, formatting, etc.)
        prompt_overhead = 5000  # Increased buffer for actual prompt overhead
        available_tokens = max_tokens - prompt_overhead
        
        prd_tokens = self.estimate_tokens(prd_content)
        tech_spec_tokens = self.estimate_tokens(tech_spec_content) if tech_spec_content else 0
        codebase_tokens = self.estimate_tokens(codebase_context)
        
        total_content_tokens = prd_tokens + tech_spec_tokens + codebase_tokens
        
        print(f"🧮 Token estimation:")
        print(f"  • PRD: {prd_tokens:,} tokens")
        if tech_spec_content:
            print(f"  • Tech spec: {tech_spec_tokens:,} tokens")
        print(f"  • Codebase: {codebase_tokens:,} tokens")
        print(f"  • Content total: {total_content_tokens:,} tokens")
        print(f"  • Prompt overhead: {prompt_overhead:,} tokens")
        print(f"  • Grand total: {total_content_tokens + prompt_overhead:,} tokens")
        print(f"  • Limit: {max_tokens:,} tokens")
        print(f"  • Available for content: {available_tokens:,} tokens")
        
        if total_content_tokens <= available_tokens:
            print("✅ Within token limit")
            return prd_content, tech_spec_content, codebase_context
        
        print("⚠️ Exceeds token limit, truncating...")
        
        # Always keep full PRD (highest priority)
        remaining_tokens = available_tokens - prd_tokens
        print(f"📋 Keeping full PRD ({prd_tokens:,} tokens)")
        
        # Try to keep tech spec if it fits
        if tech_spec_content and tech_spec_tokens <= remaining_tokens:
            remaining_tokens -= tech_spec_tokens
            print(f"📋 Keeping full tech spec ({tech_spec_tokens:,} tokens)")
        elif tech_spec_content:
            # Truncate tech spec to fit partially
            if remaining_tokens > 1000:  # Keep some space for codebase
                tech_spec_chars = int((remaining_tokens - 1000) * 3.5)
                tech_spec_content = tech_spec_content[:tech_spec_chars] + "\n\n... [TRUNCATED FOR TOKEN LIMIT]"
                tech_spec_tokens = self.estimate_tokens(tech_spec_content)
                remaining_tokens -= tech_spec_tokens
                print(f"📋 Truncated tech spec to {tech_spec_tokens:,} tokens")
            else:
                tech_spec_content = None
                tech_spec_tokens = 0
                print("📋 Removed tech spec (no space)")
        
        # Use remaining tokens for codebase
        if remaining_tokens > 0:
            codebase_chars = int(remaining_tokens * 3.5)
            if len(codebase_context) > codebase_chars:
                codebase_context = codebase_context[:codebase_chars] + "\n\n... [TRUNCATED FOR TOKEN LIMIT]"
                codebase_tokens = self.estimate_tokens(codebase_context)
                print(f"📋 Truncated codebase to {codebase_tokens:,} tokens")
            else:
                print(f"📋 Keeping full codebase ({codebase_tokens:,} tokens)")
        else:
            codebase_context = ""
            print("📋 Removed codebase (no space)")
        
        final_content_total = self.estimate_tokens(prd_content) + (self.estimate_tokens(tech_spec_content) if tech_spec_content else 0) + self.estimate_tokens(codebase_context)
        final_grand_total = final_content_total + prompt_overhead
        print(f"✅ Final content: {final_content_total:,} tokens")
        print(f"✅ Final grand total: {final_grand_total:,} tokens")
        
        return prd_content, tech_spec_content, codebase_context

    def clear_notion_cache(self, page_id: Optional[str] = None):
        """Clear the Notion page cache (all pages or specific page)."""
        if not self.notion_cache_index:
            print("❌ Notion cache not available")
            return
        
        try:
            if page_id:
                print(f"🗑️ Clearing cache for page {page_id[:8]}...")
                # Note: ChromaDB doesn't have a direct delete by metadata feature
                # For now, we'll recreate the entire cache index
                print("⚠️ Individual page cache clearing not implemented yet")
                print("💡 Use clear_notion_cache() without page_id to clear all cache")
            else:
                print("🗑️ Clearing entire Notion page cache...")
                
                # Delete the cache directory and recreate
                notion_cache_path = self.storage_path / "notion_cache"
                if notion_cache_path.exists():
                    import shutil
                    shutil.rmtree(notion_cache_path)
                    print("✅ Cache directory deleted")
                
                # Reinitialize the cache index
                self.notion_cache_index = self._load_notion_cache_index()
                print("✅ Notion cache cleared and reinitialized")
                
        except Exception as e:
            print(f"⚠️ Error clearing cache: {e}")


async def main():
    """Main entry point - automatically analyze PRD + Tech Spec against codebase."""
    try:
        print("🚀 Starting PRD-Codebase Analysis...")
        bot = SimpleClaudeBot()
        
        # Run the analysis automatically
        result = await bot.analyze_notion_with_codebase()
        
        if result.startswith("❌"):
            print(f"\n{result}")
            return
        
        print(f"\n✅ Analysis complete! Check the detailed output above.")
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error starting application: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 