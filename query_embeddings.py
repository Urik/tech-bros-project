#!/usr/bin/env python3
"""
Codebase Embedding Query Tool

Query the vector embeddings created by create_embeddings.py to find relevant
code snippets and files based on semantic similarity using OpenAI embeddings.
"""

import os
import logging
import time
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

import openai
import chromadb
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CodebaseQueryEngine:
    """Query engine for codebase embeddings using OpenAI."""
    
    def __init__(
        self,
        storage_path: str = "./vector_storage",
        collection_name: str = "maintainx_codebase"
    ):
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name
        
        # Validate storage path
        if not self.storage_path.exists():
            raise ValueError(f"Storage path does not exist: {self.storage_path}")
        
        # Initialize OpenAI client
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        self.model_name = "text-embedding-3-large"
        
        # Load the ChromaDB collection
        self.collection = self.load_collection()
    
    def load_collection(self):
        """Load the ChromaDB collection."""
        try:
            # Initialize ChromaDB client
            chroma_client = chromadb.PersistentClient(path=str(self.storage_path / "chroma_db"))
            
            # Get collection
            collection = chroma_client.get_collection(name=self.collection_name)
            
            logger.info(f"✅ Loaded collection from {self.storage_path}")
            return collection
            
        except Exception as e:
            raise RuntimeError(f"Failed to load collection: {e}")
    
    def get_query_embedding(self, query_text: str) -> List[float]:
        """Generate embedding for the query text using OpenAI."""
        try:
            response = self.openai_client.embeddings.create(
                input=[query_text],
                model=self.model_name
            )
            
            if not response or not response.data:
                raise RuntimeError("No embedding returned from OpenAI")
            
            return response.data[0].embedding
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate query embedding: {e}")
    
    def query(self, query_text: str, top_k: int = 5) -> dict:
        """Query the codebase for relevant code snippets."""
        start_time = time.time()
        
        logger.info(f"Querying: {query_text}")
        
        try:
            # Generate embedding for the query
            query_embedding = self.get_query_embedding(query_text)
            
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )
            
            end_time = time.time()
            query_time = end_time - start_time
            
            # Process results
            query_results = {
                "query": query_text,
                "query_time": query_time,
                "num_results": len(results['documents'][0]) if results['documents'] else 0,
                "sources": []
            }
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0] if results['metadatas'] else []
                distances = results['distances'][0] if results['distances'] else []
                
                for i, (doc, metadata, distance) in enumerate(zip(documents, metadatas, distances)):
                    # Convert distance to similarity score (lower distance = higher similarity)
                    similarity_score = 1.0 / (1.0 + distance) if distance > 0 else 1.0
                    
                    source_info = {
                        "rank": i + 1,
                        "score": similarity_score,
                        "distance": distance,
                        "file_path": metadata.get('file_path', 'Unknown'),
                        "chunk_index": metadata.get('chunk_index', 0),
                        "content": doc[:500] + "..." if len(doc) > 500 else doc,
                        "full_content": doc
                    }
                    query_results["sources"].append(source_info)
            
            return query_results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {
                "query": query_text,
                "error": str(e),
                "query_time": time.time() - start_time
            }
    
    def search_similar_code(self, code_snippet: str, top_k: int = 5) -> dict:
        """Find code similar to the provided snippet."""
        query_text = f"Find code similar to this: {code_snippet}"
        return self.query(query_text, top_k)
    
    def find_function_usage(self, function_name: str, top_k: int = 10) -> dict:
        """Find usage of a specific function."""
        query_text = f"Show me usage of function {function_name}"
        return self.query(query_text, top_k)
    
    def find_class_definition(self, class_name: str, top_k: int = 5) -> dict:
        """Find class definitions."""
        query_text = f"Find class definition for {class_name}"
        return self.query(query_text, top_k)
    
    def get_stats(self) -> dict:
        """Get statistics about the loaded collection."""
        try:
            stats = {
                "collection_name": self.collection.name,
                "document_count": self.collection.count(),
                "storage_path": str(self.storage_path)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}


def print_query_results(results: dict, show_full_content: bool = False):
    """Pretty print query results."""
    if "error" in results:
        print(f"❌ Query failed: {results['error']}")
        return
    
    print(f"\n🔍 Query: {results['query']}")
    print(f"⏱️  Query time: {results['query_time']:.2f} seconds")
    print(f"📊 Found {results['num_results']} relevant code snippets")
    
    if results['sources']:
        print(f"\n📝 Top {len(results['sources'])} Most Relevant Code Snippets:")
        print("=" * 80)
        
        for source in results['sources']:
            print(f"\n#{source['rank']} - Similarity: {source['score']:.4f} (Distance: {source['distance']:.4f})")
            print(f"📁 File: {source['file_path']} (Chunk {source['chunk_index']})")
            print(f"📄 Content:")
            print("-" * 40)
            
            content_to_show = source['full_content'] if show_full_content else source['content']
            print(content_to_show)
            print("-" * 40)


def interactive_query_session(query_engine: CodebaseQueryEngine):
    """Run an interactive query session."""
    print("🚀 Codebase Query Engine Started (OpenAI Embeddings)!")
    print("Commands:")
    print("  • Type any question about the codebase")
    print("  • 'similar <code>' - Find similar code")
    print("  • 'function <name>' - Find function usage")
    print("  • 'class <name>' - Find class definition")
    print("  • 'stats' - Show collection statistics")
    print("  • 'quit' - Exit")
    print("-" * 60)
    
    # Show initial stats
    stats = query_engine.get_stats()
    print("📊 Collection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n💭 Your query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if user_input.lower() == 'stats':
                stats = query_engine.get_stats()
                print("\n📊 Collection Statistics:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
                continue
            
            if user_input.lower().startswith('similar '):
                code_snippet = user_input[8:].strip()
                results = query_engine.search_similar_code(code_snippet)
                print_query_results(results)
                continue
            
            if user_input.lower().startswith('function '):
                function_name = user_input[9:].strip()
                results = query_engine.find_function_usage(function_name)
                print_query_results(results)
                continue
            
            if user_input.lower().startswith('class '):
                class_name = user_input[6:].strip()
                results = query_engine.find_class_definition(class_name)
                print_query_results(results)
                continue
            
            if not user_input:
                continue
            
            # Regular query
            results = query_engine.query(user_input)
            print_query_results(results)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Query codebase embeddings using OpenAI")
    parser.add_argument(
        "--storage-path",
        default="./vector_storage",
        help="Path to the vector index storage (default: ./vector_storage)"
    )
    parser.add_argument(
        "--collection-name",
        default="maintainx_codebase",
        help="Name of the ChromaDB collection (default: maintainx_codebase)"
    )
    parser.add_argument(
        "--query",
        help="Single query to execute (non-interactive mode)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return (default: 5)"
    )
    parser.add_argument(
        "--show-full-content",
        action="store_true",
        help="Show full content of code snippets (not truncated)"
    )
    
    args = parser.parse_args()
    
    try:
        # Create query engine
        query_engine = CodebaseQueryEngine(
            storage_path=args.storage_path,
            collection_name=args.collection_name
        )
        
        if args.query:
            # Single query mode
            results = query_engine.query(args.query, top_k=args.top_k)
            print_query_results(results, show_full_content=args.show_full_content)
        else:
            # Interactive mode
            interactive_query_session(query_engine)
        
    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 