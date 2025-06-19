import os
import chromadb
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client for manual embedding generation
openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize ChromaDB client (use PersistentClient to match the embeddings files)
chroma_client = chromadb.PersistentClient(path="./vector_storage/chroma_db")

# Get existing collection (without embedding function since it was created without one)
collection = chroma_client.get_collection(name="maintainx_codebase")

# Query ChromaDB - generate embedding manually since collection has no embedding function
query_text = "I am looking for files that are dealing with showing messages in the frontend"
print(f"Querying with: {query_text}")

# Generate query embedding using OpenAI (same as the embedding scripts used)
response = openai_client.embeddings.create(
    input=[query_text],
    model="text-embedding-3-large"
)
query_embedding = response.data[0].embedding

results = collection.query(
    query_embeddings=[query_embedding],  # Use manually generated embedding
    n_results=50,
    # include=['documents', 'metadatas', 'distances']
)

# Print file paths from results
if results['metadatas'] and results['metadatas'][0]:
    print("\nFound files:")
    for metadata in results['metadatas'][0]:
        print(f"- {metadata['file_path']}")
