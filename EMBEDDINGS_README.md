# Codebase Embeddings with OpenAI

This system creates semantic embeddings for your codebase using OpenAI's `text-embedding-3-large` model and stores them locally in ChromaDB. Your code never leaves your machine except for the embedding generation calls to OpenAI.

## Setup

1. **Install Dependencies**
   ```bash
   python setup_embeddings.py
   ```

2. **Set up API Key**
   Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```

3. **Verify Setup**
   The setup script will check for required dependencies and API keys.

## Usage

### Generate Embeddings

Use the simplified embedding generator:

```bash
# Generate embeddings for default path (~/repos/maintainx)
python simple_embeddings.py

# Generate embeddings for a specific codebase
python simple_embeddings.py /path/to/your/codebase

# Custom storage location
python simple_embeddings.py --storage-path ./my_embeddings

# Show stats only
python simple_embeddings.py --stats-only
```

### Query Embeddings

Once embeddings are generated, query them:

```bash
# Interactive mode
python query_embeddings.py

# Single query
python query_embeddings.py --query "How to handle authentication?"

# Show more results
python query_embeddings.py --query "database connection" --top-k 10
```

## Features

- **Local Storage**: All embeddings are stored locally in ChromaDB
- **Privacy**: Only text chunks are sent to OpenAI for embedding; your full codebase stays local
- **Fast Queries**: Semantic search across your entire codebase
- **Code-Aware**: Handles multiple programming languages and file types
- **Chunking**: Intelligently splits large files into manageable chunks
- **Interactive**: Query your codebase using natural language

## Supported File Types

- Programming languages: `.py`, `.js`, `.ts`, `.jsx`, `.tsx`, `.java`, `.cpp`, `.c`, `.h`, `.cs`, `.php`, `.rb`, `.go`, `.rs`, `.swift`, `.kt`, `.scala`
- Web: `.html`, `.css`, `.scss`, `.less`
- Data: `.json`, `.xml`, `.yaml`, `.yml`, `.sql`
- Documentation: `.md`, `.txt`
- Scripts: `.sh`, `.bash`, `.zsh`, `.fish`, `.ps1`, `.bat`
- Config: `Dockerfile`, `Makefile`, etc.

## Query Examples

### Interactive Mode Commands

```
# General questions
"How is user authentication handled?"
"Show me database migration code"
"Find error handling patterns"

# Specific searches
similar def authenticate_user():
function createConnection
class UserService

# Get statistics
stats
```

### Command Line Queries

```bash
# Find authentication code
python query_embeddings.py --query "user authentication login"

# Find database code
python query_embeddings.py --query "database connection pooling"

# Find error handling
python query_embeddings.py --query "error handling try catch"
```

## Architecture

- **Embedding Model**: OpenAI `text-embedding-3-large` (3072 dimensions)
- **Vector Store**: ChromaDB for local storage
- **Chunking**: Intelligent code-aware chunking with overlap
- **Query**: Semantic similarity search using cosine distance

## Privacy & Security

- ✅ All embeddings stored locally
- ✅ No codebase structure sent to OpenAI
- ✅ Only individual text chunks sent for embedding
- ✅ Persistent local storage
- ✅ Works offline after embeddings are generated

## Troubleshooting

### Common Issues

1. **"No existing collection found"**
   - Run `python simple_embeddings.py` first to generate embeddings

2. **"OPENAI_API_KEY not found"**
   - Set your API key in `.env` file or environment variable

3. **"No files found to process"**
   - Check that your codebase path is correct
   - Ensure you have supported file types in your codebase

4. **"Failed to generate embeddings"**
   - Check your OpenAI API key and billing status
   - Check internet connection

### Performance Tips

- Use smaller batch sizes if you hit rate limits
- The initial embedding generation can take time for large codebases
- Queries are fast once embeddings are created
- ChromaDB is optimized for local storage and retrieval

## 📊 Expected Performance

- **Small Project** (< 1,000 files): 5-15 minutes
- **Medium Project** (1,000-5,000 files): 15-45 minutes  
- **Large Project** (5,000+ files): 1-3 hours

Processing time depends on:
- Number of files
- File sizes
- OpenAI API response time
- System performance

## 🔍 Query Performance

- **Index Loading**: 1-3 seconds
- **Simple Queries**: < 1 second
- **Complex Queries**: 1-5 seconds
- **Batch Queries**: Scales linearly

## 💡 Tips for Better Results

### Query Writing
- Be specific: "React component for user authentication" vs "authentication"
- Use technical terms: "async function", "class definition", "error handling"
- Include context: "database connection in Node.js" vs "database"

### Code Organization
- Keep related code in logical directories
- Use descriptive file and function names
- Add comments for complex logic

### Maintenance
- Regenerate embeddings when codebase changes significantly
- Use `--stats-only` to monitor index size
- Clean up old vector storage periodically

## 🤝 Integration

This embedding system can be integrated with:
- **Claude Code Bot** (main.py) - Add semantic context to queries
- **CI/CD Pipelines** - Automated documentation generation
- **Code Review Tools** - Find similar code patterns
- **Documentation Systems** - Semantic code search

## 📝 License

Same as parent project license. 