# 🚀 Notion + Claude Code Integration

A powerful application that reads content from Notion pages, processes it with Claude AI, and uses Claude Code to intelligently analyze and query your codebase to provide contextual insights and answers.

## ✨ Features

- 📖 **Notion Integration**: Read and process content from any Notion page
- 🤖 **Claude AI Analysis**: Intelligent content analysis and summarization
- 🔍 **Claude Code Querying**: Advanced codebase analysis and querying with Claude Code
- 🔗 **Context-Aware Responses**: Combine Notion content with code insights
- 📊 **Multi-Language Support**: Python, JavaScript, TypeScript, and more
- 🎯 **Interactive CLI**: Easy-to-use command-line interface

## 🛠️ Setup

### Prerequisites

- Python 3.10+ (required for Claude Code SDK)
- Node.js (required for Claude Code CLI)
- Notion API integration token
- Anthropic API key for Claude

### Installation

1. **Install Node.js** (required):
   Download and install from [https://nodejs.org/](https://nodejs.org/)

2. **Install Claude Code CLI** (required):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

3. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd tech-bros
   ```

4. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` with your API keys and configuration:
   ```env
   NOTION_TOKEN=your_notion_integration_token_here
   NOTION_PAGE_URL=https://www.notion.so/your-page-url
   # OR use NOTION_PAGE_ID for backward compatibility
   # NOTION_PAGE_ID=your_notion_page_id_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   CODEBASE_PATH=./your_codebase_path
   ```

## 🔑 Getting API Keys

### Notion API Token

1. Go to [Notion Integrations](https://www.notion.so/my-integrations)
2. Click "New integration"
3. Give it a name and workspace
4. Copy the "Internal Integration Token"
5. Share your Notion page with the integration

### Claude API Key

1. Sign up at [Anthropic Console](https://console.anthropic.com)
2. Generate an API key
3. Add it to your `.env` file

### Using Notion Page URLs

You can now use Notion page URLs directly! The application will automatically extract the page ID from URLs like:

```
https://www.notion.so/workspace/Page-Title-abc123def456
https://notion.so/Page-Title-abc123def456
https://www.notion.so/abc123def456
```

**Backward Compatibility**: You can still use page IDs directly if preferred. The page ID from the URL above would be: `abc123def456`

## 🚀 Usage

### Basic Usage

**Recommended**: Use the startup script (checks environment automatically):
```bash
python start.py
```

**Alternative**: Run the main application directly:
```bash
python main.py
```

### Available Commands

- **Query the codebase**: Ask questions about your code using Claude Code
- **`summary`**: Get an overview of your codebase structure
- **`reload`**: Refresh Notion page content from the default URL/ID
- **`reload <URL>`**: Load content from a specific Notion page URL
- **`quit`**: Exit the application

### Example Queries

- "How is authentication handled in this codebase?"
- "Show me all the API endpoints"
- "What testing frameworks are being used?"
- "Find functions related to user management"
- "How are errors handled across the application?"

## 📁 Project Structure

```
tech-bros/
├── start.py                # Startup script with environment checks
├── main.py                 # Main application entry point
├── notion_processor.py     # Notion API integration
├── claude_integration.py   # Claude AI processing
├── claude_code_client.py   # Claude Code integration for codebase analysis
├── codebase_analyzer.py    # Code analysis utilities
├── requirements.txt        # Python dependencies
├── setup.py               # Installation script
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `NOTION_TOKEN` | Notion integration token | Yes |
| `ANTHROPIC_API_KEY` | Claude API key | Yes |
| `NOTION_PAGE_URL` | Default Notion page URL | No |
| `NOTION_PAGE_ID` | Default Notion page ID (legacy) | No |
| `CODEBASE_PATH` | Path to your codebase | No (default: `./`) |
| `LOG_LEVEL` | Logging level | No (default: `INFO`) |

## 🎯 How It Works

1. **Notion Content Processing**:
   - Connects to Notion API
   - Extracts page content (text, headings, code blocks, todos)
   - Structures content for analysis

2. **Claude Analysis**:
   - Sends content to Claude for intelligent processing
   - Extracts key points, technical concepts, and action items
   - Generates structured summaries

3. **Claude Code SDK Integration**:
   - Uses the official Claude Code SDK for codebase analysis
   - Provides intelligent code understanding and navigation
   - Supports multiple programming languages and frameworks
   - Leverages Claude Code's subprocess for advanced analysis

4. **Intelligent Querying**:
   - Combines Notion context with user queries
   - Uses Claude Code SDK for advanced codebase analysis
   - Returns contextually relevant answers with code examples

## 🔍 Supported File Types

- **Programming Languages**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more
- **Configuration**: JSON, YAML, TOML, INI files
- **Documentation**: Markdown, reStructuredText, plain text
- **Web Technologies**: HTML, CSS, SCSS

## 🚨 Troubleshooting

### Common Issues

1. **Notion API Errors**:
   - Ensure your integration has access to the page
   - Check that the page ID is correct
   - Verify the integration token

2. **Claude API Errors**:
   - Check your API key is valid
   - Ensure you have sufficient credits
   - Monitor rate limits

3. **Claude Code SDK Issues**:
   - Ensure Node.js is installed and accessible
   - Verify Claude Code CLI is installed: `claude --version`
   - Check that the codebase path exists and is accessible
   - Ensure you have the required Python version (3.10+)

### Debug Mode

Set `LOG_LEVEL=DEBUG` in your `.env` file for detailed logging.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

If you encounter any issues or have questions:

1. Check the troubleshooting section
2. Review the logs with debug mode enabled
3. Open an issue on GitHub with details about your setup and the problem

## 🔮 Future Enhancements

- Support for multiple Notion pages
- Web interface for easier interaction
- Integration with more AI models
- Advanced code analysis features
- Real-time Notion page monitoring
- Export functionality for analysis results # tech-bros-project
# tech-bros-project
