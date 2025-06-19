# Simple Claude Code Bot

A simplified tool to query codebases using Claude AI with optional Notion page context.

## Setup

1. Install Claude CLI:
   ```bash
   npm install -g claude-cli
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```bash
   ANTHROPIC_API_KEY=your_api_key_here
   CODEBASE_PATH=./
   
   # Optional: For Notion integration
   NOTION_TOKEN=your_notion_integration_token
   NOTION_PAGE_ID=your_notion_page_id
   ```

## Usage

Run the bot:
```bash
python main.py
```

## Commands

- **Ask questions**: Type any question about your codebase
- **`analyze`** - Get a detailed codebase overview  
- **`notion`** - Load content from your default Notion page
- **`notion <page_id>`** - Load content from a specific Notion page
- **`context`** - Show current Notion context
- **`quit`** - Exit the application

## How it works

1. **Without Notion**: Claude analyzes your codebase directly
2. **With Notion**: Claude gets additional context from your Notion page and uses both the page content and codebase to answer questions

## Example Questions

- "What is this codebase about?"
- "How does the authentication work?"
- "Show me the main entry points"
- "What dependencies does this project use?"
- "Based on the Notion requirements, what changes need to be made?"

## Notion Setup (Optional)

1. Go to [Notion Integrations](https://www.notion.so/my-integrations)
2. Create a new integration and copy the token
3. Share your Notion page with the integration
4. Add the token and page ID to your `.env` file

That's it! Simple, clean, and functional. 🎉
