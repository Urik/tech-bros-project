#!/usr/bin/env python3
"""
Notion + Claude Code Integration

This application reads from Notion pages, processes the content with Claude,
and queries a codebase using Claude Code to provide intelligent responses.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import nest_asyncio
from dotenv import load_dotenv

from codebase_analyzer import CodebaseAnalyzer
from notion_processor import NotionProcessor
from claude_integration import ClaudeIntegration
from claude_code_client import ClaudeCodeClient

# Enable nested asyncio for Jupyter compatibility
nest_asyncio.apply()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NotionClaudeCodeBot:
    """Main application class for Notion + Claude Code integration."""
    
    def __init__(self):
        self.notion_token = os.getenv('NOTION_TOKEN')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        self.notion_page_url = os.getenv('NOTION_PAGE_URL')
        self.notion_page_id = os.getenv('NOTION_PAGE_ID')  # Fallback for backward compatibility
        self.codebase_path = Path(os.getenv('CODEBASE_PATH', './'))
        
        # Validate required environment variables
        self._validate_env_vars()
        
        # Initialize components
        self.notion_processor = NotionProcessor(self.notion_token)
        self.claude_integration = ClaudeIntegration(self.anthropic_api_key)
        self.codebase_analyzer = CodebaseAnalyzer(self.codebase_path)
        self.claude_code_client = ClaudeCodeClient(self.anthropic_api_key, self.codebase_path)
        
    def _validate_env_vars(self):
        """Validate required environment variables."""
        required_vars = ['NOTION_TOKEN', 'ANTHROPIC_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    async def initialize_codebase_analysis(self) -> Dict[str, Any]:
        """Initialize codebase analysis using Claude Code."""
        logger.info(f"Analyzing codebase structure at {self.codebase_path}")
        
        try:
            # Get codebase structure analysis
            structure = await self.claude_code_client.analyze_codebase_structure()
            logger.info("Codebase analysis completed successfully")
            return structure
            
        except Exception as e:
            logger.error(f"Error analyzing codebase: {e}")
            raise
    
    async def process_notion_content(self, page_url_or_id: Optional[str] = None) -> Dict[str, Any]:
        """Process content from a Notion page using URL or page ID."""
        # Determine what to use as the target
        target = page_url_or_id or self.notion_page_url or self.notion_page_id
        
        if not target:
            raise ValueError("No Notion page URL or ID provided")
        
        logger.info(f"Processing Notion page: {target}")
        
        try:
            # Check if it's a URL or page ID and get content accordingly
            if target.startswith('http'):
                # It's a URL
                notion_content = await self.notion_processor.get_page_content_from_url(target)
                page_id = self.notion_processor.extract_page_id_from_url(target)
            else:
                # It's a page ID
                page_id = target
                # Format the page ID if needed
                if self.notion_processor._is_valid_page_id(target):
                    page_id = self.notion_processor._format_page_id(target)
                notion_content = await self.notion_processor.get_page_content(page_id)
            
            # Process with Claude
            processed_content = await self.claude_integration.process_notion_content(notion_content)
            
            return {
                'raw_content': notion_content,
                'processed_content': processed_content,
                'page_id': page_id,
                'source': target
            }
            
        except Exception as e:
            logger.error(f"Error processing Notion content: {e}")
            raise
    
    async def query_codebase(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Query the codebase using Claude Code with optional Notion context."""
        logger.info(f"Querying codebase with Claude Code: {query}")
        
        try:
            # Use Claude Code to query the codebase
            response = await self.claude_code_client.query_codebase(query, context)
            return response
            
        except Exception as e:
            logger.error(f"Error querying codebase with Claude Code: {e}")
            raise
    
    async def get_codebase_summary(self) -> str:
        """Get a summary of the codebase structure and overview."""
        try:
            structure = await self.initialize_codebase_analysis()
            
            summary_parts = [
                f"📁 Codebase Analysis: {self.codebase_path.name}",
                f"📄 Overview: {structure.get('overview', 'N/A')}",
                ""
            ]
            
            # Languages
            languages = structure.get('main_languages', [])
            if languages:
                summary_parts.append("🔤 Main Languages:")
                for lang in languages:
                    summary_parts.append(f"  • {lang}")
                summary_parts.append("")
            
            # Architecture patterns
            patterns = structure.get('architecture_patterns', [])
            if patterns:
                summary_parts.append("🏗️ Architecture Patterns:")
                for pattern in patterns:
                    summary_parts.append(f"  • {pattern}")
                summary_parts.append("")
            
            # Key directories
            directories = structure.get('key_directories', [])
            if directories:
                summary_parts.append("📂 Key Directories:")
                for dir_info in directories:
                    name = dir_info.get('name', 'Unknown') if isinstance(dir_info, dict) else str(dir_info)
                    purpose = dir_info.get('purpose', '') if isinstance(dir_info, dict) else ''
                    if purpose:
                        summary_parts.append(f"  • {name}/ - {purpose}")
                    else:
                        summary_parts.append(f"  • {name}/")
            
            return '\n'.join(summary_parts)
            
        except Exception as e:
            logger.error(f"Error getting codebase summary: {e}")
            return f"Error analyzing codebase: {e}"
    
    async def run_interactive_session(self):
        """Run an interactive session for querying."""
        print("🚀 Notion + Claude Code Integration Started!")
        print("Available commands:")
        print("  • 'quit' - Exit the application")
        print("  • 'reload' - Reload Notion content")
        print("  • 'reload <URL>' - Load content from a specific Notion page URL")
        print("  • 'summary' - Get codebase summary")
        print("  • 'analyze [query]' - Query the codebase")
        print("-" * 50)
        
        # Process initial Notion content
        notion_context = None
        try:
            notion_context = await self.process_notion_content()
            print(f"✅ Loaded Notion page content from: {notion_context.get('source', 'unknown')}")
        except Exception as e:
            print(f"⚠️  Could not load Notion content: {e}")
            print("💡 You can load a specific page using: reload <notion-page-url>")
        
        # Show initial codebase summary
        print("\n📊 Getting codebase overview...")
        try:
            summary = await self.get_codebase_summary()
            print(f"\n{summary}")
        except Exception as e:
            print(f"⚠️  Could not analyze codebase: {e}")
        
        while True:
            try:
                user_input = input("\n🤔 Enter your query or command: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower().startswith('reload'):
                    parts = user_input.split(None, 1)
                    if len(parts) > 1:
                        # Reload with specific URL
                        notion_url = parts[1].strip()
                        print(f"🔄 Loading Notion content from: {notion_url}")
                        notion_context = await self.process_notion_content(notion_url)
                        print(f"✅ Notion content loaded from: {notion_context.get('source', 'unknown')}")
                    else:
                        # Reload default
                        print("🔄 Reloading Notion content...")
                        notion_context = await self.process_notion_content()
                        print(f"✅ Notion content reloaded from: {notion_context.get('source', 'unknown')}")
                    continue
                
                if user_input.lower() == 'summary':
                    print("📊 Getting codebase summary...")
                    summary = await self.get_codebase_summary()
                    print(f"\n{summary}")
                    continue
                
                if not user_input:
                    continue
                
                # Query codebase with Claude Code
                print("🔍 Querying codebase with Claude Code...")
                response = await self.query_codebase(user_input, notion_context)
                
                print(f"\n💡 Response:\n{response}")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


async def main():
    """Main entry point."""
    try:
        bot = NotionClaudeCodeBot()
        await bot.run_interactive_session()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Error starting application: {e}")


if __name__ == "__main__":
    asyncio.run(main()) 