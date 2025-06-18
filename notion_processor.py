"""
Notion Content Processor

This module handles reading and processing content from Notion pages.
"""

import asyncio
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
from urllib.parse import urlparse

from notion_client import AsyncClient
from notion_client.errors import APIResponseError

logger = logging.getLogger(__name__)


class NotionProcessor:
    """Process content from Notion pages."""
    
    def __init__(self, notion_token: str):
        # Initialize Notion client with explicit parameters
        self.client = AsyncClient(auth=notion_token)
    
    def extract_page_id_from_url(self, notion_url: str) -> str:
        """
        Extract page ID from a Notion URL.
        
        Supports various Notion URL formats:
        - https://www.notion.so/Page-Title-1234567890abcdef1234567890abcdef
        - https://notion.so/workspace/Page-Title-1234567890abcdef1234567890abcdef
        - https://www.notion.so/1234567890abcdef1234567890abcdef
        - Page IDs with or without dashes
        """
        try:
            # If it's already a clean page ID (with or without dashes), return it formatted
            if self._is_valid_page_id(notion_url):
                return self._format_page_id(notion_url)
            
            # Parse the URL
            parsed_url = urlparse(notion_url)
            
            # Extract the path and remove leading slash
            path = parsed_url.path.lstrip('/')
            
            # Remove query parameters and fragments
            if '?' in path:
                path = path.split('?')[0]
            if '#' in path:
                path = path.split('#')[0]
            
            # Pattern to match Notion page IDs (32 hex characters)
            # They can be at the end of the path or followed by query params
            page_id_pattern = r'([a-f0-9]{32})'
            
            # Look for page ID in the path
            matches = re.findall(page_id_pattern, path, re.IGNORECASE)
            
            if matches:
                # Take the last match (most likely to be the page ID)
                page_id = matches[-1]
                return self._format_page_id(page_id)
            
            # Alternative pattern: look for page ID with dashes
            dashed_pattern = r'([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})'
            dashed_matches = re.findall(dashed_pattern, path, re.IGNORECASE)
            
            if dashed_matches:
                return dashed_matches[-1]
            
            raise ValueError(f"Could not extract page ID from URL: {notion_url}")
            
        except Exception as e:
            logger.error(f"Error extracting page ID from URL '{notion_url}': {e}")
            raise ValueError(f"Invalid Notion URL format: {notion_url}")
    
    def _is_valid_page_id(self, text: str) -> bool:
        """Check if text is a valid Notion page ID."""
        # Remove any dashes
        clean_text = text.replace('-', '')
        
        # Check if it's 32 hex characters
        return len(clean_text) == 32 and re.match(r'^[a-f0-9]{32}$', clean_text, re.IGNORECASE) is not None
    
    def _format_page_id(self, page_id: str) -> str:
        """Format page ID with proper UUID dashes."""
        # Remove any existing dashes
        clean_id = page_id.replace('-', '')
        
        # Add dashes in UUID format: 8-4-4-4-12
        if len(clean_id) == 32:
            return f"{clean_id[:8]}-{clean_id[8:12]}-{clean_id[12:16]}-{clean_id[16:20]}-{clean_id[20:]}"
        
        raise ValueError(f"Invalid page ID length: {page_id}")
    
    async def get_page_content_from_url(self, notion_url: str) -> Dict[str, Any]:
        """Get content from a Notion page using its URL."""
        page_id = self.extract_page_id_from_url(notion_url)
        return await self.get_page_content(page_id)
    
    async def get_page_content(self, page_id: str) -> Dict[str, Any]:
        """Get content from a Notion page."""
        try:
            # Get page metadata
            page = await self.client.pages.retrieve(page_id=page_id)
            
            # Get page content blocks
            blocks = await self._get_all_blocks(page_id)
            
            # Extract and structure content
            content = self._extract_content_from_blocks(blocks)
            
            return {
                'page_id': page_id,
                'title': self._extract_page_title(page),
                'created_time': page.get('created_time'),
                'last_edited_time': page.get('last_edited_time'),
                'content': content,
                'raw_blocks': blocks,
                'properties': page.get('properties', {})
            }
            
        except APIResponseError as e:
            logger.error(f"Notion API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error retrieving Notion page: {e}")
            raise
    
    async def _get_all_blocks(self, page_id: str) -> List[Dict[str, Any]]:
        """Get all blocks from a page, handling pagination."""
        all_blocks = []
        start_cursor = None
        
        while True:
            try:
                response = await self.client.blocks.children.list(
                    block_id=page_id,
                    start_cursor=start_cursor,
                    page_size=100
                )
                
                all_blocks.extend(response['results'])
                
                if not response.get('has_more', False):
                    break
                    
                start_cursor = response.get('next_cursor')
                
            except APIResponseError as e:
                logger.error(f"Error fetching blocks: {e}")
                break
        
        return all_blocks
    
    def _extract_page_title(self, page: Dict[str, Any]) -> str:
        """Extract page title from page object."""
        properties = page.get('properties', {})
        
        # Try to find title property
        for prop_name, prop_data in properties.items():
            if prop_data.get('type') == 'title':
                title_array = prop_data.get('title', [])
                if title_array:
                    return ''.join([t.get('plain_text', '') for t in title_array])
        
        return 'Untitled'
    
    def _extract_content_from_blocks(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract structured content from Notion blocks."""
        content = {
            'text_content': [],
            'headings': [],
            'lists': [],
            'code_blocks': [],
            'tables': [],
            'callouts': [],
            'todos': []
        }
        
        for block in blocks:
            block_type = block.get('type', '')
            block_content = self._process_block(block, block_type)
            
            if block_content:
                # Categorize content by type
                if block_type in ['paragraph', 'quote']:
                    content['text_content'].append(block_content)
                elif block_type.startswith('heading_'):
                    content['headings'].append(block_content)
                elif block_type in ['bulleted_list_item', 'numbered_list_item']:
                    content['lists'].append(block_content)
                elif block_type == 'code':
                    content['code_blocks'].append(block_content)
                elif block_type == 'table':
                    content['tables'].append(block_content)
                elif block_type == 'callout':
                    content['callouts'].append(block_content)
                elif block_type == 'to_do':
                    content['todos'].append(block_content)
        
        return content
    
    def _process_block(self, block: Dict[str, Any], block_type: str) -> Optional[Dict[str, Any]]:
        """Process individual block based on its type."""
        if not block_type or block_type not in block:
            return None
        
        block_data = block[block_type]
        
        try:
            if block_type == 'paragraph':
                return {
                    'type': 'paragraph',
                    'text': self._extract_rich_text(block_data.get('rich_text', [])),
                    'id': block.get('id')
                }
            
            elif block_type.startswith('heading_'):
                level = int(block_type.split('_')[1])
                return {
                    'type': f'heading_{level}',
                    'level': level,
                    'text': self._extract_rich_text(block_data.get('rich_text', [])),
                    'id': block.get('id')
                }
            
            elif block_type in ['bulleted_list_item', 'numbered_list_item']:
                return {
                    'type': block_type,
                    'text': self._extract_rich_text(block_data.get('rich_text', [])),
                    'id': block.get('id')
                }
            
            elif block_type == 'code':
                return {
                    'type': 'code',
                    'language': block_data.get('language', 'plain'),
                    'text': self._extract_rich_text(block_data.get('rich_text', [])),
                    'id': block.get('id')
                }
            
            elif block_type == 'callout':
                return {
                    'type': 'callout',
                    'icon': block_data.get('icon', {}),
                    'text': self._extract_rich_text(block_data.get('rich_text', [])),
                    'id': block.get('id')
                }
            
            elif block_type == 'to_do':
                return {
                    'type': 'todo',
                    'checked': block_data.get('checked', False),
                    'text': self._extract_rich_text(block_data.get('rich_text', [])),
                    'id': block.get('id')
                }
            
            elif block_type == 'quote':
                return {
                    'type': 'quote',
                    'text': self._extract_rich_text(block_data.get('rich_text', [])),
                    'id': block.get('id')
                }
            
        except Exception as e:
            logger.warning(f"Error processing block {block.get('id')}: {e}")
            return None
        
        return None
    
    def _extract_rich_text(self, rich_text_array: List[Dict[str, Any]]) -> str:
        """Extract plain text from Notion rich text array."""
        if not rich_text_array:
            return ""
        
        return ''.join([
            text_obj.get('plain_text', '') 
            for text_obj in rich_text_array
        ])
    
    def get_content_summary(self, content: Dict[str, Any]) -> str:
        """Generate a summary of the page content."""
        summary_parts = []
        
        # Add title
        if content.get('title'):
            summary_parts.append(f"Title: {content['title']}")
        
        # Add headings
        headings = content.get('content', {}).get('headings', [])
        if headings:
            summary_parts.append("Headings:")
            for heading in headings[:5]:  # Limit to first 5 headings
                summary_parts.append(f"  - {heading.get('text', '')}")
        
        # Add text content preview
        text_content = content.get('content', {}).get('text_content', [])
        if text_content:
            summary_parts.append("Content Preview:")
            for i, paragraph in enumerate(text_content[:3]):  # First 3 paragraphs
                text = paragraph.get('text', '')[:100]  # First 100 chars
                if text:
                    summary_parts.append(f"  {i+1}. {text}...")
        
        # Add code blocks if any
        code_blocks = content.get('content', {}).get('code_blocks', [])
        if code_blocks:
            summary_parts.append(f"Code blocks: {len(code_blocks)} found")
        
        # Add todos if any
        todos = content.get('content', {}).get('todos', [])
        if todos:
            completed = sum(1 for todo in todos if todo.get('checked'))
            summary_parts.append(f"Tasks: {len(todos)} total, {completed} completed")
        
        return '\n'.join(summary_parts) 