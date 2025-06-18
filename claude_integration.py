"""
Claude Integration Module

This module handles integration with Claude API for content processing and analysis.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import json

from anthropic import AsyncAnthropic

logger = logging.getLogger(__name__)


class ClaudeIntegration:
    """Integration with Claude API for content processing."""
    
    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.client = AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_tokens = 4096
    
    async def process_notion_content(self, notion_content: Dict[str, Any]) -> Dict[str, Any]:
        """Process Notion content and extract key information using Claude."""
        try:
            # Prepare content for Claude
            content_text = self._format_notion_content_for_claude(notion_content)
            
            # Create prompt for content analysis
            prompt = self._create_content_analysis_prompt(content_text)
            
            # Call Claude API
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.1,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Parse Claude's response
            claude_response = response.content[0].text
            processed_content = self._parse_claude_analysis(claude_response)
            
            return {
                'original_content': notion_content,
                'claude_analysis': claude_response,
                'summary': processed_content.get('summary', ''),
                'key_points': processed_content.get('key_points', []),
                'action_items': processed_content.get('action_items', []),
                'questions': processed_content.get('questions', []),
                'technical_concepts': processed_content.get('technical_concepts', []),
                'code_related_items': processed_content.get('code_related_items', [])
            }
            
        except Exception as e:
            logger.error(f"Error processing content with Claude: {e}")
            raise
    
    def _format_notion_content_for_claude(self, notion_content: Dict[str, Any]) -> str:
        """Format Notion content for Claude analysis."""
        formatted_parts = []
        
        # Add title
        if notion_content.get('title'):
            formatted_parts.append(f"TITLE: {notion_content['title']}")
        
        # Add metadata
        if notion_content.get('created_time'):
            formatted_parts.append(f"CREATED: {notion_content['created_time']}")
        
        # Process content sections
        content = notion_content.get('content', {})
        
        # Add headings
        headings = content.get('headings', [])
        if headings:
            formatted_parts.append("\nHEADINGS:")
            for heading in headings:
                level = heading.get('level', 1)
                indent = "  " * (level - 1)
                formatted_parts.append(f"{indent}- {heading.get('text', '')}")
        
        # Add text content
        text_content = content.get('text_content', [])
        if text_content:
            formatted_parts.append("\nCONTENT:")
            for paragraph in text_content:
                text = paragraph.get('text', '').strip()
                if text:
                    formatted_parts.append(f"  {text}")
        
        # Add lists
        lists = content.get('lists', [])
        if lists:
            formatted_parts.append("\nLISTS:")
            for item in lists:
                formatted_parts.append(f"  • {item.get('text', '')}")
        
        # Add code blocks
        code_blocks = content.get('code_blocks', [])
        if code_blocks:
            formatted_parts.append("\nCODE BLOCKS:")
            for code in code_blocks:
                lang = code.get('language', 'plain')
                text = code.get('text', '')
                formatted_parts.append(f"  Language: {lang}")
                formatted_parts.append(f"  Code: {text}")
        
        # Add callouts
        callouts = content.get('callouts', [])
        if callouts:
            formatted_parts.append("\nCALLOUTS:")
            for callout in callouts:
                formatted_parts.append(f"  📝 {callout.get('text', '')}")
        
        # Add todos
        todos = content.get('todos', [])
        if todos:
            formatted_parts.append("\nTODOS:")
            for todo in todos:
                status = "✅" if todo.get('checked') else "☐"
                formatted_parts.append(f"  {status} {todo.get('text', '')}")
        
        return '\n'.join(formatted_parts)
    
    def _create_content_analysis_prompt(self, content_text: str) -> str:
        """Create a prompt for Claude to analyze the content."""
        return f"""
Please analyze the following Notion page content and provide a structured analysis. 
Return your response in JSON format with the following structure:

{{
    "summary": "A concise summary of the main content",
    "key_points": ["List of key points or main ideas"],
    "action_items": ["Any tasks, todos, or action items mentioned"],
    "questions": ["Any questions that arise from the content"],
    "technical_concepts": ["Any technical concepts, frameworks, or tools mentioned"],
    "code_related_items": ["Any programming languages, code snippets, or development-related items"]
}}

Content to analyze:
{content_text}

Please provide a thorough but concise analysis focusing on:
1. Main themes and ideas
2. Technical content and code-related information
3. Actionable items and questions
4. Concepts that might be relevant for codebase queries

Response (JSON format only):
"""
    
    def _parse_claude_analysis(self, claude_response: str) -> Dict[str, Any]:
        """Parse Claude's analysis response."""
        try:
            # Try to extract JSON from the response
            response = claude_response.strip()
            
            # Find JSON content (look for opening and closing braces)
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            
            # If JSON parsing fails, create a basic structure
            return {
                'summary': claude_response[:500] + '...' if len(claude_response) > 500 else claude_response,
                'key_points': [],
                'action_items': [],
                'questions': [],
                'technical_concepts': [],
                'code_related_items': []
            }
            
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse Claude response as JSON: {e}")
            return {
                'summary': claude_response,
                'key_points': [],
                'action_items': [],
                'questions': [],
                'technical_concepts': [],
                'code_related_items': []
            }
        except Exception as e:
            logger.error(f"Error parsing Claude analysis: {e}")
            return {
                'summary': 'Error parsing analysis',
                'key_points': [],
                'action_items': [],
                'questions': [],
                'technical_concepts': [],
                'code_related_items': []
            }
    
    async def generate_codebase_query(self, notion_context: Dict[str, Any], user_query: str) -> str:
        """Generate an enhanced query for codebase search based on Notion context."""
        try:
            prompt = f"""
Based on the following Notion page content and user query, generate an enhanced search query 
that would be most effective for searching a codebase.

Notion Content Summary:
{notion_context.get('summary', '')}

Key Technical Concepts:
{', '.join(notion_context.get('technical_concepts', []))}

Code-Related Items:
{', '.join(notion_context.get('code_related_items', []))}

User Query: {user_query}

Please generate an enhanced query that:
1. Incorporates relevant technical concepts from the Notion content
2. Uses appropriate programming terminology
3. Focuses on the most relevant aspects for code search
4. Maintains the intent of the original user query

Enhanced Query:
"""
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.2,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            enhanced_query = response.content[0].text.strip()
            
            # Clean up the response to extract just the query
            if "Enhanced Query:" in enhanced_query:
                enhanced_query = enhanced_query.split("Enhanced Query:")[-1].strip()
            
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error generating enhanced query: {e}")
            return user_query  # Fall back to original query
    
    async def analyze_code_context(self, code_results: str, notion_context: Dict[str, Any]) -> str:
        """Analyze code search results in the context of Notion content."""
        try:
            prompt = f"""
Based on the Notion page content and the code search results below, provide a comprehensive analysis 
that connects the information from both sources.

Notion Context:
Summary: {notion_context.get('summary', '')}
Key Points: {', '.join(notion_context.get('key_points', [])[:5])}
Technical Concepts: {', '.join(notion_context.get('technical_concepts', []))}

Code Search Results:
{code_results}

Please provide an analysis that:
1. Explains how the code relates to the Notion content
2. Identifies relevant patterns or implementations
3. Suggests potential improvements or considerations
4. Highlights any gaps or questions that arise
5. Provides actionable insights

Analysis:
"""
            
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=0.3,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Error analyzing code context: {e}")
            return code_results  # Fall back to original results 