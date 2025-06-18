"""
Claude Code Client

This module handles integration with Claude Code SDK for codebase analysis and querying.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import os

from claude_code_sdk import query, ClaudeCodeOptions, Message

logger = logging.getLogger(__name__)


class ClaudeCodeClient:
    """Client for interacting with Claude Code SDK for codebase analysis."""
    
    def __init__(self, api_key: str, codebase_path: Path):
        # Set the API key in environment if not already set
        if not os.getenv('ANTHROPIC_API_KEY'):
            os.environ['ANTHROPIC_API_KEY'] = api_key
        
        self.codebase_path = Path(codebase_path)
        self.default_options = ClaudeCodeOptions(
            max_turns=5,
            cwd=self.codebase_path,
            # permission_mode="acceptEdits"
        )
    
    async def query_codebase(self, user_query: str, notion_context: Optional[Dict[str, Any]] = None) -> str:
        """Query the codebase using Claude Code SDK with optional Notion context."""
        try:
            # Enhance query with Notion context if provided
            enhanced_query = self._enhance_query_with_context(user_query, notion_context)
            
            logger.info(f"Querying Claude Code SDK: {enhanced_query[:100]}...")
            
            # Collect all messages from the Claude Code query
            messages: List[Message] = []
            result_text = ""
            
            async for message in query(
                prompt=enhanced_query,
                options=self.default_options
            ):
                messages.append(message)
                
                # Handle different message types
                if hasattr(message, 'content') and message.content:
                    # Direct content attribute
                    if isinstance(message.content, str):
                        result_text += message.content + "\n"
                    elif isinstance(message.content, list):
                        for item in message.content:
                            if hasattr(item, 'text'):
                                result_text += item.text + "\n"
                            else:
                                result_text += str(item) + "\n"
                elif hasattr(message, 'message') and hasattr(message.message, 'content'):
                    # Nested message content
                    content = message.message.content
                    if isinstance(content, str):
                        result_text += content + "\n"
                    elif isinstance(content, list):
                        for item in content:
                            if hasattr(item, 'text'):
                                result_text += item.text + "\n"
                            else:
                                result_text += str(item) + "\n"
                elif hasattr(message, 'result'):
                    # Result message
                    result_text += str(message.result) + "\n"
            
            return result_text if result_text else "No response received from Claude Code"
            
        except Exception as e:
            logger.error(f"Error querying codebase with Claude Code SDK: {e}")
            raise
    
    def _enhance_query_with_context(self, user_query: str, notion_context: Optional[Dict[str, Any]]) -> str:
        """Enhance the query with context from Notion content."""
        if not notion_context:
            return user_query
        
        processed_content = notion_context.get('processed_content', {})
        
        context_parts = []
        
        # Add summary
        summary = processed_content.get('summary', '')
        if summary:
            context_parts.append(f"Context Summary: {summary}")
        
        # Add key points
        key_points = processed_content.get('key_points', [])
        if key_points:
            context_parts.append("Key Points:")
            for point in key_points[:5]:  # Limit to top 5 points
                context_parts.append(f"  • {point}")
        
        # Add technical concepts
        technical_concepts = processed_content.get('technical_concepts', [])
        if technical_concepts:
            context_parts.append(f"Technical Concepts: {', '.join(technical_concepts[:10])}")
        
        # Add code-related items
        code_related = processed_content.get('code_related_items', [])
        if code_related:
            context_parts.append(f"Code-Related Items: {', '.join(code_related[:10])}")
        
        if context_parts:
            enhanced_query = f"""
Context from Notion page:
{chr(10).join(context_parts)}

User Query: {user_query}

Please analyze the codebase and answer the query considering both the Notion context above and the actual code in this project.
"""
        else:
            enhanced_query = user_query
        
        return enhanced_query
    
    async def analyze_codebase_structure(self) -> Dict[str, Any]:
        """Get a high-level analysis of the codebase structure using Claude Code SDK."""
        try:
            prompt = """
Analyze this codebase and provide a comprehensive overview. Please examine the files and structure, then provide analysis in the following JSON format:

{
    "overview": "Brief description of what this codebase appears to be",
    "main_languages": ["list", "of", "primary", "programming", "languages"],
    "architecture_patterns": ["patterns", "or", "frameworks", "used"],
    "key_directories": [
        {"name": "dir_name", "purpose": "what this directory contains"},
    ],
    "entry_points": ["main files or entry points"],
    "dependencies": ["key dependencies or frameworks"],
    "testing_approach": "testing strategy used",
    "documentation": "state of documentation",
    "potential_improvements": ["suggested", "improvements"]
}

Please examine the actual files and provide accurate, specific information.
"""
            
            logger.info("Analyzing codebase structure with Claude Code SDK...")
            
            # Collect all messages from the Claude Code query
            messages: List[Message] = []
            response_text = ""
            
            async for message in query(
                prompt=prompt,
                options=self.default_options
            ):
                messages.append(message)
                
                # Handle different message types
                if hasattr(message, 'content') and message.content:
                    # Direct content attribute
                    if isinstance(message.content, str):
                        response_text += message.content + "\n"
                    elif isinstance(message.content, list):
                        for item in message.content:
                            if hasattr(item, 'text'):
                                response_text += item.text + "\n"
                            else:
                                response_text += str(item) + "\n"
                elif hasattr(message, 'message') and hasattr(message.message, 'content'):
                    # Nested message content
                    content = message.message.content
                    if isinstance(content, str):
                        response_text += content + "\n"
                    elif isinstance(content, list):
                        for item in content:
                            if hasattr(item, 'text'):
                                response_text += item.text + "\n"
                            else:
                                response_text += str(item) + "\n"
                elif hasattr(message, 'result'):
                    # Result message
                    response_text += str(message.result) + "\n"
            
            # Try to parse JSON response
            try:
                # Extract JSON from response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    return json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("Could not parse codebase analysis as JSON")
            
            # Fallback to text response
            return {
                "overview": response_text if response_text else "Analysis failed",
                "main_languages": [],
                "architecture_patterns": [],
                "key_directories": [],
                "entry_points": [],
                "dependencies": [],
                "testing_approach": "Unknown",
                "documentation": "Unknown",
                "potential_improvements": []
            }
            
        except Exception as e:
            logger.error(f"Error analyzing codebase structure: {e}")
            raise
    
    async def explain_code_section(self, file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
        """Explain a specific section of code using Claude Code SDK."""
        try:
            location_info = f"file '{file_path}'"
            if start_line and end_line:
                location_info += f" (lines {start_line}-{end_line})"
            elif start_line:
                location_info += f" (starting from line {start_line})"
            
            prompt = f"""
Please examine and explain the code in {location_info}.

Provide:
1. What this code does (high-level purpose)
2. How it works (implementation details)
3. Key components or functions
4. Relationships to other parts of the codebase
5. Any notable patterns or design decisions
6. Potential improvements or concerns

Be specific and reference the actual code you're analyzing.
"""
            
            return await self.query_codebase(prompt)
            
        except Exception as e:
            logger.error(f"Error explaining code section: {e}")
            raise
    
    async def suggest_implementation(self, requirements: str, notion_context: Optional[Dict[str, Any]] = None) -> str:
        """Suggest implementation approach based on requirements and existing codebase."""
        try:
            enhanced_requirements = self._enhance_query_with_context(requirements, notion_context)
            
            prompt = f"""
Based on this codebase and the following requirements, suggest an implementation approach:

{enhanced_requirements}

Please provide:
1. Analysis of existing codebase patterns and architecture
2. Recommended implementation approach that fits the existing codebase
3. Specific files that should be modified or created
4. Code examples or patterns to follow
5. Potential challenges and how to address them
6. Testing strategy for the new implementation

Be specific and reference existing code patterns where applicable.
"""
            
            return await self.query_codebase(prompt, notion_context)
            
        except Exception as e:
            logger.error(f"Error suggesting implementation: {e}")
            raise 