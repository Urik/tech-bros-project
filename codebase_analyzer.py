"""
Codebase Analyzer Module

This module handles analysis and indexing of codebases for intelligent querying.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import fnmatch
import ast
import re

import git
from git.exc import InvalidGitRepositoryError

logger = logging.getLogger(__name__)


class CodebaseAnalyzer:
    """Analyze and process codebase for intelligent querying."""
    
    def __init__(self, codebase_path: Path):
        self.codebase_path = Path(codebase_path)
        self.supported_extensions = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.r': 'r',
            '.sql': 'sql',
            '.sh': 'shell',
            '.bash': 'shell',
            '.zsh': 'shell',
            '.fish': 'shell',
            '.ps1': 'powershell',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.md': 'markdown',
            '.rst': 'restructuredtext',
            '.txt': 'text',
            '.conf': 'config',
            '.cfg': 'config',
            '.ini': 'config',
            '.toml': 'toml',
            '.dockerfile': 'dockerfile',
            '.makefile': 'makefile'
        }
        
        # Common ignore patterns
        self.ignore_patterns = [
            '*.pyc', '*.pyo', '*.pyd', '__pycache__',
            '*.so', '*.dylib', '*.dll',
            '.git', '.svn', '.hg',
            'node_modules', 'bower_components',
            '.venv', 'venv', 'env',
            'dist', 'build', 'target',
            '*.log', '*.tmp', '*.temp',
            '.DS_Store', 'Thumbs.db',
            '*.min.js', '*.min.css',
            'package-lock.json', 'yarn.lock',
            'composer.lock', 'Pipfile.lock',
            '.pytest_cache', '.coverage',
            '*.egg-info', '*.whl'
        ]
        
        self.git_repo = None
        self._initialize_git_repo()
    
    def _initialize_git_repo(self):
        """Initialize Git repository if available."""
        try:
            self.git_repo = git.Repo(self.codebase_path, search_parent_directories=True)
            logger.info("Git repository detected")
        except InvalidGitRepositoryError:
            logger.info("No Git repository found")
    
    def should_ignore_file(self, file_path: Path) -> bool:
        """Check if a file should be ignored based on patterns."""
        file_name = file_path.name
        relative_path = str(file_path.relative_to(self.codebase_path))
        
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(file_name, pattern) or fnmatch.fnmatch(relative_path, pattern):
                return True
        
        return False
    
    def get_file_language(self, file_path: Path) -> Optional[str]:
        """Determine the programming language of a file."""
        suffix = file_path.suffix.lower()
        
        # Special cases
        if file_path.name.lower() in ['dockerfile', 'makefile']:
            return self.supported_extensions.get(f'.{file_path.name.lower()}')
        
        return self.supported_extensions.get(suffix)
    
    def analyze_file_structure(self) -> Dict[str, Any]:
        """Analyze the overall structure of the codebase."""
        structure = {
            'total_files': 0,
            'languages': {},
            'directories': [],
            'large_files': [],
            'recent_files': [],
            'file_types': {}
        }
        
        try:
            for file_path in self.codebase_path.rglob('*'):
                if file_path.is_file() and not self.should_ignore_file(file_path):
                    structure['total_files'] += 1
                    
                    # Language detection
                    language = self.get_file_language(file_path)
                    if language:
                        structure['languages'][language] = structure['languages'].get(language, 0) + 1
                    
                    # File type
                    extension = file_path.suffix.lower()
                    structure['file_types'][extension] = structure['file_types'].get(extension, 0) + 1
                    
                    # File size analysis
                    try:
                        file_size = file_path.stat().st_size
                        if file_size > 50000:  # Files larger than 50KB
                            structure['large_files'].append({
                                'path': str(file_path.relative_to(self.codebase_path)),
                                'size': file_size,
                                'language': language
                            })
                    except OSError:
                        pass
                
                elif file_path.is_dir() and not self.should_ignore_file(file_path):
                    structure['directories'].append(str(file_path.relative_to(self.codebase_path)))
            
            # Sort large files by size
            structure['large_files'].sort(key=lambda x: x['size'], reverse=True)
            structure['large_files'] = structure['large_files'][:20]  # Top 20
            
            # Git-based recent files
            if self.git_repo:
                structure['recent_files'] = self._get_recent_git_files()
            
        except Exception as e:
            logger.error(f"Error analyzing file structure: {e}")
        
        return structure
    
    def _get_recent_git_files(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recently modified files from Git history."""
        try:
            # Get commits from the last 30 days
            recent_commits = list(self.git_repo.iter_commits(max_count=100))
            
            file_modifications = {}
            
            for commit in recent_commits:
                try:
                    for item in commit.stats.files:
                        if item not in file_modifications:
                            file_modifications[item] = {
                                'path': item,
                                'last_commit': commit.hexsha[:8],
                                'last_author': commit.author.name,
                                'last_modified': commit.committed_datetime,
                                'message': commit.message.strip().split('\n')[0]
                            }
                except Exception:
                    continue
            
            # Sort by modification time and return recent files
            recent_files = sorted(
                file_modifications.values(),
                key=lambda x: x['last_modified'],
                reverse=True
            )
            
            return recent_files[:limit]
            
        except Exception as e:
            logger.error(f"Error getting recent Git files: {e}")
            return []
    
    def extract_python_symbols(self, file_path: Path) -> Dict[str, List[str]]:
        """Extract symbols (classes, functions, etc.) from Python files."""
        symbols = {
            'classes': [],
            'functions': [],
            'imports': [],
            'constants': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    symbols['classes'].append(node.name)
                elif isinstance(node, ast.FunctionDef):
                    symbols['functions'].append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        symbols['imports'].append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            symbols['imports'].append(f"{node.module}.{alias.name}")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            symbols['constants'].append(target.id)
            
        except Exception as e:
            logger.debug(f"Error parsing Python file {file_path}: {e}")
        
        return symbols
    
    def extract_javascript_symbols(self, file_path: Path) -> Dict[str, List[str]]:
        """Extract symbols from JavaScript/TypeScript files using regex."""
        symbols = {
            'functions': [],
            'classes': [],
            'exports': [],
            'imports': []
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Function patterns
            function_patterns = [
                r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)',
                r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\(',
                r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:\s*function',
                r'([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=>\s*',
            ]
            
            for pattern in function_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                symbols['functions'].extend(matches)
            
            # Class patterns
            class_matches = re.findall(r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', content)
            symbols['classes'].extend(class_matches)
            
            # Import patterns
            import_matches = re.findall(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', content)
            symbols['imports'].extend(import_matches)
            
            # Export patterns
            export_matches = re.findall(r'export\s+(?:default\s+)?(?:const\s+|function\s+|class\s+)?([a-zA-Z_$][a-zA-Z0-9_$]*)', content)
            symbols['exports'].extend(export_matches)
            
        except Exception as e:
            logger.debug(f"Error parsing JavaScript file {file_path}: {e}")
        
        return symbols
    
    def analyze_file_symbols(self, file_path: Path) -> Dict[str, List[str]]:
        """Analyze symbols in a file based on its language."""
        language = self.get_file_language(file_path)
        
        if language == 'python':
            return self.extract_python_symbols(file_path)
        elif language in ['javascript', 'typescript']:
            return self.extract_javascript_symbols(file_path)
        else:
            return {}
    
    def get_codebase_summary(self) -> str:
        """Generate a comprehensive summary of the codebase."""
        structure = self.analyze_file_structure()
        
        summary_parts = [
            f"📁 Codebase Analysis: {self.codebase_path.name}",
            f"📊 Total Files: {structure['total_files']}",
            ""
        ]
        
        # Languages
        if structure['languages']:
            summary_parts.append("🔤 Languages:")
            for lang, count in sorted(structure['languages'].items(), key=lambda x: x[1], reverse=True):
                summary_parts.append(f"  • {lang}: {count} files")
            summary_parts.append("")
        
        # Recent files (if Git is available)
        if structure['recent_files']:
            summary_parts.append("📝 Recent Changes (Git):")
            for file_info in structure['recent_files'][:5]:
                summary_parts.append(f"  • {file_info['path']} ({file_info['last_author']})")
            summary_parts.append("")
        
        # Large files
        if structure['large_files']:
            summary_parts.append("📦 Largest Files:")
            for file_info in structure['large_files'][:5]:
                size_kb = file_info['size'] // 1024
                summary_parts.append(f"  • {file_info['path']} ({size_kb}KB)")
            summary_parts.append("")
        
        # Directory structure (top level)
        top_dirs = [d for d in structure['directories'] if '/' not in d and '\\' not in d]
        if top_dirs:
            summary_parts.append("📂 Top-level Directories:")
            for directory in sorted(top_dirs)[:10]:
                summary_parts.append(f"  • {directory}/")
        
        return '\n'.join(summary_parts) 