"""
Project-S V2 File Tools
----------------------
Essential file operations for the V2 architecture:
- File reading and writing
- Directory listing
- File information
"""

import os
import asyncio
import aiofiles
import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path
import shutil

from tools.tool_interface import BaseTool

logger = logging.getLogger(__name__)

class FileReaderTool(BaseTool):
    """
    File content reading tool.
    
    Category: file
    Version: 2.0.0
    Requires permissions: Yes
    Safe: Yes
    """
    
    def __init__(self):
        super().__init__()
        self.name = "file_reader"
        self.description = "Read file content and return as string"
        
    async def execute(self, 
                    path: str, 
                    encoding: str = 'utf-8',
                    max_size: int = 1024 * 1024) -> Dict[str, Any]:
        """
        Read file content.
        
        Args:
            path: File path to read
            encoding: File encoding (default: utf-8)
            max_size: Maximum file size in bytes (default: 1MB)
            
        Returns:
            Dict: Result with success status and content
        """
        try:
            file_path = Path(path).resolve()
            
            # Check if file exists
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File does not exist: {path}"
                }
                
            # Check if it's actually a file
            if not file_path.is_file():
                return {
                    "success": False,
                    "error": f"Path is not a file: {path}"
                }
                
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > max_size:
                return {
                    "success": False,
                    "error": f"File size ({file_size} bytes) exceeds maximum ({max_size} bytes)"
                }
                
            # Read file content
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
                
            return {
                "success": True,
                "content": content,
                "size": len(content),
                "encoding": encoding,
                "path": str(file_path)
            }
        
        except UnicodeDecodeError:
            return {
                "success": False,
                "error": f"Encoding error: file cannot be read with {encoding} encoding"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error reading file: {str(e)}"
            }

class FileWriterTool(BaseTool):
    """
    File content writing tool.
    
    Category: file
    Version: 2.0.0
    Requires permissions: Yes
    Safe: Yes
    """
    
    def __init__(self):
        super().__init__()
        self.name = "file_writer"
        self.description = "Write content to file"
        
    async def execute(self, 
                    path: str, 
                    content: str,
                    encoding: str = 'utf-8',
                    create_dirs: bool = True,
                    append: bool = False) -> Dict[str, Any]:
        """
        Write content to file.
        
        Args:
            path: File path to write
            content: Content to write
            encoding: File encoding (default: utf-8)
            create_dirs: Create parent directories if needed (default: True)
            append: Append to file instead of overwrite (default: False)
            
        Returns:
            Dict: Result with success status
        """
        try:
            file_path = Path(path).resolve()
            
            # Create parent directories if needed
            if create_dirs:
                file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write mode
            mode = 'a' if append else 'w'
            
            # Write file content
            async with aiofiles.open(file_path, mode, encoding=encoding) as f:
                await f.write(content)
                
            return {
                "success": True,
                "path": str(file_path),
                "size": len(content),
                "mode": "append" if append else "write"
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Error writing file: {str(e)}"
            }

class DirectoryListerTool(BaseTool):
    """
    Directory listing tool.
    
    Category: file
    Version: 2.0.0
    Requires permissions: No
    Safe: Yes
    """
    
    def __init__(self):
        super().__init__()
        self.name = "directory_lister"
        self.description = "List directory contents"
        
    async def execute(self, 
                    path: str = ".",
                    include_hidden: bool = False,
                    recursive: bool = False,
                    max_depth: int = 3) -> Dict[str, Any]:
        """
        List directory contents.
        
        Args:
            path: Directory path to list (default: current directory)
            include_hidden: Include hidden files/directories (default: False)
            recursive: List recursively (default: False)
            max_depth: Maximum recursion depth (default: 3)
            
        Returns:
            Dict: Result with directory contents
        """
        try:
            dir_path = Path(path).resolve()
            
            if not dir_path.exists():
                return {
                    "success": False,
                    "error": f"Directory does not exist: {path}"
                }
                
            if not dir_path.is_dir():
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}"
                }
            
            items = []
            
            def scan_directory(current_path: Path, depth: int = 0):
                if depth > max_depth:
                    return
                    
                try:
                    for item in current_path.iterdir():
                        # Skip hidden files if not requested
                        if not include_hidden and item.name.startswith('.'):
                            continue
                            
                        item_info = {
                            "name": item.name,
                            "path": str(item),
                            "type": "directory" if item.is_dir() else "file",
                            "size": item.stat().st_size if item.is_file() else None,
                            "depth": depth
                        }
                        
                        items.append(item_info)
                        
                        # Recurse into subdirectories
                        if recursive and item.is_dir() and depth < max_depth:
                            scan_directory(item, depth + 1)
                            
                except PermissionError:
                    # Skip directories we can't access
                    pass
            
            scan_directory(dir_path)
            
            return {
                "success": True,
                "path": str(dir_path),
                "items": items,
                "count": len(items)
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Error listing directory: {str(e)}"
            }

# Tool instances for registration
file_reader = FileReaderTool()
file_writer = FileWriterTool()
directory_lister = DirectoryListerTool()

# Export tools for registry
__all__ = ['FileReaderTool', 'FileWriterTool', 'DirectoryListerTool', 
           'file_reader', 'file_writer', 'directory_lister']
