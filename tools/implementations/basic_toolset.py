"""
Project-S V2 Basic Tool Set
===========================
Essential tools for immediate execution capability.
These are the tools that AI models expect to exist.
"""

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class FileSystemUtility:
    """File system operations tool."""
    
    @staticmethod
    async def create_directory(path: str) -> Dict[str, Any]:
        """Create a directory."""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Directory created: {path}")
            return {"status": "success", "message": f"Directory '{path}' created", "path": path}
        except Exception as e:
            logger.error(f"❌ Failed to create directory {path}: {e}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    async def create_file(path: str, content: str = "") -> Dict[str, Any]:
        """Create a file with optional content."""
        try:
            file_path = Path(path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding='utf-8')
            logger.info(f"✅ File created: {path}")
            return {"status": "success", "message": f"File '{path}' created", "path": path}
        except Exception as e:
            logger.error(f"❌ Failed to create file {path}: {e}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    async def write_to_file(path: str, content: str, append: bool = False) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            action = "appended to" if append else "written to"
            logger.info(f"✅ Content {action}: {path}")
            return {"status": "success", "message": f"Content {action} '{path}'", "path": path}
        except Exception as e:
            logger.error(f"❌ Failed to write to file {path}: {e}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    async def read_file(path: str) -> Dict[str, Any]:
        """Read content from a file."""
        try:
            content = Path(path).read_text(encoding='utf-8')
            logger.info(f"✅ File read: {path}")
            return {"status": "success", "content": content, "path": path}
        except Exception as e:
            logger.error(f"❌ Failed to read file {path}: {e}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    async def list_directory(path: str = ".") -> Dict[str, Any]:
        """List directory contents."""
        try:
            items = []
            for item in Path(path).iterdir():
                items.append({
                    "name": item.name,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else None
                })
            logger.info(f"✅ Directory listed: {path}")
            return {"status": "success", "items": items, "path": path}
        except Exception as e:
            logger.error(f"❌ Failed to list directory {path}: {e}")
            return {"status": "error", "message": str(e)}

class DateUtility:
    """Date and time operations tool."""
    
    @staticmethod
    async def get_current_date(format: str = "%Y-%m-%d") -> Dict[str, Any]:
        """Get current date in specified format."""
        try:
            current_date = datetime.now().strftime(format)
            logger.info(f"✅ Current date: {current_date}")
            return {"status": "success", "date": current_date, "format": format}
        except Exception as e:
            logger.error(f"❌ Failed to get current date: {e}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    async def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> Dict[str, Any]:
        """Get current datetime in specified format."""
        try:
            current_datetime = datetime.now().strftime(format)
            logger.info(f"✅ Current datetime: {current_datetime}")
            return {"status": "success", "datetime": current_datetime, "format": format}
        except Exception as e:
            logger.error(f"❌ Failed to get current datetime: {e}")
            return {"status": "error", "message": str(e)}

class SystemCommandUtility:
    """System command execution tool."""
    
    @staticmethod
    async def execute_command(command: str, working_dir: Optional[str] = None) -> Dict[str, Any]:
        """Execute a system command."""
        try:
            import subprocess
            
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=working_dir,
                timeout=30
            )
            
            logger.info(f"✅ Command executed: {command}")
            return {
                "status": "success",
                "command": command,
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except subprocess.TimeoutExpired:
            logger.error(f"❌ Command timeout: {command}")
            return {"status": "error", "message": "Command timeout"}
        except Exception as e:
            logger.error(f"❌ Failed to execute command {command}: {e}")
            return {"status": "error", "message": str(e)}

# Tool Registry for Smart Orchestrator
BASIC_TOOLS = {
    "file_system_utility": FileSystemUtility,
    "date_utility": DateUtility,
    "system_command_utility": SystemCommandUtility,
}

# Tool Capabilities Map
TOOL_CAPABILITIES = {
    "file_system_utility": [
        "create_directory", "create_file", "write_to_file", 
        "read_file", "list_directory"
    ],
    "date_utility": [
        "get_current_date", "get_current_datetime"
    ],
    "system_command_utility": [
        "execute_command"
    ]
}
