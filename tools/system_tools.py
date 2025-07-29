"""
Project-S V2 System Tools
------------------------
Essential system operations for the V2 architecture:
- Safe system command execution
- System information retrieval
"""

import os
import asyncio
import logging
import subprocess
import platform
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from tools.tool_interface import BaseTool

logger = logging.getLogger(__name__)

class SystemCommandTool(BaseTool):
    """
    Development system command execution tool - UNRESTRICTED for development use.
    
    Category: system
    Version: 2.1.0-dev
    Requires permissions: Yes
    Safe: DEVELOPMENT MODE - All commands allowed
    """
    
    def __init__(self):
        super().__init__()
        self.name = "system_command"
        self.description = "Execute any system command (development mode)"
        
    async def execute(self, 
                    command: str,
                    timeout: int = 30,
                    working_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute any system command - DEVELOPMENT MODE.
        
        Args:
            command: Command to execute
            timeout: Timeout in seconds (default: 30)
            working_dir: Working directory for command execution
            
        Returns:
            Dict: Result with command output and status
        """
        try:
            # DEVELOPMENT MODE: No security restrictions
            print(f"🔧 DEV MODE: Executing command: {command}")
            
            # Set working directory
            cwd = None
            if working_dir:
                cwd = Path(working_dir).resolve()
                if not cwd.exists() or not cwd.is_dir():
                    return {
                        "success": False,
                        "error": f"Working directory does not exist: {working_dir}"
                    }
                cwd = str(cwd)
            
            # Execute command through shell
            if platform.system() == 'Windows':
                # Always use shell=True for Windows for maximum compatibility
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd
                )
            else:
                # Unix/Linux systems
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd
                )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds"
                }
            
            # Decode output
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            return {
                "success": True,
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "working_dir": cwd or "current"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Command execution failed: {str(e)}"
            }

class SystemInfoTool(BaseTool):
    """
    System information retrieval tool.
    
    Category: system
    Version: 2.0.0
    Requires permissions: No
    Safe: Yes
    """
    
    def __init__(self):
        super().__init__()
        self.name = "system_info"
        self.description = "Get system information"
        
    async def execute(self, 
                    info_type: str = "basic") -> Dict[str, Any]:
        """
        Get system information.
        
        Args:
            info_type: Type of info to retrieve (basic, detailed, performance)
            
        Returns:
            Dict: System information
        """
        try:
            info = {
                "platform": platform.system(),
                "platform_release": platform.release(),
                "platform_version": platform.version(),
                "architecture": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version()
            }
            
            if info_type in ["detailed", "performance"]:
                try:
                    import psutil
                    info.update({
                        "cpu_count": psutil.cpu_count(),
                        "cpu_count_logical": psutil.cpu_count(logical=True),
                        "memory_total": psutil.virtual_memory().total,
                        "memory_available": psutil.virtual_memory().available,
                        "disk_usage": {
                            "total": psutil.disk_usage('/').total if platform.system() != 'Windows' else psutil.disk_usage('C:').total,
                            "free": psutil.disk_usage('/').free if platform.system() != 'Windows' else psutil.disk_usage('C:').free
                        }
                    })
                except ImportError:
                    info["psutil_available"] = False
            
            if info_type == "performance":
                try:
                    import psutil
                    info.update({
                        "cpu_percent": psutil.cpu_percent(interval=1),
                        "memory_percent": psutil.virtual_memory().percent,
                        "boot_time": psutil.boot_time()
                    })
                except ImportError:
                    pass
            
            return {
                "success": True,
                "info_type": info_type,
                "system_info": info
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": f"Error getting system info: {str(e)}"
            }

class PowerCommandTool(BaseTool):
    """
    PowerShell command execution tool for Windows-specific operations.
    
    Category: system
    Version: 2.0.0
    Requires permissions: Yes
    Safe: Development mode
    """
    
    def __init__(self):
        super().__init__()
        self.name = "power_command"
        self.description = "Execute PowerShell commands on Windows"
        
    async def execute(self, 
                    command: str,
                    timeout: int = 30) -> Dict[str, Any]:
        """
        Execute PowerShell command.
        
        Args:
            command: PowerShell command to execute
            timeout: Timeout in seconds (default: 30)
            
        Returns:
            Dict: Result with command output and status
        """
        try:
            if platform.system() != 'Windows':
                return {
                    "success": False,
                    "error": "PowerShell commands only available on Windows"
                }
            
            print(f"⚡ PowerShell: {command}")
            
            # Execute via PowerShell
            ps_command = ["powershell", "-Command", command]
            
            process = await asyncio.create_subprocess_exec(
                *ps_command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return {
                    "success": False,
                    "error": f"PowerShell command timed out after {timeout} seconds"
                }
            
            # Decode output
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')
            
            return {
                "success": True,
                "command": command,
                "return_code": process.returncode,
                "stdout": stdout_text,
                "stderr": stderr_text,
                "shell": "PowerShell"
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"PowerShell execution failed: {str(e)}"
            }

# Tool instances for registration
system_command = SystemCommandTool()
system_info = SystemInfoTool()
power_command = PowerCommandTool()

# Export tools for registry
__all__ = ['SystemCommandTool', 'SystemInfoTool', 'PowerCommandTool', 'system_command', 'system_info', 'power_command']
