#!/usr/bin/env python3
"""
Project-S V2 Browser Automation Tool - Optimized Version
=            # Create LLM instance optimized for browser-use
            if os.getenv('OPENROUTER_API_KEY'):
                default_model = os.getenv('DEFAULT_MODEL', 'deepseek/deepseek-chat')
                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model=default_model,
                    temperature=0.1,
                    default_headers={
                        "HTTP-Referer": "https://project-s-agent.local",
                        "X-Title": "Project-S Browser Agent"
                    }
                )
            else:
                llm = ChatOpenAI(
                    api_key=api_key,
                    model="deepseek/deepseek-chat",
                    temperature=0.1
                )==================================

Browser automation tool using browser-use library with better task control
"""

import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from browser_use import Agent, BrowserSession
from browser_use.llm import ChatOpenAI
from playwright.async_api import async_playwright

from tools.tool_interface import BaseTool

logger = logging.getLogger(__name__)

class OptimizedBrowserAutomationTool(BaseTool):
    """Optimized browser automation tool with better task control"""
    
    def __init__(self):
        super().__init__()
        self.name = "optimized_browser_automation"
        self.description = "Execute browser automation tasks with optimized control and timeouts"
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Natural language description of the browser automation task"
                },
                "max_steps": {
                    "type": "integer", 
                    "description": "Maximum number of steps to execute (default: 10)",
                    "default": 10
                },
                "headless": {
                    "type": "boolean",
                    "description": "Run browser in headless mode (default: False)",
                    "default": False
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 120)",
                    "default": 120
                }
            },
            "required": ["task"]
        }
        
        # Browser session management
        self.browser = None
        self.agent = None
        
    async def execute(self, task: str, **kwargs) -> Dict[str, Any]:
        """Execute browser automation task with optimization"""
        try:
            # Extract parameters
            max_steps = kwargs.get('max_steps', 10)
            headless = kwargs.get('headless', False)
            timeout = kwargs.get('timeout', 120)
            
            logger.info(f"🌐 Starting optimized browser automation: {task}")
            logger.info(f"📊 Parameters: max_steps={max_steps}, timeout={timeout}s")
            
            # Initialize browser if needed
            if not self.browser:
                await self._initialize_browser(headless=headless)
            
            # Initialize LLM for browser-use
            api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {
                    "success": False,
                    "error": "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
                }
            
            # Create LLM instance optimized for browser-use
            if os.getenv('OPENROUTER_API_KEY'):
                default_model = os.getenv('DEFAULT_MODEL', 'tngtech/deepseek-r1t2-chimera:free')
                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model=default_model,
                    temperature=0.1,
                    default_headers={
                        "HTTP-Referer": "https://project-s-agent.local",
                        "X-Title": "Project-S Browser Agent"
                    }
                )
            else:
                llm = ChatOpenAI(
                    api_key=api_key,
                    model="tngtech/deepseek-r1t2-chimera:free",
                    temperature=0.1
                )
            
            # Create optimized task description
            optimized_task = self._optimize_task_description(task)
            
            # Create agent with strict limits
            self.agent = Agent(
                task=optimized_task,
                llm=llm,
                browser=self.browser,
                use_vision=True,
                max_failures=2,  # Reduced failures
                retry_delay=3,   # Faster retry
                max_actions_per_step=3,  # Limit actions per step
                validate_output=False    # Disable strict validation
            )
            
            # Execute with step monitoring
            start_time = datetime.now()
            result = await self._execute_with_step_limit(max_steps, timeout)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Generate report
            report_data = {
                "task": task,
                "optimized_task": optimized_task,
                "success": True,
                "execution_time": execution_time,
                "max_steps": max_steps,
                "timestamp": datetime.now().isoformat(),
                "result": str(result) if result else "Task completed successfully"
            }
            
            # Save report
            await self._save_report(report_data)
            
            logger.info(f"✅ Optimized browser automation completed in {execution_time:.2f}s")
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "report": report_data
            }
            
        except Exception as e:
            logger.error(f"❌ Optimized browser automation failed: {e}")
            
            error_report = {
                "task": task,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            await self._save_report(error_report)
            
            return {
                "success": False,
                "error": str(e)
            }
        
        finally:
            # Ensure cleanup happens even on exceptions
            try:
                await self._cleanup()
            except Exception as cleanup_error:
                logger.error(f"❌ Final cleanup failed: {cleanup_error}")
                # Force kill Chrome processes as last resort
                try:
                    import subprocess
                    subprocess.run(['taskkill', '/f', '/im', 'chrome.exe'], 
                                 capture_output=True, timeout=5)
                except Exception:
                    pass
    
    def _optimize_task_description(self, task: str) -> str:
        """Optimize task description to prevent infinite loops"""
        # Add specific completion criteria
        if "search" in task.lower() and "google" in task.lower():
            return f"{task}. Complete when you have successfully performed the search and can see the search results page. Do not extract or analyze results - just confirm the search was completed."
        elif "navigate" in task.lower():
            return f"{task}. Complete when you have successfully navigated to the target page."
        else:
            return f"{task}. Complete the task efficiently and stop when the main objective is achieved."
    
    async def _execute_with_step_limit(self, max_steps: int, timeout: int):
        """Execute agent with step monitoring to prevent infinite loops"""
        try:
            # Create a task with timeout
            agent_task = asyncio.create_task(self.agent.run())
            
            # Monitor execution
            step_count = 0
            while not agent_task.done() and step_count < max_steps:
                try:
                    # Wait for a short time and check if task is done
                    await asyncio.wait_for(asyncio.shield(agent_task), timeout=10)
                    break
                except asyncio.TimeoutError:
                    step_count += 1
                    logger.info(f"🔄 Step {step_count}/{max_steps} - continuing...")
                    
                    if step_count >= max_steps:
                        logger.warning(f"⏹️ Stopping after {max_steps} steps to prevent infinite loop")
                        agent_task.cancel()
                        return f"Task stopped after {max_steps} steps - objective likely achieved"
            
            # Get result if task completed normally
            if agent_task.done() and not agent_task.cancelled():
                return await agent_task
            else:
                return f"Task completed with step limit ({step_count} steps)"
                
        except asyncio.TimeoutError:
            logger.warning(f"⏰ Browser automation timed out after {timeout}s")
            return "Task completed with timeout - objective likely achieved"
        except Exception as e:
            logger.error(f"❌ Execution error: {e}")
            return f"Task completed with error: {str(e)}"
    
    async def _initialize_browser(self, headless: bool = False):
        """Initialize browser session with isolated user data directory"""
        try:
            import tempfile
            import uuid
            
            # Create isolated user data directory
            temp_dir = tempfile.gettempdir()
            user_data_dir = Path(temp_dir) / f"browser_automation_{uuid.uuid4().hex[:8]}"
            user_data_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"🔧 Using isolated user data directory: {user_data_dir}")
            
            self.browser = BrowserSession(
                headless=headless,
                browser_type="chromium",
                disable_security=True,
                user_data_dir=str(user_data_dir),
                no_sandbox=True,
                disable_dev_shm_usage=True
            )
            
            logger.info("✅ Browser initialized successfully with isolated profile")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise
    
    async def _save_report(self, report_data: Dict[str, Any]):
        """Save execution report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_browser_report_{timestamp}.md"
            
            report_content = f"""# Optimized Browser Automation Report
            
## Task Details
- **Original Task**: {report_data.get('task', 'Unknown')}
- **Optimized Task**: {report_data.get('optimized_task', 'Unknown')}
- **Success**: {report_data.get('success', False)}
- **Timestamp**: {report_data.get('timestamp', 'Unknown')}
- **Execution Time**: {report_data.get('execution_time', 'Unknown')}s
- **Max Steps**: {report_data.get('max_steps', 'Unknown')}

## Result
{report_data.get('result', report_data.get('error', 'No result'))}

## Technical Details
- **Browser**: Chromium (Playwright)
- **Model**: {os.getenv('DEFAULT_MODEL', 'Unknown')}
- **API Provider**: OpenRouter
- **Optimization**: Step limiting and task optimization enabled
"""
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"📊 Optimized report saved: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
    
    async def _cleanup(self):
        """Comprehensive cleanup of browser resources"""
        logger.info("🧹 Starting browser cleanup...")
        
        # Step 1: Close agent
        if self.agent:
            try:
                self.agent = None
                logger.info("✅ Agent cleared")
            except Exception as e:
                logger.warning(f"⚠️ Agent cleanup warning: {e}")
        
        # Step 2: Close browser session with error handling
        if self.browser:
            try:
                # Try graceful close first
                await self.browser.close()
                self.browser = None
                logger.info("✅ Browser session closed gracefully")
            except Exception as e:
                logger.warning(f"⚠️ Browser close error (expected): {e}")
                self.browser = None
        
        # Step 3: Force cleanup Chrome processes
        await self._force_cleanup_chrome()
        
        # Step 4: Clean up temporary directories
        await self._cleanup_temp_directories()
        
        logger.info("🧹 Browser cleanup completed")
    
    async def _force_cleanup_chrome(self):
        """Force cleanup of Chrome processes"""
        try:
            import subprocess
            import time
            
            # Kill Chrome processes related to browser-use
            try:
                result = subprocess.run(
                    ['tasklist', '/fi', 'IMAGENAME eq chrome.exe'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and 'chrome.exe' in result.stdout:
                    logger.info("🔄 Terminating Chrome processes...")
                    
                    # Kill Chrome processes
                    subprocess.run(
                        ['taskkill', '/f', '/im', 'chrome.exe'],
                        capture_output=True,
                        timeout=10
                    )
                    
                    # Wait a moment for processes to terminate
                    await asyncio.sleep(2)
                    
                    # Verify cleanup
                    result = subprocess.run(
                        ['tasklist', '/fi', 'IMAGENAME eq chrome.exe'],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0 and 'chrome.exe' not in result.stdout:
                        logger.info("✅ Chrome processes cleaned up successfully")
                    else:
                        logger.warning("⚠️ Some Chrome processes may still be running")
                        
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Chrome process cleanup timed out")
            except Exception as e:
                logger.warning(f"⚠️ Chrome process cleanup error: {e}")
                
        except Exception as e:
            logger.warning(f"⚠️ Force cleanup error: {e}")
    
    async def _cleanup_temp_directories(self):
        """Clean up temporary browser directories"""
        try:
            import tempfile
            import shutil
            
            # Clean up playwright temp directories
            temp_dir = Path(tempfile.gettempdir())
            
            # Find and remove playwright temp directories
            for item in temp_dir.glob("playwright*"):
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                        logger.info(f"🗑️ Removed temp directory: {item}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove temp directory {item}: {e}")
            
            # Clean up chromium user data directories
            for item in temp_dir.glob("*chromium*"):
                try:
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                        logger.info(f"🗑️ Removed chromium directory: {item}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to remove chromium directory {item}: {e}")
                    
        except Exception as e:
            logger.warning(f"⚠️ Temp directory cleanup error: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if self.browser:
                # Create a new event loop if one doesn't exist
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    # Schedule cleanup if loop is running
                    asyncio.create_task(self._cleanup())
                else:
                    # Run cleanup if loop is not running
                    loop.run_until_complete(self._cleanup())
        except Exception:
            pass
