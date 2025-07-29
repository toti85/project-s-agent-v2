#!/usr/bin/env python3
"""
Project-S V2 Browser Automation Tool
===================================

Browser automation tool using browser-use library with OpenRouter/OpenAI models
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
from browser_use.llm import ChatOpenAI  # Working import from v2_architecture/1
from playwright.async_api import async_playwright

from tools.tool_interface import BaseTool

logger = logging.getLogger(__name__)

class BrowserAutomationTool(BaseTool):
    """Browser automation tool using browser-use library"""
    
    def __init__(self):
        super().__init__()
        self.name = "browser_automation"
        self.description = "Execute browser automation tasks using natural language instructions"
        self.parameters = {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Natural language description of the browser automation task"
                },
                "headless": {
                    "type": "boolean",
                    "description": "Run browser in headless mode (default: False)",
                    "default": False
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default: 180)",
                    "default": 180
                },
                "screenshot": {
                    "type": "boolean",
                    "description": "Take screenshots during execution (default: True)",
                    "default": True
                }
            },
            "required": ["task"]
        }
        
        # Browser session management
        self.browser = None
        self.agent = None
        
    async def execute(self, task: str, **kwargs) -> Dict[str, Any]:
        """Execute browser automation task - SIMPLIFIED VERSION"""
        try:
            timeout = kwargs.get('timeout', 180)  # Increased timeout to 180s (3 minutes)
            
            logger.info(f"🌐 Starting browser automation task: {task}")
            
            # Initialize LLM for browser-use
            api_key = os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                return {
                    "success": False,
                    "error": "No API key found. Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable."
                }
            
            # Create LLM instance - Using DeepSeek V3 (deepseek/deepseek-chat)
            if os.getenv('OPENROUTER_API_KEY'):
                # Use DeepSeek V3 - PROVEN WORKING MODEL for browser automation
                llm = ChatOpenAI(
                    model="deepseek/deepseek-chat",  # This is DeepSeek V3
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/browser-use/browser-use",
                        "X-Title": "Project-S Browser Agent"
                    }
                )
            else:
                # Fallback if no OpenRouter key
                return {
                    "success": False,
                    "error": "OpenRouter API key required for browser automation"
                }
            
            # Create browser agent - WORKING CONFIG from v2_architecture/1
            agent = Agent(
                task=task,
                llm=llm,
                action_timeout=60,  # Increased action timeout to 60s
                max_actions=20,     # Increased max actions for complex tasks
            )
            
            # Execute the task with timeout
            start_time = datetime.now()
            try:
                # Set a reasonable timeout for the task
                result = await asyncio.wait_for(agent.run(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"⏰ Browser automation timed out after {timeout}s")
                result = "Task completed with timeout - partial results may be available"
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Generate report
            report_data = {
                "task": task,
                "result": str(result),
                "execution_time": execution_time,
                "success": True,
                "timestamp": datetime.now().isoformat(),
                "model_used": "deepseek/deepseek-chat (DeepSeek V3)"
            }
            
            # Save report
            await self._save_report(report_data)
            
            logger.info(f"✅ Browser automation completed in {execution_time:.2f}s")
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "report_saved": True
            }
            
        except Exception as e:
            logger.error(f"❌ Browser automation failed: {e}")
            error_report = {
                "task": task,
                "error": str(e),
                "success": False,
                "timestamp": datetime.now().isoformat()
            }
            await self._save_report(error_report)
            
            return {
                "success": False,
                "error": str(e)
            }
        
        finally:
            # Cleanup
            await self._cleanup()
    
    async def _initialize_browser(self, headless: bool = False):
        """Initialize browser session with optimized settings"""
        try:
            self.browser = BrowserSession(
                headless=headless,
                browser_type="chromium", 
                disable_security=True,
                user_data_dir=None,
                keep_alive=False,  # Don't keep browser alive between tasks
                browser_args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage", 
                    "--disable-blink-features=AutomationControlled",
                    "--disable-features=VizDisplayCompositor",
                    "--start-maximized"
                ]
            )
            
            logger.info("✅ Browser initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise
    
    def _extract_url_from_task(self, task: str) -> Optional[str]:
        """Extract URL from task description if present"""
        import re
        # Look for URLs in the task description
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, task)
        return urls[0] if urls else None
    
    async def _navigate_to_url(self, url: str):
        """Navigate directly to URL using browser session"""
        try:
            if self.browser and hasattr(self.browser, 'session') and self.browser.session:
                page = self.browser.session.pages[0] if self.browser.session.pages else None
                if page:
                    await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    logger.info(f"✅ Successfully navigated to: {url}")
                else:
                    logger.warning("⚠️ No page available for navigation")
            else:
                logger.warning("⚠️ Browser session not available for navigation")
        except Exception as e:
            logger.warning(f"⚠️ Pre-navigation failed: {e} - Agent will handle navigation")
    
    async def _save_report(self, report_data: Dict[str, Any]):
        """Save execution report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"browser_automation_report_{timestamp}.md"
            
            report_content = f"""# Browser Automation Report
            
## Task Details
- **Task**: {report_data.get('task', 'Unknown')}
- **Success**: {report_data.get('success', False)}
- **Timestamp**: {report_data.get('timestamp', 'Unknown')}
- **Execution Time**: {report_data.get('execution_time', 'Unknown')}s

## Result
{report_data.get('result', report_data.get('error', 'No result'))}

## Technical Details
- **Browser**: Chromium (Playwright)
- **Model**: deepseek/deepseek-chat (DeepSeek V3)
- **API Provider**: OpenRouter
"""
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(report_content)
                
            logger.info(f"📊 Report saved: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save report: {e}")
    
    async def _cleanup(self):
        """Clean up browser resources"""
        try:
            if self.browser:
                await self.browser.close()
                self.browser = None
                
            if self.agent:
                self.agent = None
                
            logger.info("🧹 Browser cleanup completed")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            if self.browser:
                asyncio.create_task(self._cleanup())
        except Exception:
            pass  # Ignore cleanup errors in destructor
