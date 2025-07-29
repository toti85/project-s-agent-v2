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
from langchain_openai import ChatOpenAI
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
                    "description": "Timeout in seconds (default: 60)",
                    "default": 60
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
        """Execute browser automation task"""
        try:
            # Extract parameters
            headless = kwargs.get('headless', False)
            timeout = kwargs.get('timeout', 60)
            screenshot = kwargs.get('screenshot', True)
            
            logger.info(f"🌐 Starting browser automation task: {task}")
            
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
                # Use OpenRouter with configured model from environment
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
                # Use OpenAI directly (fallback)
                llm = ChatOpenAI(
                    api_key=api_key,
                    model="tngtech/deepseek-r1t2-chimera:free",
                    temperature=0.1
                )
            
            # Create agent for task execution
            self.agent = Agent(
                task=task,
                llm=llm,
                browser=self.browser,
                use_vision=True,
                max_failures=3,
                retry_delay=5
            )
            
            # Execute the task
            start_time = datetime.now()
            result = await self.agent.run()
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Generate report
            report_data = {
                "task": task,
                "success": True,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "result": str(result) if result else "Task completed"
            }
            
            # Save report
            await self._save_report(report_data)
            
            logger.info(f"✅ Browser automation completed successfully in {execution_time:.2f}s")
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "report": report_data
            }
            
        except Exception as e:
            logger.error(f"❌ Browser automation failed: {e}")
            
            # Generate error report
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
            # Cleanup
            await self._cleanup()
    
    async def _initialize_browser(self, headless: bool = False):
        """Initialize browser session"""
        try:
            self.browser = BrowserSession(
                headless=headless,
                browser_type="chromium",
                disable_security=True,
                user_data_dir=None
            )
            
            logger.info("✅ Browser initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize browser: {e}")
            raise
    
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
- **Model**: {os.getenv('DEFAULT_MODEL', 'Unknown')}
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
