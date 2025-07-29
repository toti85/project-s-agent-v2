#!/usr/bin/env python3
"""
Project-S V2 Browser Automation Tool
===================================            # Create LLM instance optimized for browser-use
            if os.getenv('OPENROUTER_API_KEY'):
                # Use OpenRouter with Claude for better browser-use compatibility
                llm = ChatOpenAI(
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    model="anthropic/claude-3-haiku",  # Claude modell, jó a browser-use-hoz
                    temperature=0.1,  # Lower temperature for more consistent actions
                    default_headers={
                        "HTTP-Referer": "https://project-s-agent.local",
                        "X-Title": "Project-S Browser Agent"
                    }
                )
            else:
                # Use OpenAI directly
                llm = ChatOpenAI(
                    api_key=api_key,
                    model="tngtech/deepseek-r1t2-chimera:free",
                    temperature=0.1
                )ser automation tool using browser-use library for Project-S V2.
Provides comprehensive web automation capabilities for autonomous agents.

Author: Project-S V2 AI System
Date: 2025-07-10
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Browser-use imports
try:
    from browser_use import Agent, Controller
    from browser_use.browser.browser import Browser
    from browser_use.browser.context import BrowserContext
    from browser_use.llm import ChatOpenAI
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False

from tools.tool_interface import BaseTool

logger = logging.getLogger(__name__)

class BrowserAutomationTool(BaseTool):
    """
    Advanced browser automation tool using browser-use library.
    
    Capabilities:
    - Web navigation and page interaction
    - Form filling and data extraction
    - Screenshot capture
    - Element interaction (click, type, scroll)
    - Multi-tab management
    - Async execution
    """
    
    def __init__(self):
        super().__init__()
        self.name = "browser_automation"
        self.description = "Advanced browser automation for web navigation, form filling, and data extraction"
        self.browser = None
        self.context = None
        self.agent = None
        self.controller = None
        
        if not BROWSER_USE_AVAILABLE:
            logger.warning("browser-use library not available. Browser automation disabled.")
    
    async def execute(self, task: str, **kwargs) -> Dict[str, Any]:
        """
        Execute browser automation task.
        
        Args:
            task: Natural language description of the task
            **kwargs: Additional parameters
                - headless: Run in headless mode (default: False)
                - timeout: Task timeout in seconds (default: 60)
                - screenshot: Take screenshot after completion (default: True)
                - wait_time: Wait time between actions (default: 1)
        
        Returns:
            Dict containing execution results
        """
        if not BROWSER_USE_AVAILABLE:
            return {
                "success": False,
                "error": "browser-use library not available. Please install: pip install browser-use"
            }
        
        try:
            # Extract parameters
            headless = kwargs.get('headless', False)
            timeout = kwargs.get('timeout', 60)
            screenshot = kwargs.get('screenshot', True)
            wait_time = kwargs.get('wait_time', 1)
            
            logger.info(f"🌐 Starting browser automation task: {task}")
            
            # Initialize browser if needed
            if not self.browser:
                await self._initialize_browser(headless=headless)
            
            # Initialize LLM for browser-use
            # Use OpenRouter API key if available, fallback to OpenAI
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
                    model=default_model,  # Use model from environment
                    temperature=0.1,  # Lower temperature for more consistent actions
                    default_headers={
                        "HTTP-Referer": "https://project-s-agent.local",
                        "X-Title": "Project-S Browser Agent"
                    }
                )
            else:
                # Use OpenAI directly (fallback)
                llm = ChatOpenAI(
                    api_key=api_key,
                    model="gpt-3.5-turbo",
                    temperature=0.1
                )
            
            # Create agent for task execution with proper parameters
            self.agent = Agent(
                task=task,
                llm=llm,
                browser=self.browser,
                use_vision=True,  # Enable vision for better page understanding
                max_failures=3,   # Allow up to 3 failures before giving up
                retry_delay=5,    # Wait 5 seconds between retries
                validate_output=True  # Validate action outputs
            )
            
            # Execute the task
            start_time = time.time()
            result = await asyncio.wait_for(
                self.agent.run(),
                timeout=timeout
            )
            
            execution_time = time.time() - start_time
            
            # Take screenshot if requested
            screenshot_path = None
            if screenshot and self.context:
                screenshot_path = await self._take_screenshot()
            
            # Extract results
            extracted_data = await self._extract_page_data()
            
            # Cleanup
            await self._cleanup()
            
            return {
                "success": True,
                "task": task,
                "execution_time": execution_time,
                "result": result,
                "screenshot": screenshot_path,
                "extracted_data": extracted_data,
                "message": f"Browser automation task completed successfully in {execution_time:.2f}s"
            }
            
        except asyncio.TimeoutError:
            await self._cleanup()
            return {
                "success": False,
                "error": f"Task timeout after {timeout} seconds",
                "task": task
            }
        except Exception as e:
            await self._cleanup()
            logger.error(f"Browser automation failed: {e}")
            return {
                "success": False,
                "error": f"Browser automation failed: {str(e)}",
                "task": task
            }
    
    async def navigate(self, url: str, wait_time: int = 3) -> Dict[str, Any]:
        """Navigate to a specific URL."""
        try:
            if not self.browser:
                await self._initialize_browser()
            
            page = await self.context.new_page()
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            await asyncio.sleep(wait_time)
            
            return {
                "success": True,
                "url": url,
                "title": await page.title(),
                "message": f"Successfully navigated to {url}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Navigation failed: {str(e)}",
                "url": url
            }
    
    async def click_element(self, selector: str, wait_time: int = 1) -> Dict[str, Any]:
        """Click on an element by selector."""
        try:
            if not self.context:
                return {"success": False, "error": "Browser not initialized"}
            
            page = self.context.pages[0] if self.context.pages else None
            if not page:
                return {"success": False, "error": "No active page"}
            
            await page.click(selector)
            await asyncio.sleep(wait_time)
            
            return {
                "success": True,
                "selector": selector,
                "message": f"Successfully clicked element: {selector}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Click failed: {str(e)}",
                "selector": selector
            }
    
    async def type_text(self, selector: str, text: str, clear_first: bool = True) -> Dict[str, Any]:
        """Type text into an element."""
        try:
            if not self.context:
                return {"success": False, "error": "Browser not initialized"}
            
            page = self.context.pages[0] if self.context.pages else None
            if not page:
                return {"success": False, "error": "No active page"}
            
            if clear_first:
                await page.fill(selector, text)
            else:
                await page.type(selector, text)
            
            return {
                "success": True,
                "selector": selector,
                "text": text,
                "message": f"Successfully typed text into: {selector}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Type failed: {str(e)}",
                "selector": selector
            }
    
    async def extract_text(self, selector: str = None) -> Dict[str, Any]:
        """Extract text from page or specific element."""
        try:
            if not self.context:
                return {"success": False, "error": "Browser not initialized"}
            
            page = self.context.pages[0] if self.context.pages else None
            if not page:
                return {"success": False, "error": "No active page"}
            
            if selector:
                text = await page.text_content(selector)
                return {
                    "success": True,
                    "selector": selector,
                    "text": text,
                    "message": f"Text extracted from: {selector}"
                }
            else:
                # Extract all text from page
                text = await page.text_content("body")
                return {
                    "success": True,
                    "text": text,
                    "message": "Page text extracted"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Text extraction failed: {str(e)}"
            }
    
    async def _initialize_browser(self, headless: bool = False) -> None:
        """Initialize browser instance."""
        try:
            self.browser = Browser(
                headless=headless,
                browser_type="chromium"
            )
            await self.browser.start()
            self.context = await self.browser.new_context()
            self.controller = Controller()
            
            logger.info("✅ Browser initialized successfully")
        except Exception as e:
            logger.error(f"❌ Browser initialization failed: {e}")
            raise
    
    async def _take_screenshot(self) -> Optional[str]:
        """Take a screenshot of the current page."""
        try:
            if not self.context or not self.context.pages:
                return None
            
            page = self.context.pages[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_path = f"screenshots/browser_automation_{timestamp}.png"
            
            # Create screenshots directory if it doesn't exist
            Path("screenshots").mkdir(exist_ok=True)
            
            await page.screenshot(path=screenshot_path)
            logger.info(f"📸 Screenshot saved: {screenshot_path}")
            return screenshot_path
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return None
    
    async def _extract_page_data(self) -> Dict[str, Any]:
        """Extract basic page data."""
        try:
            if not self.context or not self.context.pages:
                return {}
            
            page = self.context.pages[0]
            
            return {
                "url": page.url,
                "title": await page.title(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Page data extraction failed: {e}")
            return {}
    
    async def _cleanup(self) -> None:
        """Cleanup browser resources."""
        try:
            if self.context:
                await self.context.close()
                self.context = None
            
            if self.browser:
                await self.browser.close()
                self.browser = None
            
            self.agent = None
            self.controller = None
            
            logger.info("🧹 Browser cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check tool health status."""
        return {
            "tool_name": self.name,
            "available": BROWSER_USE_AVAILABLE,
            "browser_active": self.browser is not None,
            "context_active": self.context is not None,
            "status": "healthy" if BROWSER_USE_AVAILABLE else "library_missing"
        }

# Export for tool registry
__all__ = ["BrowserAutomationTool"]
