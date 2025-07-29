#!/usr/bin/env python3
"""
SIMPLIFIED Lazy Engine Manager for Project-S V2
================================================
Single engine approach for maximum simplicity and performance.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class SimplifiedEngineManager:
    """
    SIMPLIFIED engine manager that always uses the Enhanced Workflow Engine.
    Same power, half the complexity!
    """
    
    def __init__(self):
        """Initialize the simplified engine manager."""
        self._primary_engine: Optional[Any] = None
        self._browser_tool: Optional[Any] = None
        self._load_time: Optional[float] = None
        self._usage_count = 0
        
        logger.info("🚀 SIMPLIFIED Engine Manager initialized")
    
    def get_engine(self, task_type: str = "any") -> str:
        """
        SIMPLIFIED: Always return the Enhanced Workflow Engine.
        No complex routing, just the best engine every time.
        """
        return "enhanced"
    
    def get_web_tool(self, web_task: str = "any") -> str:
        """
        SIMPLIFIED: Always return Browser Use for web tasks.
        No complex routing, just the proven tool every time.
        """
        return "browser_use"
        """
        Get an engine instance, loading it if necessary.
        
        Args:
            engine_type: Type of engine ('enhanced', 'langgraph', 'golden_age')
            force_load: Force reload even if already cached
            
        Returns:
            Engine instance or None if loading failed
        """
        start_time = time.time()
        
        # Return cached engine if available and not forcing reload
        if engine_type in self._loaded_engines and not force_load:
            logger.debug(f"🔄 Using cached {engine_type} engine")
            self._update_usage_stats(engine_type, from_cache=True)
            return self._loaded_engines[engine_type]
        
        # Load engine on demand
        logger.info(f"⚡ Loading {engine_type} engine on demand...")
        engine = self._load_engine(engine_type)
        
        if engine:
            self._loaded_engines[engine_type] = engine
            load_time = time.time() - start_time
            self._load_times[engine_type] = load_time
            self._update_usage_stats(engine_type, load_time=load_time)
            logger.info(f"✅ {engine_type} engine loaded in {load_time:.3f}s")
        else:
            logger.error(f"❌ Failed to load {engine_type} engine")
        
        return engine
    
    def _load_engine(self, engine_type: str) -> Optional[Any]:
        """Load a specific engine."""
        if engine_type not in self._engine_configs:
            logger.error(f"Unknown engine type: {engine_type}")
            return None
        
        config = self._engine_configs[engine_type]
        
        try:
            # Dynamic import
            module_name = config["module"]
            class_name = config["class"]
            
            module = __import__(module_name, fromlist=[class_name])
            engine_class = getattr(module, class_name)
            
            # Instantiate engine
            if engine_type == "langgraph":
                engine = engine_class("lazy_langgraph_engine")
            else:
                engine = engine_class()
            
            return engine
            
        except Exception as e:
            logger.error(f"Failed to load {engine_type} engine: {e}")
            return None
    
    def _update_usage_stats(self, engine_type: str, from_cache: bool = False, load_time: float = 0):
        """Update usage statistics."""
        if engine_type not in self._usage_stats:
            self._usage_stats[engine_type] = {
                "total_uses": 0,
                "cache_hits": 0,
                "total_load_time": 0,
                "last_used": None
            }
        
        stats = self._usage_stats[engine_type]
        stats["total_uses"] += 1
        stats["last_used"] = datetime.now()
        
        if from_cache:
            stats["cache_hits"] += 1
        else:
            stats["total_load_time"] += load_time
    
    def get_preferred_engine(self, task_complexity: str = "medium") -> str:
        """
        Get the preferred engine based on task complexity and availability.
        
        Args:
            task_complexity: 'simple', 'medium', or 'complex'
            
        Returns:
            Preferred engine type
        """
        # Intelligent engine selection based on complexity
        if task_complexity == "simple":
            return "langgraph"  # Fast for simple tasks
        elif task_complexity == "complex":
            return "enhanced"   # Best for complex workflows
        else:
            return "enhanced"   # Default to enhanced for medium tasks
    
    def preload_critical_engines(self):
        """Preload the most critical engines."""
        logger.info("🔄 Preloading critical engines...")
        
        # Load enhanced engine (most capable)
        self.get_engine("enhanced")
        
        logger.info("✅ Critical engines preloaded")
    
    def load_primary_engine(self) -> bool:
        """
        Load the primary Enhanced Workflow Engine.
        """
        if self._primary_engine is not None:
            return True
            
        try:
            start_time = time.time()
            from core.orchestration.workflow_engine import EnhancedWorkflowEngine
            self._primary_engine = EnhancedWorkflowEngine()
            self._load_time = time.time() - start_time
            logger.info(f"✅ Enhanced Workflow Engine loaded in {self._load_time:.3f}s")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Enhanced Workflow Engine: {e}")
            return False
    
    def load_browser_tool(self) -> bool:
        """
        Load the primary Browser Use tool.
        """
        if self._browser_tool is not None:
            return True
            
        try:
            from tools.implementations.browser_automation_tool import BrowserAutomationTool
            self._browser_tool = BrowserAutomationTool()
            logger.info("✅ Browser Use tool loaded")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load Browser Use tool: {e}")
            return False
    
    def get_primary_engine(self):
        """Get the primary Enhanced Workflow Engine."""
        if self._primary_engine is None:
            self.load_primary_engine()
        self._usage_count += 1
        return self._primary_engine
    
    def get_browser_tool(self):
        """Get the primary Browser Use tool."""
        if self._browser_tool is None:
            self.load_browser_tool()
        return self._browser_tool
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get simplified performance statistics."""
        return {
            "primary_engine_loaded": self._primary_engine is not None,
            "browser_tool_loaded": self._browser_tool is not None,
            "load_time": self._load_time,
            "usage_count": self._usage_count,
            "system_status": "simplified"
        }

# Global simplified engine manager instance
simplified_engine_manager = SimplifiedEngineManager()
