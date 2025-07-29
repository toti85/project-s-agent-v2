#!/usr/bin/env python3
"""
Overseer Integration Module
Integrates OverseerAgent with Project-S V2 system
"""

import asyncio
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

# Import Project-S V2 components
try:
    from core.agents.overseer_agent import OverseerAgent
    from core.agents.pattern_analyzer import PatternAnalyzer
    from core.agents.optimization_engine import OptimizationEngine
    # EventBus may not exist yet, so we'll make it optional
    EventBus = None
except ImportError as e:
    logging.warning(f"[OVERSEER_INTEGRATION] Import warning: {e}")
    EventBus = None

logger = logging.getLogger(__name__)

class OverseerIntegration:
    """
    Integrates OverseerAgent with Project-S V2 system
    """
    
    def __init__(self, event_bus: Optional[Any] = None):
        self.event_bus = event_bus
        self.overseer_agent = None
        self.pattern_analyzer = None
        self.optimization_engine = None
        self.is_running = False
        
    async def initialize(self) -> bool:
        """Initialize the overseer integration"""
        try:
            logger.info("[OVERSEER_INTEGRATION] Initializing components...")
            
            # Initialize OverseerAgent
            self.overseer_agent = OverseerAgent()
            
            # Initialize Pattern Analyzer
            self.pattern_analyzer = PatternAnalyzer()
            
            # Initialize Optimization Engine
            self.optimization_engine = OptimizationEngine()
            
            # Subscribe to event bus if available
            if self.event_bus:
                await self.subscribe_to_events()
            
            logger.info("[OVERSEER_INTEGRATION] Components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Initialization error: {e}")
            return False
    
    async def subscribe_to_events(self):
        """Subscribe to Project-S V2 events"""
        try:
            if not self.event_bus:
                return
            
            # Subscribe to tool execution events
            await self.event_bus.subscribe("tool_execution_start", self.handle_tool_execution_start)
            await self.event_bus.subscribe("tool_execution_end", self.handle_tool_execution_end)
            await self.event_bus.subscribe("workflow_start", self.handle_workflow_start)
            await self.event_bus.subscribe("workflow_end", self.handle_workflow_end)
            await self.event_bus.subscribe("system_error", self.handle_system_error)
            
            logger.info("[OVERSEER_INTEGRATION] Event subscriptions established")
            
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Event subscription error: {e}")
    
    async def handle_tool_execution_start(self, event_data: Dict):
        """Handle tool execution start event"""
        try:
            tool_name = event_data.get("tool_name")
            context = event_data.get("context", {})
            
            if self.overseer_agent:
                await self.overseer_agent.handle_tool_execution({
                    "tool_name": tool_name,
                    "status": "started",
                    "context": context,
                    "timestamp": datetime.now()
                })
            
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Tool start handling error: {e}")
    
    async def handle_tool_execution_end(self, event_data: Dict):
        """Handle tool execution end event"""
        try:
            tool_name = event_data.get("tool_name")
            success = event_data.get("success", False)
            execution_time = event_data.get("execution_time", 0)
            context = event_data.get("context", {})
            
            if self.overseer_agent:
                await self.overseer_agent.handle_tool_execution({
                    "tool_name": tool_name,
                    "status": "completed",
                    "success": success,
                    "execution_time": execution_time,
                    "context": context,
                    "timestamp": datetime.now()
                })
            
            # Update pattern analyzer
            if self.pattern_analyzer:
                tool_sequence = context.get("tool_sequence", [tool_name])
                self.pattern_analyzer.analyze_execution_sequence(
                    tool_sequence, execution_time, success
                )
            
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Tool end handling error: {e}")
    
    async def handle_workflow_start(self, event_data: Dict):
        """Handle workflow start event"""
        try:
            workflow_id = event_data.get("workflow_id")
            workflow_type = event_data.get("workflow_type")
            
            logger.info(f"[OVERSEER_INTEGRATION] Workflow started: {workflow_id} ({workflow_type})")
            
            # Predict workflow outcome
            if self.pattern_analyzer:
                tools = event_data.get("tools", [])
                prediction = self.pattern_analyzer.predict_execution_outcome(tools)
                
                # Send prediction to event bus
                if self.event_bus:
                    await self.event_bus.publish("workflow_prediction", {
                        "workflow_id": workflow_id,
                        "prediction": prediction,
                        "timestamp": datetime.now()
                    })
            
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Workflow start handling error: {e}")
    
    async def handle_workflow_end(self, event_data: Dict):
        """Handle workflow end event"""
        try:
            workflow_id = event_data.get("workflow_id")
            success = event_data.get("success", False)
            execution_time = event_data.get("execution_time", 0)
            
            logger.info(f"[OVERSEER_INTEGRATION] Workflow completed: {workflow_id} (success: {success})")
            
            # Generate optimization suggestions
            if self.optimization_engine:
                suggestions = self.optimization_engine.generate_optimization_suggestions()
                
                if suggestions:
                    # Send suggestions to event bus
                    if self.event_bus:
                        await self.event_bus.publish("optimization_suggestions", {
                            "workflow_id": workflow_id,
                            "suggestions": [s.__dict__ for s in suggestions],
                            "timestamp": datetime.now()
                        })
            
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Workflow end handling error: {e}")
    
    async def handle_system_error(self, event_data: Dict):
        """Handle system error event"""
        try:
            error_type = event_data.get("error_type")
            error_message = event_data.get("error_message")
            context = event_data.get("context", {})
            
            logger.warning(f"[OVERSEER_INTEGRATION] System error: {error_type} - {error_message}")
            
            # Record error in overseer agent
            if self.overseer_agent:
                await self.overseer_agent.record_performance_metric(
                    "system_error", 1.0, json.dumps({
                        "error_type": error_type,
                        "error_message": error_message,
                        "context": context
                    })
                )
            
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Error handling error: {e}")
    
    async def start_monitoring(self):
        """Start continuous monitoring"""
        try:
            self.is_running = True
            logger.info("[OVERSEER_INTEGRATION] Starting continuous monitoring...")
            
            while self.is_running:
                # Get system health
                if self.overseer_agent:
                    health = self.overseer_agent.assess_system_health()
                    
                    # If health is poor, generate suggestions
                    if health.get("health_score", 0) < 0.7:
                        suggestions = self.optimization_engine.generate_optimization_suggestions()
                        
                        if suggestions and self.event_bus:
                            await self.event_bus.publish("health_alert", {
                                "health": health,
                                "suggestions": [s.__dict__ for s in suggestions],
                                "timestamp": datetime.now()
                            })
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Monitoring error: {e}")
        finally:
            self.is_running = False
    
    async def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_running = False
        logger.info("[OVERSEER_INTEGRATION] Monitoring stopped")
    
    async def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        try:
            status = {
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "overseer_agent": self.overseer_agent is not None,
                    "pattern_analyzer": self.pattern_analyzer is not None,
                    "optimization_engine": self.optimization_engine is not None,
                    "event_bus": self.event_bus is not None
                },
                "monitoring_active": self.is_running
            }
            
            # Add overseer agent status
            if self.overseer_agent:
                status["system_health"] = self.overseer_agent.assess_system_health()
            
            # Add pattern analyzer status
            if self.pattern_analyzer:
                status["pattern_insights"] = self.pattern_analyzer.get_pattern_insights()
            
            # Add optimization engine status
            if self.optimization_engine:
                status["optimization_stats"] = self.optimization_engine.get_optimization_stats()
            
            return status
            
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Status error: {e}")
            return {"error": str(e)}
    
    async def execute_optimization_suggestion(self, suggestion_id: str) -> bool:
        """Execute an optimization suggestion"""
        try:
            if not self.optimization_engine:
                return False
            
            # Get suggestion details
            suggestions = self.optimization_engine.get_active_suggestions()
            suggestion = next((s for s in suggestions if s.suggestion_id == suggestion_id), None)
            
            if not suggestion:
                return False
            
            # Mark as implemented (actual implementation would be more complex)
            result = self.optimization_engine.mark_suggestion_implemented(suggestion_id)
            
            if result and self.event_bus:
                await self.event_bus.publish("optimization_implemented", {
                    "suggestion_id": suggestion_id,
                    "suggestion_type": suggestion.type,
                    "expected_improvement": suggestion.expected_improvement,
                    "timestamp": datetime.now()
                })
            
            return result
            
        except Exception as e:
            logger.error(f"[OVERSEER_INTEGRATION] Execute suggestion error: {e}")
            return False

# Test function
async def test_overseer_integration():
    """Test the overseer integration"""
    print("[TEST] Testing Overseer Integration...")
    
    integration = OverseerIntegration()
    
    # Test initialization
    success = await integration.initialize()
    print(f"[TEST] Initialization: {success}")
    
    # Test status
    status = await integration.get_system_status()
    print(f"[TEST] System status: {status}")
    
    # Test event handling
    await integration.handle_tool_execution_start({
        "tool_name": "test_tool",
        "context": {"test": True}
    })
    
    await integration.handle_tool_execution_end({
        "tool_name": "test_tool",
        "success": True,
        "execution_time": 2.0,
        "context": {"test": True}
    })
    
    print("[TEST] Overseer Integration test completed")

if __name__ == "__main__":
    asyncio.run(test_overseer_integration())
