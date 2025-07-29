#!/usr/bin/env python3
"""
OverseerAgent - Intelligent AI Monitoring and Optimization System
Adaptív AI megfigyelő rendszer amely intelligensen tanul és proaktívan optimalizál.

[CORE INTELLIGENCE FEATURES]:
- Pattern Recognition & Learning
- Proactive Decision Making  
- Advanced Monitoring Beyond Basic Events
- Workflow Optimization
- Predictive Analytics
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Set
from pathlib import Path
from collections import defaultdict, deque
import logging
import sqlite3
from dataclasses import dataclass, asdict

# Add project root to path
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

from core.infrastructure.event_bus import EventBus
from utils.performance_monitor import EnhancedLogger, monitor_performance

@dataclass
class WorkflowPattern:
    """Workflow minta adatstruktúra"""
    pattern_id: str
    tool_sequence: List[str]
    success_rate: float
    avg_execution_time: float
    frequency: int
    last_used: datetime
    optimization_score: float

@dataclass
class PerformanceMetric:
    """Teljesítmény metrika"""
    timestamp: datetime
    metric_type: str
    value: float
    context: Dict[str, Any]

@dataclass
class OptimizationSuggestion:
    """Optimalizálási javaslat"""
    suggestion_id: str
    type: str  # 'tool_order', 'resource_allocation', 'timing', 'alternative_approach'
    description: str
    expected_improvement: float
    confidence: float
    implementation_cost: float
    created_at: datetime

class OverseerAgent:
    """
    Intelligent AI Monitoring and Optimization System
    
    [CORE CAPABILITIES]:
    - Pattern Recognition & Learning
    - Proactive Decision Making
    - Advanced Performance Monitoring
    - Workflow Optimization
    - Predictive Analytics
    """
    
    def __init__(self, event_bus: EventBus, memory_db_path: str = "overseer_memory.db"):
        """Initialize the OverseerAgent"""
        self.event_bus = event_bus
        self.logger = EnhancedLogger("overseer_agent")
        self.memory_db_path = memory_db_path
        
        # Intelligence modules
        self.workflow_patterns: Dict[str, WorkflowPattern] = {}
        self.performance_history: deque = deque(maxlen=10000)
        self.optimization_suggestions: List[OptimizationSuggestion] = []
        
        # Learning parameters
        self.learning_enabled = True
        self.prediction_accuracy = 0.0
        self.optimization_success_rate = 0.0
        
        # Initialize memory database
        self._init_memory_database()
        
        # Subscribe to events
        self._subscribe_to_events()
        
        self.logger.info("OverseerAgent initialized with intelligence modules")
    
    def _init_memory_database(self):
        """Initialize SQLite database for persistent learning"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            # Workflow patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS workflow_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    tool_sequence TEXT,
                    success_rate REAL,
                    avg_execution_time REAL,
                    frequency INTEGER,
                    last_used TIMESTAMP,
                    optimization_score REAL
                )
            ''')
            
            # Performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    metric_type TEXT,
                    value REAL,
                    context TEXT
                )
            ''')
            
            # Optimization suggestions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_suggestions (
                    suggestion_id TEXT PRIMARY KEY,
                    type TEXT,
                    description TEXT,
                    expected_improvement REAL,
                    confidence REAL,
                    implementation_cost REAL,
                    created_at TIMESTAMP,
                    status TEXT DEFAULT 'pending'
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("Memory database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize memory database: {e}")
    
    def _subscribe_to_events(self):
        """Subscribe to relevant events for monitoring and learning"""
        # Core events
        self.event_bus.subscribe("tool.execution.started", self._on_tool_execution_started)
        self.event_bus.subscribe("tool.execution.completed", self._on_tool_execution_completed)
        self.event_bus.subscribe("workflow.started", self._on_workflow_started)
        self.event_bus.subscribe("workflow.completed", self._on_workflow_completed)
        
        # Performance events
        self.event_bus.subscribe("performance.metric", self._on_performance_metric)
        self.event_bus.subscribe("resource.usage", self._on_resource_usage)
        
        # Error events
        self.event_bus.subscribe("execution.failed", self._on_execution_failed)
        self.event_bus.subscribe("error.occurred", self._on_error_occurred)
        
        # Optimization events
        self.event_bus.subscribe("optimization.requested", self._on_optimization_requested)
        
        self.logger.info("Event subscriptions configured")
    
    # ===========================================
    # PATTERN RECOGNITION & LEARNING
    # ===========================================
    
    async def _on_tool_execution_started(self, event_data: Dict[str, Any]):
        """Handle tool execution start event"""
        try:
            tool_name = event_data.get('tool_name')
            params = event_data.get('params', {})
            
            self.logger.debug(f"[TOOL] Tool execution started: {tool_name}")
            
            # Start tracking execution pattern
            await self._track_execution_pattern(tool_name, params)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling tool execution start: {e}")
    
    async def _on_tool_execution_completed(self, event_data: Dict[str, Any]):
        """Handle tool execution completion event"""
        try:
            tool_name = event_data.get('tool_name')
            execution_time = event_data.get('execution_time', 0)
            success = event_data.get('success', False)
            
            self.logger.debug(f"[SUCCESS] Tool execution completed: {tool_name} ({execution_time:.2f}s)")
            
            # Update performance metrics
            await self._update_performance_metrics(tool_name, execution_time, success)
            
            # Learn from execution
            await self._learn_from_execution(tool_name, execution_time, success, event_data)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling tool execution completion: {e}")
    
    async def _track_execution_pattern(self, tool_name: str, params: Dict[str, Any]):
        """Track execution patterns for learning"""
        try:
            # Implementation for pattern tracking
            pattern_key = f"{tool_name}_{hash(str(sorted(params.items())))}"
            
            # Store pattern information
            # This would be expanded with actual pattern analysis
            self.logger.debug(f"[PATTERN] Tracking pattern: {pattern_key}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error tracking execution pattern: {e}")
    
    async def _update_performance_metrics(self, tool_name: str, execution_time: float, success: bool):
        """Update performance metrics"""
        try:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type="execution_time",
                value=execution_time,
                context={"tool_name": tool_name, "success": success}
            )
            
            self.performance_history.append(metric)
            
            # Persist to database
            await self._persist_performance_metric(metric)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error updating performance metrics: {e}")
    
    async def _learn_from_execution(self, tool_name: str, execution_time: float, success: bool, event_data: Dict[str, Any]):
        """Learn from execution results"""
        try:
            if not self.learning_enabled:
                return
            
            # Analyze execution for learning opportunities
            await self._analyze_execution_for_patterns(tool_name, execution_time, success, event_data)
            
            # Generate optimization suggestions if needed
            await self._generate_optimization_suggestions(tool_name, execution_time, success)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error learning from execution: {e}")
    
    # ===========================================
    # PROACTIVE DECISION MAKING
    # ===========================================
    
    async def _on_workflow_started(self, event_data: Dict[str, Any]):
        """Handle workflow start - proactive optimization opportunity"""
        try:
            workflow_id = event_data.get('workflow_id')
            
            # Proactive optimization
            await self._suggest_workflow_optimizations(workflow_id, event_data)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling workflow start: {e}")
    
    async def _suggest_workflow_optimizations(self, workflow_id: str, workflow_data: Dict[str, Any]):
        """Suggest proactive workflow optimizations"""
        try:
            # Analyze similar past workflows
            similar_patterns = await self._find_similar_workflow_patterns(workflow_data)
            
            for pattern in similar_patterns:
                if pattern.optimization_score > 0.7:
                    suggestion = OptimizationSuggestion(
                        suggestion_id=f"workflow_{workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        type="workflow_optimization",
                        description=f"Based on similar workflows, consider using tool sequence: {pattern.tool_sequence}",
                        expected_improvement=pattern.optimization_score,
                        confidence=0.8,
                        implementation_cost=0.2,
                        created_at=datetime.now()
                    )
                    
                    self.optimization_suggestions.append(suggestion)
                    
                    # Publish proactive suggestion
                    await self.event_bus.publish("optimization.suggestion", {
                        "suggestion": asdict(suggestion),
                        "workflow_id": workflow_id
                    })
                    
                    self.logger.info(f"Proactive optimization suggested for workflow {workflow_id}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error suggesting workflow optimizations: {e}")
    
    async def _on_workflow_completed(self, event_data: Dict[str, Any]):
        """Handle workflow completion event"""
        try:
            workflow_id = event_data.get('workflow_id')
            success = event_data.get('success', False)
            execution_time = event_data.get('execution_time', 0)
            
            self.logger.debug(f"Workflow completed: {workflow_id} (success: {success})")
            
            # Learn from workflow completion
            await self._learn_from_workflow_completion(workflow_id, success, execution_time, event_data)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling workflow completion: {e}")
    
    async def _learn_from_workflow_completion(self, workflow_id: str, success: bool, execution_time: float, event_data: Dict[str, Any]):
        """Learn from workflow completion"""
        try:
            # Implementation for learning from workflow completion
            self.logger.debug(f"Learning from workflow completion: {workflow_id}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error learning from workflow completion: {e}")
    
    # ===========================================
    # RESOURCE USAGE HANDLING
    # ===========================================
    
    async def _on_resource_usage(self, event_data: Dict[str, Any]):
        """Handle resource usage events"""
        try:
            resource_type = event_data.get('resource_type')
            usage = event_data.get('usage', 0)
            
            self.logger.debug(f"Resource usage: {resource_type} = {usage}")
            
            # Track resource usage patterns
            await self._track_resource_usage(resource_type, usage, event_data)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling resource usage: {e}")
    
    async def _track_resource_usage(self, resource_type: str, usage: float, event_data: Dict[str, Any]):
        """Track resource usage patterns"""
        try:
            # Implementation for tracking resource usage
            self.logger.debug(f"Tracking resource usage: {resource_type}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error tracking resource usage: {e}")
    
    # ===========================================
    # ADVANCED MONITORING
    # ===========================================
    
    async def _on_performance_metric(self, event_data: Dict[str, Any]):
        """Handle performance metrics"""
        try:
            metric_type = event_data.get('metric_type')
            value = event_data.get('value')
            
            # Advanced performance analysis
            await self._analyze_performance_trends(metric_type, value, event_data)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling performance metric: {e}")
    
    async def _analyze_performance_trends(self, metric_type: str, value: float, context: Dict[str, Any]):
        """Analyze performance trends for anomaly detection"""
        try:
            # Get recent metrics of same type
            recent_metrics = [m for m in self.performance_history 
                           if m.metric_type == metric_type and 
                           m.timestamp > datetime.now() - timedelta(hours=1)]
            
            if len(recent_metrics) < 5:
                return
            
            # Calculate trend
            values = [m.value for m in recent_metrics]
            avg_value = sum(values) / len(values)
            
            # Detect anomalies
            if abs(value - avg_value) > avg_value * 0.5:  # 50% deviation
                await self.event_bus.publish("performance.anomaly", {
                    "metric_type": metric_type,
                    "current_value": value,
                    "average_value": avg_value,
                    "deviation": abs(value - avg_value) / avg_value,
                    "context": context
                })
                
                self.logger.warning(f"[WARNING] Performance anomaly detected: {metric_type} = {value} (avg: {avg_value:.2f})")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error analyzing performance trends: {e}")
    
    # ===========================================
    # PREDICTIVE ANALYTICS
    # ===========================================
    
    async def predict_resource_usage(self, time_horizon: int = 3600) -> Dict[str, Any]:
        """Predict resource usage for the next time horizon (seconds)"""
        try:
            # Simple prediction based on recent trends
            recent_metrics = [m for m in self.performance_history 
                           if m.timestamp > datetime.now() - timedelta(hours=1)]
            
            if not recent_metrics:
                return {"prediction": "insufficient_data"}
            
            # Calculate trends
            cpu_metrics = [m for m in recent_metrics if m.metric_type == "cpu_usage"]
            memory_metrics = [m for m in recent_metrics if m.metric_type == "memory_usage"]
            
            prediction = {
                "time_horizon": time_horizon,
                "predicted_cpu": self._predict_metric_trend(cpu_metrics),
                "predicted_memory": self._predict_metric_trend(memory_metrics),
                "confidence": 0.7,
                "generated_at": datetime.now()
            }
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error predicting resource usage: {e}")
            return {"prediction": "error", "error": str(e)}
    
    def _predict_metric_trend(self, metrics: List[PerformanceMetric]) -> float:
        """Simple linear trend prediction"""
        if len(metrics) < 2:
            return 0.0
        
        values = [m.value for m in metrics]
        # Simple linear regression (slope)
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        if n * sum_x2 - sum_x * sum_x == 0:
            return values[-1]
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Predict next value
        return slope * n + intercept
    
    # ===========================================
    # UTILITY METHODS
    # ===========================================
    
    async def _find_similar_workflow_patterns(self, workflow_data: Dict[str, Any]) -> List[WorkflowPattern]:
        """Find similar workflow patterns from history"""
        try:
            # This would implement similarity matching
            # For now, return empty list
            return []
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error finding similar workflow patterns: {e}")
            return []
    
    async def _persist_performance_metric(self, metric: PerformanceMetric):
        """Persist performance metric to database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (timestamp, metric_type, value, context)
                VALUES (?, ?, ?, ?)
            ''', (metric.timestamp, metric.metric_type, metric.value, json.dumps(metric.context)))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error persisting performance metric: {e}")
    
    async def _analyze_execution_for_patterns(self, tool_name: str, execution_time: float, success: bool, event_data: Dict[str, Any]):
        """Analyze execution for patterns"""
        try:
            # Pattern analysis implementation
            pass
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error analyzing execution for patterns: {e}")
    
    async def _generate_optimization_suggestions(self, tool_name: str, execution_time: float, success: bool):
        """Generate optimization suggestions"""
        try:
            # Optimization suggestion generation
            pass
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error generating optimization suggestions: {e}")
    
    # ===========================================
    # ERROR HANDLING
    # ===========================================
    
    async def _on_execution_failed(self, event_data: Dict[str, Any]):
        """Handle execution failure events"""
        try:
            tool_name = event_data.get('tool_name')
            error = event_data.get('error')
            
            self.logger.warning(f"[WARNING] Execution failed: {tool_name} - {error}")
            
            # Learn from failure
            await self._learn_from_failure(tool_name, error, event_data)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling execution failure: {e}")
    
    async def _on_error_occurred(self, event_data: Dict[str, Any]):
        """Handle general error events"""
        try:
            error_type = event_data.get('error_type')
            error_message = event_data.get('error_message')
            
            self.logger.error(f"[ERROR] Error occurred: {error_type} - {error_message}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling error event: {e}")
    
    async def _learn_from_failure(self, tool_name: str, error: str, event_data: Dict[str, Any]):
        """Learn from execution failures"""
        try:
            # Failure analysis and learning
            pass
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error learning from failure: {e}")
    
    # ===========================================
    # OPTIMIZATION REQUESTS
    # ===========================================
    
    async def _on_optimization_requested(self, event_data: Dict[str, Any]):
        """Handle optimization requests"""
        try:
            request_type = event_data.get('request_type')
            
            if request_type == "workflow_optimization":
                await self._handle_workflow_optimization_request(event_data)
            elif request_type == "resource_optimization":
                await self._handle_resource_optimization_request(event_data)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling optimization request: {e}")
    
    async def _handle_workflow_optimization_request(self, event_data: Dict[str, Any]):
        """Handle workflow optimization request"""
        try:
            # Workflow optimization logic
            pass
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling workflow optimization request: {e}")
    
    async def _handle_resource_optimization_request(self, event_data: Dict[str, Any]):
        """Handle resource optimization request"""
        try:
            # Resource optimization logic
            pass
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error handling resource optimization request: {e}")
    
    # ===========================================
    # PUBLIC API
    # ===========================================
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health assessment"""
        try:
            recent_metrics = [m for m in self.performance_history 
                           if m.timestamp > datetime.now() - timedelta(minutes=30)]
            
            if not recent_metrics:
                return {"health": "unknown", "reason": "insufficient_data"}
            
            # Calculate health score
            success_rate = len([m for m in recent_metrics if m.context.get('success', False)]) / len(recent_metrics)
            avg_execution_time = sum(m.value for m in recent_metrics if m.metric_type == "execution_time") / len(recent_metrics)
            
            health_score = success_rate * 0.7 + (1 - min(avg_execution_time / 10, 1)) * 0.3
            
            health_status = "excellent" if health_score > 0.9 else "good" if health_score > 0.7 else "fair" if health_score > 0.5 else "poor"
            
            return {
                "health": health_status,
                "health_score": health_score,
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "total_metrics": len(recent_metrics),
                "optimization_suggestions": len(self.optimization_suggestions),
                "learning_enabled": self.learning_enabled
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error getting system health: {e}")
            return {"health": "error", "error": str(e)}
    
    async def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """Get current optimization suggestions"""
        try:
            return [asdict(suggestion) for suggestion in self.optimization_suggestions]
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error getting optimization suggestions: {e}")
            return []
    
    async def enable_learning(self, enabled: bool = True):
        """Enable/disable learning functionality"""
        self.learning_enabled = enabled
        self.logger.info(f"[BRAIN] Learning {'enabled' if enabled else 'disabled'}")
    
    async def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and trends"""
        try:
            recent_metrics = [m for m in self.performance_history 
                           if m.timestamp > datetime.now() - timedelta(hours=1)]
            
            if not recent_metrics:
                return {"insights": "insufficient_data"}
            
            # Group by metric type
            metric_groups = defaultdict(list)
            for metric in recent_metrics:
                metric_groups[metric.metric_type].append(metric.value)
            
            insights = {}
            for metric_type, values in metric_groups.items():
                insights[metric_type] = {
                    "count": len(values),
                    "average": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "trend": "stable"  # Would implement trend analysis
                }
            
            return {
                "insights": insights,
                "total_metrics": len(recent_metrics),
                "time_range": "1_hour",
                "generated_at": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error getting performance insights: {e}")
            return {"insights": "error", "error": str(e)}

# ===========================================
# FACTORY FUNCTION
# ===========================================

def create_overseer_agent(event_bus: EventBus) -> OverseerAgent:
    """Factory function to create OverseerAgent instance"""
    return OverseerAgent(event_bus)

# ===========================================
# MAIN EXECUTION
# ===========================================

if __name__ == "__main__":
    # Test the OverseerAgent
    async def test_overseer():
        from core.infrastructure.event_bus import EventBus
        
        event_bus = EventBus()
        overseer = create_overseer_agent(event_bus)
        
        # Test health check
        health = await overseer.get_system_health()
        print(f"System Health: {health}")
        
        # Test resource prediction
        prediction = await overseer.predict_resource_usage()
        print(f"Resource Prediction: {prediction}")
        
        # Test performance insights
        insights = await overseer.get_performance_insights()
        print(f"Performance Insights: {insights}")
    
    # Run test
    asyncio.run(test_overseer())
