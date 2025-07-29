"""
Project-S v2 Architecture - Cognitive-Enhanced Workflow Integration
==================================================================
Integration layer that connects the cognitive core with the orchestration system,
providing intelligent task analysis, routing, and execution.

This module creates a bridge between:
- Cognitive Core (intent analysis, task decomposition, reasoning)
- Orchestration Engine (LangGraph workflows, state management)
- Decision Routing (pattern-based routing decisions)

Key Features:
- Cognitive task analysis before workflow execution
- Intent-driven workflow selection
- Intelligent task decomposition with cognitive insights
- Context-aware processing with semantic reasoning
- Learning from workflow execution patterns
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
from pathlib import Path
import sys

# Add path for imports
v2_arch_path = Path(__file__).parent.parent
sys.path.insert(0, str(v2_arch_path))

from core.cognitive import (
    cognitive_registry,
    EnhancedCognitiveProcessor,
    AdvancedIntentAnalyzer,
    SemanticReasoningEngine,
    CognitiveContext,
    CognitiveResult,
    IntentMatch,
    TaskDecomposition
)
from core.orchestration import WorkflowRegistry, LangGraphWorkflowEngine, WorkflowResult, workflow_registry
from integrations.decision_routing import DecisionRouterRegistry, AdvancedDecisionRouter, router_registry
from core.state import PersistentStateManager, default_state_manager, StateType

logger = logging.getLogger(__name__)

class CognitiveWorkflowIntegration:
    """
    Integration layer that provides cognitive-enhanced workflow execution.
    
    This class orchestrates the interaction between cognitive analysis and
    workflow execution, providing intelligent task processing capabilities.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cognitive workflow integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Component registries (use global instances)
        self.cognitive_registry = cognitive_registry
        self.workflow_registry = workflow_registry
        self.router_registry = router_registry
        
        # Core components
        self.cognitive_processor: Optional[EnhancedCognitiveProcessor] = None
        self.intent_analyzer: Optional[AdvancedIntentAnalyzer] = None
        self.reasoning_engine: Optional[SemanticReasoningEngine] = None
        self.workflow_engine: Optional[LangGraphWorkflowEngine] = None
        self.decision_router: Optional[AdvancedDecisionRouter] = None
        self.state_manager: Optional[PersistentStateManager] = None
        
        # Performance tracking
        self.execution_metrics = {
            "total_tasks_processed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_processing_time": 0.0,
            "cognitive_insights_generated": 0
        }
        
        self._initialize_components()
        logger.info("Cognitive workflow integration initialized")
    
    def _initialize_components(self) -> None:
        """Initialize all required components."""
        try:
            # Initialize cognitive components using the registry's get methods
            self.cognitive_processor = self.cognitive_registry.get_processor("enhanced_cognitive_processor")
            self.intent_analyzer = self.cognitive_registry.get_analyzer("advanced_intent_analyzer")
            self.reasoning_engine = self.cognitive_registry.get_reasoning_engine("semantic_reasoning_engine")
            
            # Initialize workflow components
            self.workflow_engine = self.workflow_registry.get_workflow("langgraph_engine")
            
            # Initialize routing components
            self.decision_router = self.router_registry.get_router("advanced_router")
            
            # Initialize state management
            self.state_manager = default_state_manager
            
            # Connect dependencies
            if self.workflow_engine and self.state_manager:
                self.workflow_engine.set_dependencies(
                    model_manager=None,  # Will be set when available
                    state_manager=self.state_manager
                )
            
            logger.info("All integration components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize integration components: {e}")
            raise
    
    async def process_task_cognitively(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a task using cognitive-enhanced workflow execution.
        
        This is the main entry point that combines cognitive analysis
        with intelligent workflow execution.
        
        Args:
            task_description: Natural language description of the task
            context: Additional context information
            user_id: User identifier for personalization
            
        Returns:
            Dictionary containing execution results and cognitive insights
        """
        start_time = datetime.now()
        execution_id = f"exec_{int(start_time.timestamp())}"
        
        try:
            logger.info(f"Starting cognitive task processing: {execution_id}")
            
            # Phase 1: Cognitive Analysis
            cognitive_analysis = await self._perform_cognitive_analysis(
                task_description, context, user_id
            )
            
            # Phase 2: Workflow Selection and Configuration
            workflow_config = await self._select_and_configure_workflow(
                cognitive_analysis
            )
            
            # Phase 3: Execute Workflow with Cognitive Enhancement
            execution_result = await self._execute_cognitive_workflow(
                workflow_config, cognitive_analysis
            )
            
            # Phase 4: Post-processing and Learning
            final_result = await self._post_process_and_learn(
                execution_result, cognitive_analysis, start_time
            )
            
            # Update metrics
            self._update_metrics(start_time, True)
            
            logger.info(f"Cognitive task processing completed: {execution_id}")
            return final_result
            
        except Exception as e:
            logger.error(f"Cognitive task processing failed: {e}")
            self._update_metrics(start_time, False)
            
            return {
                "success": False,
                "error": str(e),
                "execution_id": execution_id,
                "timestamp": start_time.isoformat()
            }
    
    async def _perform_cognitive_analysis(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]],
        user_id: Optional[str]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive cognitive analysis of the task.
        
        Args:
            task_description: Task description
            context: Additional context
            user_id: User identifier
            
        Returns:
            Cognitive analysis results
        """
        logger.info("Performing cognitive analysis...")
        
        # Create processing context
        processing_context = CognitiveContext(
            session_id=f"session_{int(datetime.now().timestamp())}",
            metadata=context or {}
        )
        
        # Analyze intent
        intent_matches = await self.intent_analyzer.detect_intent(
            task_description, processing_context
        )
        intent_result = intent_matches[0] if intent_matches else IntentMatch(
            intent_type="general", confidence=0.5, operation="process"
        )
        
        # Decompose task using cognitive processor
        task_data = {
            "id": f"task_{int(datetime.now().timestamp())}",
            "description": task_description,
            "context": context or {},
            "metadata": {
                "user_id": user_id,
                "intent": intent_result.intent_type
            }
        }
        
        decomposition_result = await self.cognitive_processor.decompose_task(
            task_data, processing_context
        )
        
        # Perform semantic reasoning
        reasoning_premises = [
            {
                "type": "task_description",
                "content": task_description,
                "confidence": 1.0
            },
            {
                "type": "intent",
                "content": intent_result.intent_type,
                "confidence": intent_result.confidence
            },
            {
                "type": "context",
                "content": str(context or {}),
                "confidence": 0.8
            }
        ]
        
        reasoning_result = await self.reasoning_engine.reason(
            reasoning_premises, processing_context
        )
        
        return {
            "intent_analysis": {
                "intent_type": intent_result.intent_type,
                "confidence": intent_result.confidence,
                "operation": intent_result.operation,
                "parameters": intent_result.parameters,
                "suggested_approach": intent_result.operation
            },
            "task_decomposition": {
                "subtasks": [
                    {
                        "id": subtask.get("id", f"subtask_{i}"),
                        "description": subtask.get("description", ""),
                        "priority": subtask.get("priority", "normal"),
                        "complexity": subtask.get("complexity", "medium")
                    }
                    for i, subtask in enumerate(decomposition_result.subtasks)
                ],
                "total_complexity": decomposition_result.estimated_complexity,
                "execution_strategy": decomposition_result.execution_strategy
            },
            "semantic_reasoning": {
                "strategy": reasoning_result.get("strategy", "sequential_execution"),
                "confidence": reasoning_result.get("confidence", 0.7),
                "explanation": reasoning_result.get("explanation", "Standard reasoning applied"),
                "similar_patterns": reasoning_result.get("similar_patterns", [])
            },
            "processing_context": {
                "user_id": user_id or "anonymous",
                "session_id": processing_context.session_id,
                "task_type": "general"
            }
        }
    
    async def _select_and_configure_workflow(
        self, cognitive_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Select and configure the appropriate workflow based on cognitive analysis.
        
        Args:
            cognitive_analysis: Results from cognitive analysis
            
        Returns:
            Workflow configuration
        """
        logger.info("Selecting and configuring workflow...")
        
        # Extract key information
        intent_type = cognitive_analysis["intent_analysis"]["intent_type"]
        task_complexity = cognitive_analysis["task_decomposition"]["total_complexity"]
        reasoning_strategy = cognitive_analysis["semantic_reasoning"]["strategy"]
        
        # Use decision router to select workflow type
        routing_criteria = {
            "intent_type": intent_type,
            "complexity": task_complexity,
            "strategy": reasoning_strategy,
            "subtask_count": len(cognitive_analysis["task_decomposition"]["subtasks"])
        }
        
        # Prepare options and criteria for the decision router
        workflow_options = ["basic", "multi_step", "multi_model"]
        decision_criteria = f"complexity_{task_complexity}_intent_{intent_type}"
        
        routing_decision = await self.decision_router.make_decision(
            StateType.WORKFLOW,
            workflow_options,
            decision_criteria,
            **routing_criteria
        )
        
        # Configure workflow based on decision
        workflow_type = routing_decision.decision if hasattr(routing_decision, 'decision') else "basic"
        
        # Map cognitive insights to workflow parameters
        workflow_config = {
            "type": workflow_type,
            "parameters": {
                "task_complexity": "complex" if task_complexity > 3 else "medium" if task_complexity > 1 else "simple",
                "subtasks": cognitive_analysis["task_decomposition"]["subtasks"],
                "intent_guidance": cognitive_analysis["intent_analysis"]["suggested_approach"],
                "reasoning_strategy": reasoning_strategy
            },
            "routing_decision": routing_decision,
            "cognitive_insights": cognitive_analysis
        }
        
        logger.info(f"Selected workflow type: {workflow_type} with complexity: {workflow_config['parameters']['task_complexity']}")
        return workflow_config
    
    async def _execute_cognitive_workflow(
        self, workflow_config: Dict[str, Any], cognitive_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute the selected workflow with cognitive enhancement.
        
        Args:
            workflow_config: Workflow configuration
            cognitive_analysis: Cognitive analysis results
            
        Returns:
            Workflow execution results
        """
        logger.info("Executing cognitive-enhanced workflow...")
        
        # Prepare workflow parameters with cognitive insights
        workflow_params = {
            "task": cognitive_analysis["processing_context"],
            "cognitive_guidance": {
                "intent": cognitive_analysis["intent_analysis"]["intent_type"],
                "strategy": cognitive_analysis["semantic_reasoning"]["strategy"],
                "subtasks": cognitive_analysis["task_decomposition"]["subtasks"]
            },
            **workflow_config["parameters"]
        }
        
        # Execute workflow using the orchestration engine
        workflow_result = await self.workflow_engine.execute(
            workflow_params, workflow_config["type"]
        )
        
        return {
            "workflow_execution": workflow_result,
            "cognitive_enhancement": {
                "applied_insights": True,
                "strategy_used": cognitive_analysis["semantic_reasoning"]["strategy"],
                "intent_guided": cognitive_analysis["intent_analysis"]["intent_type"]
            }
        }
    
    async def _post_process_and_learn(
        self,
        execution_result: Dict[str, Any],
        cognitive_analysis: Dict[str, Any],
        start_time: datetime
    ) -> Dict[str, Any]:
        """
        Post-process results and extract learning insights.
        
        Args:
            execution_result: Workflow execution results
            cognitive_analysis: Cognitive analysis results
            start_time: Execution start time
            
        Returns:
            Final processed results with learning insights
        """
        logger.info("Post-processing and extracting learning insights...")
        
        execution_duration = (datetime.now() - start_time).total_seconds()
        
        # Extract learning patterns
        learning_insights = {
            "execution_pattern": {
                "intent_type": cognitive_analysis["intent_analysis"]["intent_type"],
                "complexity": cognitive_analysis["task_decomposition"]["total_complexity"],
                "strategy_effectiveness": execution_result["workflow_execution"].get("success", False),
                "duration": execution_duration
            },
            "cognitive_accuracy": {
                "intent_confidence": cognitive_analysis["intent_analysis"]["confidence"],
                "reasoning_confidence": cognitive_analysis["semantic_reasoning"]["confidence"],
                "strategy_alignment": execution_result["cognitive_enhancement"]["strategy_used"] == cognitive_analysis["semantic_reasoning"]["strategy"]
            }
        }
        
        # Store learning data (if state manager available)
        if self.state_manager:
            await self.state_manager.create_state(
                StateType.CONTEXT,
                learning_insights,
                f"learning_pattern_{int(start_time.timestamp())}"
            )
        
        # Update cognitive insights counter
        self.execution_metrics["cognitive_insights_generated"] += 1
        
        return {
            "success": True,
            "results": execution_result,
            "cognitive_analysis": cognitive_analysis,
            "learning_insights": learning_insights,
            "execution_metrics": {
                "duration_seconds": execution_duration,
                "cognitive_enhancement_applied": True,
                "insights_generated": 1
            },
            "timestamp": start_time.isoformat()
        }
    
    def _update_metrics(self, start_time: datetime, success: bool) -> None:
        """Update execution metrics."""
        duration = (datetime.now() - start_time).total_seconds()
        
        self.execution_metrics["total_tasks_processed"] += 1
        
        if success:
            self.execution_metrics["successful_executions"] += 1
        else:
            self.execution_metrics["failed_executions"] += 1
        
        # Update average processing time
        total_tasks = self.execution_metrics["total_tasks_processed"]
        current_avg = self.execution_metrics["average_processing_time"]
        self.execution_metrics["average_processing_time"] = (
            (current_avg * (total_tasks - 1) + duration) / total_tasks
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return {
            **self.execution_metrics,
            "success_rate": (
                self.execution_metrics["successful_executions"] / 
                max(self.execution_metrics["total_tasks_processed"], 1)
            ) * 100
        }

# Global integration instance
_cognitive_workflow_integration: Optional[CognitiveWorkflowIntegration] = None

def get_cognitive_workflow_integration(config: Optional[Dict[str, Any]] = None) -> CognitiveWorkflowIntegration:
    """
    Get or create the global cognitive workflow integration instance.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CognitiveWorkflowIntegration instance
    """
    global _cognitive_workflow_integration
    
    if _cognitive_workflow_integration is None:
        _cognitive_workflow_integration = CognitiveWorkflowIntegration(config)
    
    return _cognitive_workflow_integration

# Convenience function for direct task processing
async def process_task_with_cognitive_enhancement(
    task_description: str,
    context: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Convenience function for processing tasks with cognitive enhancement.
    
    Args:
        task_description: Natural language description of the task
        context: Additional context information
        user_id: User identifier
        config: Configuration for the integration
        
    Returns:
        Processed task results with cognitive insights
    """
    integration = get_cognitive_workflow_integration(config)
    return await integration.process_task_cognitively(task_description, context, user_id)
