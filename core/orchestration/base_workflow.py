"""
Project-S v2 Architecture - Base Workflow Engine
===============================================
Modern, modular workflow orchestration system that forms the foundation
for all LangGraph-based workflow operations in Project-S v2.

This module provides:
- Abstract workflow base classes
- Common workflow patterns and utilities
- State management interfaces
- Workflow lifecycle management
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, TypeVar, Generic
from enum import Enum
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Type definitions
StateType = TypeVar('StateType', bound=Dict[str, Any])
State = Dict[str, Any]

class WorkflowStatus(str, Enum):
    """Workflow execution status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskComplexity(str, Enum):
    """Task complexity levels for workflow routing."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

@dataclass
class WorkflowResult:
    """Standard workflow result container."""
    status: WorkflowStatus
    data: Dict[str, Any]
    metadata: Dict[str, Any]
    execution_time: float
    error: Optional[str] = None

class BaseWorkflowEngine(ABC, Generic[StateType]):
    """
    Abstract base class for all workflow engines in Project-S v2.
    
    Provides common functionality and interface for different workflow types:
    - Basic workflows
    - Multi-step workflows  
    - Multi-model workflows
    - Custom workflows
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the workflow engine.
        
        Args:
            name: Unique identifier for this workflow engine
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.status = WorkflowStatus.PENDING
        self.created_at = datetime.now()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.execution_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @abstractmethod
    async def execute(self, input_data: StateType, **kwargs) -> WorkflowResult:
        """
        Execute the workflow with the given input data.
        
        Args:
            input_data: Input state/data for the workflow
            **kwargs: Additional execution parameters
            
        Returns:
            WorkflowResult: The execution result
        """
        pass
    
    @abstractmethod
    def analyze_task(self, input_data: StateType) -> Dict[str, Any]:
        """
        Analyze the input task to determine execution strategy.
        
        Args:
            input_data: Input state/data to analyze
            
        Returns:
            Dict containing analysis results
        """
        pass
    
    def get_complexity(self, input_text: str) -> TaskComplexity:
        """
        Determine task complexity based on input characteristics.
        
        Args:
            input_text: The input text to analyze
            
        Returns:
            TaskComplexity enum value
        """
        # Basic heuristics for complexity analysis
        word_count = len(input_text.split())
        
        if word_count > 100:
            return TaskComplexity.COMPLEX
        elif word_count > 30 or any(keyword in input_text.lower() for keyword in 
                                   ["majd", "utána", "azután", "első", "második", "harmadik"]):
            return TaskComplexity.MEDIUM
        else:
            return TaskComplexity.SIMPLE
    
    def start_execution(self) -> None:
        """Mark workflow execution as started."""
        self.status = WorkflowStatus.RUNNING
        self.started_at = datetime.now()
        
        self.execution_history.append({
            "event": "execution_started",
            "timestamp": self.started_at.isoformat(),
            "status": self.status.value
        })
        
        logger.info(f"Workflow {self.name} execution started")
    
    def complete_execution(self, result: WorkflowResult) -> None:
        """Mark workflow execution as completed."""
        self.status = result.status
        self.completed_at = datetime.now()
        
        self.execution_history.append({
            "event": "execution_completed",
            "timestamp": self.completed_at.isoformat(),
            "status": self.status.value,
            "execution_time": result.execution_time
        })
        
        logger.info(f"Workflow {self.name} execution completed with status: {self.status}")
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get workflow execution metrics and statistics."""
        total_time = None
        if self.started_at and self.completed_at:
            total_time = (self.completed_at - self.started_at).total_seconds()
        
        return {
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_execution_time": total_time,
            "execution_events": len(self.execution_history),
            "config": self.config
        }

class WorkflowRegistry:
    """
    Registry for managing multiple workflow engines.
    Provides workflow discovery, registration, and lifecycle management.
    """
    
    def __init__(self):
        """Initialize the workflow registry."""
        self._workflows: Dict[str, BaseWorkflowEngine] = {}
        self._default_workflow: Optional[str] = None
        
        logger.info("Workflow registry initialized")
    
    def register_workflow(self, workflow: BaseWorkflowEngine, is_default: bool = False) -> None:
        """
        Register a new workflow engine.
        
        Args:
            workflow: The workflow engine to register
            is_default: Whether this should be the default workflow
        """
        self._workflows[workflow.name] = workflow
        
        if is_default or not self._default_workflow:
            self._default_workflow = workflow.name
            
        logger.info(f"Registered workflow: {workflow.name} (default: {is_default})")
    
    def get_workflow(self, name: Optional[str] = None) -> Optional[BaseWorkflowEngine]:
        """
        Get a workflow engine by name.
        
        Args:
            name: Workflow name, or None for default
            
        Returns:
            The workflow engine or None if not found
        """
        if name is None:
            name = self._default_workflow
            
        return self._workflows.get(name) if name else None
    
    def list_workflows(self) -> List[str]:
        """Get list of all registered workflow names."""
        return list(self._workflows.keys())
    
    def get_workflow_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific workflow."""
        workflow = self._workflows.get(name)
        return workflow.get_execution_metrics() if workflow else None
    
    async def execute_workflow(self, name: Optional[str], input_data: State, **kwargs) -> WorkflowResult:
        """
        Execute a workflow by name.
        
        Args:
            name: Workflow name (None for default)
            input_data: Input data for execution
            **kwargs: Additional execution parameters
            
        Returns:
            WorkflowResult: The execution result
        """
        workflow = self.get_workflow(name)
        
        if not workflow:
            error_msg = f"Workflow not found: {name}"
            logger.error(error_msg)
            return WorkflowResult(
                status=WorkflowStatus.FAILED,
                data={},
                metadata={"error": error_msg},
                execution_time=0.0,
                error=error_msg
            )
        
        return await workflow.execute(input_data, **kwargs)

# Global workflow registry instance
workflow_registry = WorkflowRegistry()
