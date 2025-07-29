"""
Project-S v2 Architecture - Decision Routing Base
================================================
Modern decision routing framework providing flexible, pattern-aware
decision-making capabilities for workflow orchestration.

This module provides:
- Abstract decision routing interfaces
- Decision metadata and tracking
- Pattern recognition and learning
- Adaptive routing capabilities
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union, TypeVar
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)

# Type definitions
StateType = TypeVar('StateType', bound=Dict[str, Any])
DecisionCriteriaFunc = Callable[[StateType], Union[str, bool]]

class DecisionType(str, Enum):
    """Types of decisions that can be made."""
    ROUTING = "routing"
    CONDITIONAL = "conditional"
    PRIORITY = "priority"
    ADAPTIVE = "adaptive"
    COGNITIVE = "cognitive"

class DecisionStatus(str, Enum):
    """Status of a decision."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    OVERRIDDEN = "overridden"

@dataclass
class DecisionMetadata:
    """
    Metadata for tracking decisions made in workflows.
    """
    decision_id: str
    timestamp: float
    source_node: str
    decision_type: DecisionType
    decision_criteria: str
    considered_options: List[str]
    selected_option: str
    confidence_score: float
    context_snapshot: Dict[str, Any]
    status: DecisionStatus = DecisionStatus.PENDING
    execution_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionMetadata':
        """Create metadata from dictionary."""
        return cls(**data)

@dataclass
class DecisionPattern:
    """
    Represents a recognized pattern in decision making.
    """
    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    last_seen: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern to dictionary."""
        data = asdict(self)
        data['last_seen'] = self.last_seen.isoformat()
        return data

class DecisionResult:
    """
    Result of a decision operation.
    """
    def __init__(self, 
                 decision: str, 
                 confidence: float = 1.0, 
                 metadata: Optional[DecisionMetadata] = None,
                 reasoning: Optional[str] = None):
        self.decision = decision
        self.confidence = confidence
        self.metadata = metadata
        self.reasoning = reasoning
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata.to_dict() if self.metadata else None
        }

class BaseDecisionRouter(ABC):
    """
    Abstract base class for all decision routers in Project-S v2.
    
    Provides common functionality for:
    - Decision tracking and history
    - Pattern recognition
    - Criteria registration and management
    - Adaptive decision making
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the decision router.
        
        Args:
            name: Unique identifier for this router
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.decision_history: Dict[str, List[DecisionMetadata]] = {}
        self.criteria_registry: Dict[str, DecisionCriteriaFunc] = {}
        self.patterns: Dict[str, DecisionPattern] = {}
        self.created_at = datetime.now()
        
        # Initialize built-in criteria
        self._register_builtin_criteria()
        
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @abstractmethod
    async def make_decision(self, 
                          state: StateType, 
                          options: List[str], 
                          criteria: Optional[str] = None,
                          **kwargs) -> DecisionResult:
        """
        Make a decision based on state and available options.
        
        Args:
            state: Current state for decision making
            options: Available options to choose from
            criteria: Decision criteria to use
            **kwargs: Additional parameters
            
        Returns:
            DecisionResult containing the decision and metadata
        """
        pass
    
    def register_criteria(self, name: str, criteria_func: DecisionCriteriaFunc) -> None:
        """
        Register a named decision criteria function.
        
        Args:
            name: Name to register the criteria under
            criteria_func: Function that evaluates the criteria
        """
        self.criteria_registry[name] = criteria_func
        logger.info(f"Registered decision criteria '{name}' in router '{self.name}'")
    
    def get_criteria(self, name: str) -> Optional[DecisionCriteriaFunc]:
        """
        Get a registered decision criteria function.
        
        Args:
            name: Name of the criteria function
            
        Returns:
            The criteria function or None if not found
        """
        return self.criteria_registry.get(name)
    
    def list_criteria(self) -> List[str]:
        """Get list of all registered criteria names."""
        return list(self.criteria_registry.keys())
    
    def add_decision_to_history(self, 
                              workflow_id: str, 
                              metadata: DecisionMetadata) -> None:
        """
        Add a decision to the history for pattern analysis.
        
        Args:
            workflow_id: ID of the workflow
            metadata: Decision metadata to store
        """
        if workflow_id not in self.decision_history:
            self.decision_history[workflow_id] = []
        
        self.decision_history[workflow_id].append(metadata)
        
        # Trigger pattern analysis
        asyncio.create_task(self._analyze_patterns(workflow_id))
    
    async def _analyze_patterns(self, workflow_id: str) -> None:
        """
        Analyze decision patterns for a workflow.
        
        Args:
            workflow_id: ID of the workflow to analyze
        """
        if workflow_id not in self.decision_history:
            return
        
        decisions = self.decision_history[workflow_id]
        
        # Look for sequential patterns
        if len(decisions) >= 3:
            await self._detect_sequence_patterns(workflow_id, decisions[-3:])
        
        # Look for frequency patterns
        await self._detect_frequency_patterns(workflow_id, decisions)
    
    async def _detect_sequence_patterns(self, 
                                      workflow_id: str, 
                                      recent_decisions: List[DecisionMetadata]) -> None:
        """
        Detect sequential decision patterns.
        
        Args:
            workflow_id: Workflow ID
            recent_decisions: Recent decisions to analyze
        """
        sequence = " -> ".join([d.selected_option for d in recent_decisions])
        pattern_id = f"seq_{hash(sequence) % 10000}"
        
        if pattern_id in self.patterns:
            self.patterns[pattern_id].frequency += 1
            self.patterns[pattern_id].last_seen = datetime.now()
        else:
            self.patterns[pattern_id] = DecisionPattern(
                pattern_id=pattern_id,
                pattern_type="sequence",
                frequency=1,
                confidence=0.5,
                conditions={"sequence": sequence},
                outcomes={"workflow_id": workflow_id},
                last_seen=datetime.now()
            )
        
        logger.debug(f"Detected sequence pattern: {sequence}")
    
    async def _detect_frequency_patterns(self, 
                                       workflow_id: str, 
                                       decisions: List[DecisionMetadata]) -> None:
        """
        Detect frequency-based decision patterns.
        
        Args:
            workflow_id: Workflow ID
            decisions: All decisions to analyze
        """
        option_counts = {}
        for decision in decisions:
            option = decision.selected_option
            option_counts[option] = option_counts.get(option, 0) + 1
        
        # Find dominant patterns
        total_decisions = len(decisions)
        for option, count in option_counts.items():
            frequency = count / total_decisions
            
            if frequency > 0.7:  # Strong pattern threshold
                pattern_id = f"freq_{option}_{workflow_id}"
                
                if pattern_id in self.patterns:
                    self.patterns[pattern_id].frequency = count
                    self.patterns[pattern_id].confidence = frequency
                    self.patterns[pattern_id].last_seen = datetime.now()
                else:
                    self.patterns[pattern_id] = DecisionPattern(
                        pattern_id=pattern_id,
                        pattern_type="frequency",
                        frequency=count,
                        confidence=frequency,
                        conditions={"option": option},
                        outcomes={"workflow_id": workflow_id},
                        last_seen=datetime.now()
                    )
                
                logger.debug(f"Detected frequency pattern: {option} ({frequency:.2%})")
    
    def get_pattern_suggestion(self, state: StateType) -> Optional[str]:
        """
        Get a decision suggestion based on recognized patterns.
        
        Args:
            state: Current state
            
        Returns:
            Suggested decision option or None
        """
        # Find patterns that match current context
        context_keys = set(state.keys()) if isinstance(state, dict) else set()
        
        best_pattern = None
        best_confidence = 0.0
        
        for pattern in self.patterns.values():
            # Simple context matching - can be enhanced
            if (pattern.confidence > best_confidence and 
                pattern.confidence > 0.8):  # High confidence threshold
                best_pattern = pattern
                best_confidence = pattern.confidence
        
        if best_pattern:
            if best_pattern.pattern_type == "frequency":
                return best_pattern.conditions.get("option")
            elif best_pattern.pattern_type == "sequence":
                # For sequence patterns, predict next step
                # This is a simplified implementation
                sequence = best_pattern.conditions.get("sequence", "")
                if " -> " in sequence:
                    last_option = sequence.split(" -> ")[-1]
                    return last_option
        
        return None
    
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about decisions made by this router."""
        total_decisions = sum(len(decisions) for decisions in self.decision_history.values())
        
        return {
            "router_name": self.name,
            "total_decisions": total_decisions,
            "workflows_processed": len(self.decision_history),
            "registered_criteria": len(self.criteria_registry),
            "detected_patterns": len(self.patterns),
            "created_at": self.created_at.isoformat(),
            "config": self.config
        }
    
    def _register_builtin_criteria(self) -> None:
        """Register built-in decision criteria."""
        
        def has_error(state: StateType) -> bool:
            """Check if state contains error information."""
            if isinstance(state, dict):
                return "error" in state or "error_info" in state
            return False
        
        def is_complex_task(state: StateType) -> bool:
            """Check if task is complex based on input length."""
            if isinstance(state, dict):
                input_text = state.get("input", "")
                return len(str(input_text).split()) > 50
            return False
        
        def has_tools_needed(state: StateType) -> bool:
            """Check if task requires tool usage."""
            if isinstance(state, dict):
                input_text = str(state.get("input", "")).lower()
                tool_indicators = ["file", "create", "write", "run", "execute", "system"]
                return any(indicator in input_text for indicator in tool_indicators)
            return False
        
        def is_creative_task(state: StateType) -> bool:
            """Check if task is creative in nature."""
            if isinstance(state, dict):
                input_text = str(state.get("input", "")).lower()
                creative_indicators = ["write", "story", "creative", "imagine", "design"]
                return any(indicator in input_text for indicator in creative_indicators)
            return False
        
        # Register built-in criteria
        self.register_criteria("has_error", has_error)
        self.register_criteria("is_complex_task", is_complex_task)
        self.register_criteria("has_tools_needed", has_tools_needed)
        self.register_criteria("is_creative_task", is_creative_task)

class DecisionRouterRegistry:
    """
    Registry for managing multiple decision routers.
    """
    
    def __init__(self):
        """Initialize the router registry."""
        self._routers: Dict[str, BaseDecisionRouter] = {}
        self._default_router: Optional[str] = None
        
        logger.info("Decision router registry initialized")
    
    def register_router(self, router: BaseDecisionRouter, is_default: bool = False) -> None:
        """
        Register a decision router.
        
        Args:
            router: The router to register
            is_default: Whether this should be the default router
        """
        self._routers[router.name] = router
        
        if is_default or not self._default_router:
            self._default_router = router.name
            
        logger.info(f"Registered decision router: {router.name} (default: {is_default})")
    
    def get_router(self, name: Optional[str] = None) -> Optional[BaseDecisionRouter]:
        """
        Get a router by name.
        
        Args:
            name: Router name, or None for default
            
        Returns:
            The router or None if not found
        """
        if name is None:
            name = self._default_router
            
        return self._routers.get(name) if name else None
    
    def list_routers(self) -> List[str]:
        """Get list of all registered router names."""
        return list(self._routers.keys())
    
    async def make_decision(self, 
                          router_name: Optional[str],
                          state: StateType,
                          options: List[str],
                          criteria: Optional[str] = None,
                          **kwargs) -> Optional[DecisionResult]:
        """
        Make a decision using a specific router.
        
        Args:
            router_name: Name of router to use (None for default)
            state: Current state
            options: Available options
            criteria: Decision criteria
            **kwargs: Additional parameters
            
        Returns:
            DecisionResult or None if router not found
        """
        router = self.get_router(router_name)
        
        if not router:
            logger.error(f"Decision router not found: {router_name}")
            return None
        
        return await router.make_decision(state, options, criteria, **kwargs)

# Global router registry instance
router_registry = DecisionRouterRegistry()
