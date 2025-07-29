"""
PROJECT-S V2 - Base Cognitive Architecture
==========================================

This module provides the foundational cognitive architecture for the v2 system,
including abstract base classes for task processing, intent analysis, and reasoning.

Key Features:
1. Abstract cognitive processor for task understanding and decomposition
2. Intent detection and confidence scoring framework
3. Context management and learning capabilities
4. Semantic understanding integration points
5. Pattern recognition and analysis
6. Extensible reasoning engine framework

Architecture Benefits:
- Clean separation of cognitive concerns
- Pluggable reasoning modules
- Enhanced testability and modularity
- Unified cognitive interfaces
- Scalable processing pipeline
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Set, Union, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CognitiveState(Enum):
    """States of cognitive processing"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    REASONING = "reasoning"
    EXECUTING = "executing"
    LEARNING = "learning"
    ERROR = "error"


class ProcessingPriority(Enum):
    """Task processing priorities"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class CognitiveContext:
    """Context for cognitive operations"""
    session_id: str
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    active_tasks: Set[str] = field(default_factory=set)
    completed_tasks: Set[str] = field(default_factory=set)
    entities: Dict[str, Any] = field(default_factory=dict)
    workspace: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class IntentMatch:
    """Represents detected intent with confidence and metadata"""
    intent_type: str
    confidence: float
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    patterns_matched: List[str] = field(default_factory=list)
    semantic_similarity: Optional[float] = None
    context_boost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def total_confidence(self) -> float:
        """Calculate total confidence including context boost"""
        return min(1.0, self.confidence + self.context_boost)


@dataclass
class TaskDecomposition:
    """Result of task decomposition"""
    task_id: str
    subtasks: List[Dict[str, Any]]
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    execution_strategy: str = "sequential"
    estimated_complexity: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CognitiveResult:
    """Result of cognitive processing"""
    success: bool
    result_type: str
    data: Any = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class BaseCognitiveProcessor(ABC):
    """
    Abstract base class for cognitive processors.
    
    Cognitive processors handle high-level understanding, reasoning,
    and decision-making within the Project-S system.
    """
    
    def __init__(self, processor_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cognitive processor.
        
        Args:
            processor_id: Unique identifier for this processor
            config: Configuration parameters
        """
        self.processor_id = processor_id
        self.config = config or {}
        self.state = CognitiveState.IDLE
        self.context = None
        self.metrics = {
            "tasks_processed": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_confidence": 0.0,
            "total_processing_time": 0.0
        }
        
        logger.info(f"Cognitive processor '{processor_id}' initialized")
    
    @abstractmethod
    async def process_task(self, task: Dict[str, Any], context: CognitiveContext) -> CognitiveResult:
        """
        Process a high-level task using cognitive capabilities.
        
        Args:
            task: Task specification to process
            context: Current cognitive context
            
        Returns:
            Result of cognitive processing
        """
        pass
    
    @abstractmethod
    async def analyze_intent(self, input_text: str, context: CognitiveContext) -> List[IntentMatch]:
        """
        Analyze user input to detect intents and extract parameters.
        
        Args:
            input_text: User input to analyze
            context: Current cognitive context
            
        Returns:
            List of detected intents with confidence scores
        """
        pass
    
    @abstractmethod
    async def decompose_task(self, task: Dict[str, Any], context: CognitiveContext) -> TaskDecomposition:
        """
        Decompose a complex task into manageable subtasks.
        
        Args:
            task: Task to decompose
            context: Current cognitive context
            
        Returns:
            Task decomposition with subtasks and dependencies
        """
        pass
    
    async def update_context(self, context: CognitiveContext, updates: Dict[str, Any]) -> None:
        """
        Update the cognitive context with new information.
        
        Args:
            context: Context to update
            updates: Dictionary of updates to apply
        """
        context.updated_at = datetime.now()
        
        for key, value in updates.items():
            if key == "conversation_history" and isinstance(value, list):
                context.conversation_history.extend(value)
            elif key == "active_tasks" and isinstance(value, (set, list)):
                context.active_tasks.update(value)
            elif key == "completed_tasks" and isinstance(value, (set, list)):
                context.completed_tasks.update(value)
            elif key == "entities" and isinstance(value, dict):
                context.entities.update(value)
            elif key == "workspace" and isinstance(value, dict):
                context.workspace.update(value)
            elif key == "metadata" and isinstance(value, dict):
                context.metadata.update(value)
            else:
                setattr(context, key, value)
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get processing metrics for this cognitive processor"""
        return {
            "processor_id": self.processor_id,
            "state": self.state.value,
            "metrics": self.metrics.copy(),
            "config": self.config.copy()
        }
    
    def _update_metrics(self, success: bool, confidence: float, processing_time: float) -> None:
        """Update internal metrics"""
        self.metrics["tasks_processed"] += 1
        if success:
            self.metrics["successful_tasks"] += 1
        else:
            self.metrics["failed_tasks"] += 1
        
        # Update average confidence
        total_tasks = self.metrics["tasks_processed"]
        current_avg = self.metrics["average_confidence"]
        self.metrics["average_confidence"] = (current_avg * (total_tasks - 1) + confidence) / total_tasks
        
        self.metrics["total_processing_time"] += processing_time


class BaseIntentAnalyzer(ABC):
    """
    Abstract base class for intent analysis systems.
    
    Intent analyzers detect user intentions, extract parameters,
    and provide confidence scores for decision making.
    """
    
    def __init__(self, analyzer_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the intent analyzer.
        
        Args:
            analyzer_id: Unique identifier for this analyzer
            config: Configuration parameters
        """
        self.analyzer_id = analyzer_id
        self.config = config or {}
        self.pattern_database = {}
        self.learned_patterns = {}
        
        logger.info(f"Intent analyzer '{analyzer_id}' initialized")
    
    @abstractmethod
    async def detect_intent(self, input_text: str, context: Optional[CognitiveContext] = None) -> List[IntentMatch]:
        """
        Detect intents in user input.
        
        Args:
            input_text: Text to analyze
            context: Optional cognitive context for enhanced detection
            
        Returns:
            List of detected intents with confidence scores
        """
        pass
    
    @abstractmethod
    async def extract_parameters(self, input_text: str, intent_type: str) -> Dict[str, Any]:
        """
        Extract parameters for a specific intent type.
        
        Args:
            input_text: Text to extract parameters from
            intent_type: Type of intent to extract parameters for
            
        Returns:
            Dictionary of extracted parameters
        """
        pass
    
    async def learn_pattern(self, input_text: str, intent_type: str, confidence: float) -> None:
        """
        Learn a new pattern from successful intent detection.
        
        Args:
            input_text: Input that led to successful detection
            intent_type: The detected intent type
            confidence: Confidence of the detection
        """
        if intent_type not in self.learned_patterns:
            self.learned_patterns[intent_type] = []
        
        self.learned_patterns[intent_type].append({
            "pattern": input_text,
            "confidence": confidence,
            "learned_at": datetime.now()
        })
        
        logger.debug(f"Learned new pattern for intent '{intent_type}': {input_text}")


class BaseReasoningEngine(ABC):
    """
    Abstract base class for reasoning engines.
    
    Reasoning engines handle logical inference, decision making,
    and strategic planning within the cognitive system.
    """
    
    def __init__(self, engine_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the reasoning engine.
        
        Args:
            engine_id: Unique identifier for this engine
            config: Configuration parameters
        """
        self.engine_id = engine_id
        self.config = config or {}
        self.knowledge_base = {}
        self.reasoning_rules = []
        
        logger.info(f"Reasoning engine '{engine_id}' initialized")
    
    @abstractmethod
    async def reason(self, premises: List[Dict[str, Any]], context: CognitiveContext) -> Dict[str, Any]:
        """
        Perform reasoning based on given premises.
        
        Args:
            premises: List of premise statements to reason from
            context: Current cognitive context
            
        Returns:
            Reasoning result with conclusions and confidence
        """
        pass
    
    @abstractmethod
    async def plan_strategy(self, goal: Dict[str, Any], constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Plan a strategy to achieve a given goal within constraints.
        
        Args:
            goal: Goal specification to achieve
            constraints: List of constraints to consider
            
        Returns:
            Strategy plan with steps and alternatives
        """
        pass
    
    async def add_knowledge(self, knowledge: Dict[str, Any]) -> None:
        """Add knowledge to the reasoning engine's knowledge base"""
        knowledge_id = knowledge.get("id", f"knowledge_{len(self.knowledge_base)}")
        self.knowledge_base[knowledge_id] = {
            **knowledge,
            "added_at": datetime.now()
        }
        
        logger.debug(f"Added knowledge to reasoning engine: {knowledge_id}")


class CognitiveRegistry:
    """Registry for cognitive components"""
    
    def __init__(self):
        self.processors: Dict[str, BaseCognitiveProcessor] = {}
        self.analyzers: Dict[str, BaseIntentAnalyzer] = {}
        self.reasoning_engines: Dict[str, BaseReasoningEngine] = {}
        
        logger.info("Cognitive registry initialized")
    
    def register_processor(self, processor: BaseCognitiveProcessor) -> None:
        """Register a cognitive processor"""
        self.processors[processor.processor_id] = processor
        logger.info(f"Registered cognitive processor: {processor.processor_id}")
    
    def register_analyzer(self, analyzer: BaseIntentAnalyzer) -> None:
        """Register an intent analyzer"""
        self.analyzers[analyzer.analyzer_id] = analyzer
        logger.info(f"Registered intent analyzer: {analyzer.analyzer_id}")
    
    def register_reasoning_engine(self, engine: BaseReasoningEngine) -> None:
        """Register a reasoning engine"""
        self.reasoning_engines[engine.engine_id] = engine
        logger.info(f"Registered reasoning engine: {engine.engine_id}")
    
    def get_processor(self, processor_id: str) -> Optional[BaseCognitiveProcessor]:
        """Get a registered cognitive processor"""
        return self.processors.get(processor_id)
    
    def get_analyzer(self, analyzer_id: str) -> Optional[BaseIntentAnalyzer]:
        """Get a registered intent analyzer"""
        return self.analyzers.get(analyzer_id)
    
    def get_reasoning_engine(self, engine_id: str) -> Optional[BaseReasoningEngine]:
        """Get a registered reasoning engine"""
        return self.reasoning_engines.get(engine_id)
    
    def list_components(self) -> Dict[str, List[str]]:
        """List all registered components"""
        return {
            "processors": list(self.processors.keys()),
            "analyzers": list(self.analyzers.keys()),
            "reasoning_engines": list(self.reasoning_engines.keys())
        }


# Global cognitive registry
cognitive_registry = CognitiveRegistry()
