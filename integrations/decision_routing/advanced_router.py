"""
Project-S v2 Architecture - Advanced Decision Router
===================================================
Advanced decision routing implementation with pattern recognition,
cognitive integration, and adaptive learning capabilities.

This module provides:
- Pattern-aware decision making
- Cognitive-assisted routing
- Composite criteria evaluation
- Dynamic adaptation based on outcomes
"""

import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Union

from .base_router import (
    BaseDecisionRouter, 
    DecisionResult, 
    DecisionMetadata, 
    DecisionType, 
    DecisionStatus,
    StateType
)

logger = logging.getLogger(__name__)

class AdvancedDecisionRouter(BaseDecisionRouter):
    """
    Advanced decision router with pattern recognition and cognitive integration.
    
    Features:
    - Historical pattern analysis
    - Composite criteria evaluation
    - Cognitive-assisted decision making
    - Adaptive learning from outcomes
    - Dynamic criteria weighting
    """
    
    def __init__(self, name: str = "advanced_router", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced decision router.
        
        Args:
            name: Router identifier
            config: Configuration dictionary
        """
        super().__init__(name, config)
        
        # Advanced features
        self.global_patterns: Dict[str, Any] = {
            "common_decisions": {},
            "frequent_paths": {},
            "error_triggers": {},
            "success_patterns": {}
        }
        
        self.criteria_weights: Dict[str, float] = {}
        self.outcome_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # External dependencies (injected later)
        self.cognitive_processor: Optional[Any] = None
        self.event_bus: Optional[Any] = None
        
        # Register advanced criteria
        self._register_advanced_criteria()
        
        logger.info(f"Advanced decision router '{name}' initialized")
    
    def set_dependencies(self, cognitive_processor: Any = None, event_bus: Any = None) -> None:
        """
        Inject external dependencies.
        
        Args:
            cognitive_processor: Cognitive processing service
            event_bus: Event bus for notifications
        """
        self.cognitive_processor = cognitive_processor
        self.event_bus = event_bus
        
        if cognitive_processor:
            logger.info("Cognitive processor integration enabled")
        if event_bus:
            logger.info("Event bus integration enabled")
    
    async def make_decision(self, 
                          state: StateType, 
                          options: List[str], 
                          criteria: Optional[str] = None,
                          **kwargs) -> DecisionResult:
        """
        Make an advanced decision using multiple strategies.
        
        Args:
            state: Current state for decision making
            options: Available options to choose from
            criteria: Primary decision criteria to use
            **kwargs: Additional parameters
            
        Returns:
            DecisionResult containing the decision and metadata
        """
        decision_id = f"decision_{uuid.uuid4().hex[:8]}"
        start_time = datetime.now()
        
        try:
            # Strategy 1: Pattern-based suggestion
            pattern_suggestion = self.get_pattern_suggestion(state)
            pattern_confidence = 0.0
            
            if pattern_suggestion and pattern_suggestion in options:
                pattern_confidence = 0.8
                logger.debug(f"Pattern suggests: {pattern_suggestion}")
            
            # Strategy 2: Criteria-based decision
            criteria_decision = None
            criteria_confidence = 0.0
            
            if criteria:
                criteria_decision, criteria_confidence = await self._evaluate_criteria(
                    state, options, criteria
                )
            
            # Strategy 3: Cognitive-assisted decision (if available)
            cognitive_decision = None
            cognitive_confidence = 0.0
            
            if self.cognitive_processor and self._requires_cognitive_assistance(state, options):
                cognitive_decision, cognitive_confidence = await self._get_cognitive_decision(
                    state, options, kwargs.get("context", "")
                )
            
            # Strategy 4: Composite criteria evaluation
            composite_decision = None
            composite_confidence = 0.0
            
            if len(options) > 2:  # Use composite for complex decisions
                composite_decision, composite_confidence = await self._evaluate_composite_criteria(
                    state, options
                )
            
            # Combine strategies and select best decision
            decision_candidates = [
                (pattern_suggestion, pattern_confidence, "pattern"),
                (criteria_decision, criteria_confidence, "criteria"),
                (cognitive_decision, cognitive_confidence, "cognitive"),
                (composite_decision, composite_confidence, "composite")
            ]
            
            # Filter valid candidates
            valid_candidates = [
                (decision, confidence, source) 
                for decision, confidence, source in decision_candidates
                if decision and decision in options and confidence > 0
            ]
            
            # Select best candidate
            if valid_candidates:
                best_decision, best_confidence, best_source = max(
                    valid_candidates, key=lambda x: x[1]
                )
            else:
                # Fallback to first option
                best_decision = options[0]
                best_confidence = 0.1
                best_source = "fallback"
            
            # Create decision metadata
            execution_time = (datetime.now() - start_time).total_seconds()
            
            metadata = DecisionMetadata(
                decision_id=decision_id,
                timestamp=start_time.timestamp(),
                source_node=kwargs.get("source_node", "unknown"),
                decision_type=DecisionType.ADAPTIVE,
                decision_criteria=criteria or best_source,
                considered_options=options,
                selected_option=best_decision,
                confidence_score=best_confidence,
                context_snapshot=self._create_context_snapshot(state),
                status=DecisionStatus.COMPLETED,
                execution_time=execution_time
            )
            
            # Create result with reasoning
            reasoning = self._generate_reasoning(
                decision_candidates, best_decision, best_source, best_confidence
            )
            
            result = DecisionResult(
                decision=best_decision,
                confidence=best_confidence,
                metadata=metadata,
                reasoning=reasoning
            )
            
            # Store decision for learning
            workflow_id = kwargs.get("workflow_id", "unknown")
            self.add_decision_to_history(workflow_id, metadata)
            
            # Publish event if event bus available
            if self.event_bus:
                asyncio.create_task(self.event_bus.publish("decision.made", {
                    "router_name": self.name,
                    "decision_id": decision_id,
                    "decision": best_decision,
                    "confidence": best_confidence,
                    "source": best_source
                }))
            
            logger.info(f"Decision made: {best_decision} (confidence: {best_confidence:.2f}, source: {best_source})")
            return result
            
        except Exception as e:
            logger.error(f"Error making decision: {e}")
            
            # Return fallback decision
            metadata = DecisionMetadata(
                decision_id=decision_id,
                timestamp=start_time.timestamp(),
                source_node=kwargs.get("source_node", "unknown"),
                decision_type=DecisionType.ADAPTIVE,
                decision_criteria="error_fallback",
                considered_options=options,
                selected_option=options[0] if options else "unknown",
                confidence_score=0.1,
                context_snapshot={},
                status=DecisionStatus.FAILED,
                execution_time=(datetime.now() - start_time).total_seconds()
            )
            
            return DecisionResult(
                decision=options[0] if options else "unknown",
                confidence=0.1,
                metadata=metadata,
                reasoning=f"Error occurred: {e}. Used fallback decision."
            )
    
    async def _evaluate_criteria(self, 
                               state: StateType, 
                               options: List[str], 
                               criteria: str) -> tuple[Optional[str], float]:
        """
        Evaluate decision using specified criteria.
        
        Args:
            state: Current state
            options: Available options
            criteria: Criteria name to use
            
        Returns:
            Tuple of (decision, confidence)
        """
        criteria_func = self.get_criteria(criteria)
        
        if not criteria_func:
            logger.warning(f"Criteria '{criteria}' not found")
            return None, 0.0
        
        try:
            # Simple criteria evaluation
            result = criteria_func(state)
            
            if isinstance(result, bool):
                # Boolean criteria - use first option if True
                return (options[0], 0.9) if result and options else (None, 0.0)
            elif isinstance(result, str):
                # String criteria - return if in options
                return (result, 0.9) if result in options else (None, 0.0)
            else:
                logger.warning(f"Unexpected criteria result type: {type(result)}")
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Error evaluating criteria '{criteria}': {e}")
            return None, 0.0
    
    async def _get_cognitive_decision(self, 
                                    state: StateType, 
                                    options: List[str],
                                    context: str) -> tuple[Optional[str], float]:
        """
        Get decision from cognitive processor.
        
        Args:
            state: Current state
            options: Available options
            context: Additional context
            
        Returns:
            Tuple of (decision, confidence)
        """
        if not self.cognitive_processor:
            return None, 0.0
        
        try:
            # Prepare query for cognitive processor
            query = f"Given the current situation, which option should be chosen: {', '.join(options)}?"
            
            context_data = {
                "state_summary": self._summarize_state(state),
                "available_options": options,
                "additional_context": context
            }
            
            # Query cognitive processor
            response = await self.cognitive_processor.process_query(
                query=query,
                context=context_data
            )
            
            # Extract decision from response
            if isinstance(response, dict):
                decision = response.get("decision", "")
                confidence = response.get("confidence", 0.5)
                
                if decision in options:
                    return decision, float(confidence)
            
            # Try to extract decision from text response
            if isinstance(response, str):
                response_lower = response.lower()
                for option in options:
                    if option.lower() in response_lower:
                        return option, 0.7
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error getting cognitive decision: {e}")
            return None, 0.0
    
    async def _evaluate_composite_criteria(self, 
                                         state: StateType, 
                                         options: List[str]) -> tuple[Optional[str], float]:
        """
        Evaluate decision using multiple criteria with weighting.
        
        Args:
            state: Current state
            options: Available options
            
        Returns:
            Tuple of (decision, confidence)
        """
        try:
            option_scores = {option: 0.0 for option in options}
            total_weight = 0.0
            
            # Evaluate each registered criteria
            for criteria_name, criteria_func in self.criteria_registry.items():
                try:
                    result = criteria_func(state)
                    weight = self.criteria_weights.get(criteria_name, 1.0)
                    
                    if isinstance(result, bool):
                        if result:
                            # Boost first option for positive boolean results
                            if options:
                                option_scores[options[0]] += weight
                    elif isinstance(result, str) and result in options:
                        option_scores[result] += weight
                    
                    total_weight += weight
                    
                except Exception as e:
                    logger.warning(f"Error evaluating criteria '{criteria_name}': {e}")
                    continue
            
            if total_weight == 0:
                return None, 0.0
            
            # Normalize scores
            normalized_scores = {
                option: score / total_weight 
                for option, score in option_scores.items()
            }
            
            # Find best option
            best_option = max(normalized_scores, key=normalized_scores.get)
            best_score = normalized_scores[best_option]
            
            if best_score > 0.1:  # Minimum threshold
                return best_option, min(best_score, 0.95)
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error in composite criteria evaluation: {e}")
            return None, 0.0
    
    def _requires_cognitive_assistance(self, state: StateType, options: List[str]) -> bool:
        """
        Determine if cognitive assistance is needed for this decision.
        
        Args:
            state: Current state
            options: Available options
            
        Returns:
            True if cognitive assistance should be used
        """
        # Use cognitive assistance for complex decisions
        if len(options) > 3:
            return True
        
        # Use for error states
        if isinstance(state, dict) and ("error" in state or "error_info" in state):
            return True
        
        # Use for complex input
        if isinstance(state, dict):
            input_text = str(state.get("input", ""))
            if len(input_text.split()) > 100:
                return True
        
        return False
    
    def _create_context_snapshot(self, state: StateType) -> Dict[str, Any]:
        """
        Create a context snapshot for decision tracking.
        
        Args:
            state: Current state
            
        Returns:
            Context snapshot dictionary
        """
        if isinstance(state, dict):
            # Extract key information, avoid large data
            return {
                "input_length": len(str(state.get("input", ""))),
                "has_error": "error" in state or "error_info" in state,
                "state_keys": list(state.keys())[:10],  # Limit to first 10 keys
                "timestamp": datetime.now().isoformat()
            }
        
        return {"state_type": str(type(state)), "timestamp": datetime.now().isoformat()}
    
    def _summarize_state(self, state: StateType) -> str:
        """
        Create a human-readable summary of the state.
        
        Args:
            state: Current state
            
        Returns:
            State summary string
        """
        if isinstance(state, dict):
            input_text = str(state.get("input", ""))
            summary_parts = []
            
            if input_text:
                summary_parts.append(f"Input: {input_text[:100]}...")
            
            if "error" in state or "error_info" in state:
                summary_parts.append("Error state detected")
            
            if "workflow_type" in state:
                summary_parts.append(f"Workflow: {state['workflow_type']}")
            
            return "; ".join(summary_parts) if summary_parts else "Empty state"
        
        return f"State type: {type(state)}"
    
    def _generate_reasoning(self, 
                          candidates: List[tuple], 
                          decision: str, 
                          source: str, 
                          confidence: float) -> str:
        """
        Generate human-readable reasoning for the decision.
        
        Args:
            candidates: All decision candidates evaluated
            decision: Final decision made
            source: Source of the decision
            confidence: Confidence level
            
        Returns:
            Reasoning string
        """
        reasoning_parts = [f"Selected '{decision}' from {source} strategy with {confidence:.2%} confidence."]
        
        # Add information about other candidates
        other_candidates = [(d, c, s) for d, c, s in candidates if d and d != decision and c > 0]
        
        if other_candidates:
            alternatives = [f"{s}:{d}({c:.2%})" for d, c, s in other_candidates]
            reasoning_parts.append(f"Other candidates: {', '.join(alternatives)}")
        
        # Add strategy explanation
        strategy_explanations = {
            "pattern": "Based on historical decision patterns",
            "criteria": "Based on specific decision criteria evaluation",
            "cognitive": "Based on cognitive processor analysis",
            "composite": "Based on weighted evaluation of multiple criteria",
            "fallback": "Default fallback option"
        }
        
        if source in strategy_explanations:
            reasoning_parts.append(strategy_explanations[source])
        
        return " ".join(reasoning_parts)
    
    def _register_advanced_criteria(self) -> None:
        """Register advanced decision criteria."""
        
        def needs_multi_step(state: StateType) -> bool:
            """Check if task needs multi-step processing."""
            if isinstance(state, dict):
                input_text = str(state.get("input", "")).lower()
                multi_step_indicators = ["first", "then", "after", "finally", "step"]
                return any(indicator in input_text for indicator in multi_step_indicators)
            return False
        
        def has_specific_model_request(state: StateType) -> str:
            """Check for specific model requests."""
            if isinstance(state, dict):
                input_text = str(state.get("input", "")).lower()
                if "gpt-4" in input_text:
                    return "gpt-4"
                elif "claude" in input_text:
                    return "claude"
                elif "local" in input_text or "ollama" in input_text:
                    return "local"
            return ""
        
        def is_high_priority(state: StateType) -> bool:
            """Check if task is high priority."""
            if isinstance(state, dict):
                input_text = str(state.get("input", "")).lower()
                priority_indicators = ["urgent", "asap", "immediately", "critical", "emergency"]
                return any(indicator in input_text for indicator in priority_indicators)
            return False
        
        def requires_verification(state: StateType) -> bool:
            """Check if task requires verification."""
            if isinstance(state, dict):
                input_text = str(state.get("input", "")).lower()
                verification_indicators = ["verify", "check", "validate", "test", "confirm"]
                return any(indicator in input_text for indicator in verification_indicators)
            return False
        
        # Register advanced criteria
        self.register_criteria("needs_multi_step", needs_multi_step)
        self.register_criteria("has_specific_model_request", has_specific_model_request)
        self.register_criteria("is_high_priority", is_high_priority)
        self.register_criteria("requires_verification", requires_verification)
        
        # Set default weights for criteria
        self.criteria_weights.update({
            "has_error": 2.0,
            "is_high_priority": 1.8,
            "is_complex_task": 1.5,
            "needs_multi_step": 1.4,
            "requires_verification": 1.2,
            "has_tools_needed": 1.1,
            "is_creative_task": 1.0,
            "has_specific_model_request": 0.8
        })
    
    async def learn_from_outcome(self, 
                               decision_id: str, 
                               outcome: Dict[str, Any]) -> None:
        """
        Learn from decision outcome to improve future decisions.
        
        Args:
            decision_id: ID of the decision
            outcome: Outcome information
        """
        try:
            success = outcome.get("success", False)
            execution_time = outcome.get("execution_time", 0.0)
            error_info = outcome.get("error_info")
            
            # Store outcome for analysis
            if decision_id not in self.outcome_history:
                self.outcome_history[decision_id] = []
            
            outcome_record = {
                "timestamp": datetime.now().isoformat(),
                "success": success,
                "execution_time": execution_time,
                "error_info": error_info
            }
            
            self.outcome_history[decision_id].append(outcome_record)
            
            # Adjust criteria weights based on outcome
            if success:
                # Positive reinforcement - slightly increase weights used in successful decisions
                pass  # Implement learning algorithm
            else:
                # Negative reinforcement - slightly decrease weights
                pass  # Implement learning algorithm
            
            logger.debug(f"Learned from outcome for decision {decision_id}: success={success}")
            
        except Exception as e:
            logger.error(f"Error learning from outcome: {e}")
    
    def get_advanced_statistics(self) -> Dict[str, Any]:
        """Get advanced statistics about this router."""
        base_stats = self.get_decision_statistics()
        
        # Add advanced statistics
        base_stats.update({
            "global_patterns": len(self.global_patterns),
            "criteria_weights": len(self.criteria_weights),
            "outcome_records": len(self.outcome_history),
            "cognitive_enabled": self.cognitive_processor is not None,
            "event_bus_enabled": self.event_bus is not None
        })
        
        return base_stats
