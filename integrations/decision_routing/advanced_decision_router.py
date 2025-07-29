"""
Advanced Decision Router for Project-S Hybrid System
---------------------------------------------------
This module extends the basic decision router with advanced features:
1. Dynamic decision criteria based on contextual information
2. Historical decision analysis for pattern recognition
3. Deeper integration with cognitive core system
4. Advanced logging and visualization of decision paths
5. Adaptive routing based on execution results
"""
import logging
import asyncio
import json
from datetime import datetime
import uuid
from typing import Dict, Any, List, Optional, Callable, Union, TypeVar, cast, Set

from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode

from core.event_bus import event_bus
from core.error_handler import error_handler
from integrations.langgraph_integration import GraphState, langgraph_integrator
from integrations.langgraph_state_manager import state_manager
from integrations.decision_router import DecisionRouter, DecisionMetadata, decision_router
from core.cognitive_core import cognitive_core

# Create a logger for this module
logger = logging.getLogger("advanced_decision_router")

# Type aliases
DecisionCriteriaFunc = Callable[[GraphState], Union[str, bool]]
DecisionPatternFunc = Callable[[List[DecisionMetadata]], Dict[str, Any]]

class AdvancedDecisionRouter(DecisionRouter):
    """
    Extends the basic DecisionRouter with advanced features for hybrid system integration.
    """
    
    def __init__(self):
        """Initialize the advanced decision router."""
        super().__init__()
        
        # Track patterns across workflows
        self.global_patterns: Dict[str, Any] = {
            "common_decisions": {},
            "frequent_paths": {},
            "error_triggers": {}
        }
        
        # Decision criteria registry
        self.criteria_registry: Dict[str, DecisionCriteriaFunc] = {}
        
        # Register for additional events
        event_bus.subscribe("workflow.system_state_changed", self._on_system_state_changed)
        event_bus.subscribe("workflow.pattern_detected", self._on_pattern_detected)
        
        logger.info("Advanced Decision Router initialized")

    async def _on_system_state_changed(self, event_data: Dict[str, Any]) -> None:
        """
        Handle system state change events.
        Update decision logic based on global system state.
        
        Args:
            event_data: Event data containing system state information
        """
        graph_id = event_data.get("graph_id")
        system_state = event_data.get("system_state", {})
        
        if graph_id and graph_id in self.decision_history:
            logger.info(f"System state changed for workflow {graph_id}")
            
            # Update any active workflows that need to be aware of system state
            if graph_id in state_manager.active_states:
                state = state_manager.active_states[graph_id]
                state["context"]["system_state"] = system_state
                
                # Publish event to notify components
                await event_bus.publish("workflow.context.updated", {
                    "graph_id": graph_id,
                    "update_type": "system_state",
                    "timestamp": asyncio.get_event_loop().time()
                })

    async def _on_pattern_detected(self, event_data: Dict[str, Any]) -> None:
        """
        Handle pattern detection events.
        Track recurring patterns in decision making.
        
        Args:
            event_data: Event data containing pattern information
        """
        pattern_type = event_data.get("pattern_type")
        pattern_data = event_data.get("pattern_data", {})
        
        if pattern_type and pattern_data:
            logger.info(f"Decision pattern detected: {pattern_type}")
            
            # Update global patterns database
            if pattern_type == "decision_sequence":
                path = pattern_data.get("path")
                if path:
                    if path not in self.global_patterns["frequent_paths"]:
                        self.global_patterns["frequent_paths"][path] = 0
                    self.global_patterns["frequent_paths"][path] += 1
            
            elif pattern_type == "error_trigger":
                trigger = pattern_data.get("trigger")
                if trigger:
                    if trigger not in self.global_patterns["error_triggers"]:
                        self.global_patterns["error_triggers"][trigger] = 0
                    self.global_patterns["error_triggers"][trigger] += 1

    def register_decision_criteria(self, 
                                name: str, 
                                criteria_func: DecisionCriteriaFunc) -> None:
        """
        Register a named decision criteria function for reuse.
        
        Args:
            name: Name to register the criteria function under
            criteria_func: Function that evaluates the criteria
        """
        self.criteria_registry[name] = criteria_func
        logger.info(f"Registered decision criteria: {name}")
    
    def get_decision_criteria(self, name: str) -> Optional[DecisionCriteriaFunc]:
        """
        Get a registered decision criteria function by name.
        
        Args:
            name: Name of the registered criteria function
            
        Returns:
            The criteria function or None if not found
        """
        return self.criteria_registry.get(name)
    
    def create_composite_criteria(self, 
                               criteria_list: List[str], 
                               combination_logic: str = "AND") -> DecisionCriteriaFunc:
        """
        Create a composite decision criteria function from registered criteria.
        
        Args:
            criteria_list: List of registered criteria names to combine
            combination_logic: How to combine criteria ("AND" or "OR")
            
        Returns:
            A function that combines the results of multiple criteria functions
        """
        def composite_func(state: GraphState) -> bool:
            results = []
            
            for criteria_name in criteria_list:
                if criteria_name in self.criteria_registry:
                    criteria_func = self.criteria_registry[criteria_name]
                    result = criteria_func(state)
                    results.append(bool(result))
            
            if combination_logic == "AND":
                return all(results)
            elif combination_logic == "OR":
                return any(results)
            else:
                return False
                
        return composite_func
    
    async def create_cognitive_criteria(self, 
                                   state: GraphState,
                                   question: str) -> str:
        """
        Create a decision using the cognitive core for complex situations.
        
        Args:
            state: Current workflow state
            question: Question to ask the cognitive core
            
        Returns:
            Decision result from cognitive core
        """
        context = state.get("context", {})
        
        try:
            # Prepare a summary of relevant context
            context_summary = {
                "workflow_name": context.get("workflow_name", "unknown"),
                "current_stage": context.get("current_stage", "unknown"),
                "last_result": context.get("last_result", {}),
                "decisions_made": len(context.get("decisions", [])),
                "error_info": state.get("error_info")
            }
            
            # Query the cognitive core
            response = await cognitive_core.process_query(
                query=question,
                context=context_summary
            )
            
            # Extract decision from response
            if isinstance(response, dict):
                decision = response.get("decision", "unknown")
                logger.info(f"Cognitive criteria decided: {decision}")
                return decision
            
            # Default fallback if response format is unexpected
            return "default"
            
        except Exception as e:
            logger.error(f"Error in cognitive criteria: {e}")
            return "error"
    
    def add_adaptive_decision_node(self, 
                               graph: StateGraph,
                               node_name: str,
                               criteria_sources: List[Dict[str, Any]],
                               destinations: Dict[str, str],
                               fallback: str) -> StateGraph:
        """
        Add an adaptive decision node that can use multiple sources for decisions.
        
        Args:
            graph: The StateGraph to add the node to
            node_name: Name for the decision node
            criteria_sources: List of criteria sources in priority order
            destinations: Mapping of criterion values to destination nodes
            fallback: Fallback destination if no criteria match
            
        Returns:
            Modified StateGraph with the adaptive decision node added
        """
        async def adaptive_decision_node(state: GraphState) -> GraphState:
            decision_id = f"decision_{uuid.uuid4().hex[:8]}"
            decision_result = None
            source_used = None
            
            try:
                # Try each criteria source in priority order
                for source in criteria_sources:
                    source_type = source.get("type")
                    
                    if source_type == "function" and "name" in source:
                        # Use registered criteria function
                        criteria_name = source["name"]
                        if criteria_name in self.criteria_registry:
                            criteria_func = self.criteria_registry[criteria_name]
                            decision_result = criteria_func(state)
                            source_used = f"function:{criteria_name}"
                            break
                            
                    elif source_type == "path" and "path" in source:
                        # Extract decision from state path
                        path = source["path"]
                        if "." in path:
                            # Handle dot notation for nested path
                            parts = path.split(".")
                            value = state
                            for part in parts:
                                if isinstance(value, dict) and part in value:
                                    value = value[part]
                                else:
                                    value = None
                                    break
                            if value is not None:
                                decision_result = str(value)
                                source_used = f"path:{path}"
                                break
                        else:
                            # Direct state access
                            if path in state:
                                decision_result = str(state[path])
                                source_used = f"path:{path}"
                                break
                                
                    elif source_type == "cognitive" and "question" in source:
                        # Use cognitive core for decision
                        question = source["question"]
                        decision_result = await self.create_cognitive_criteria(state, question)
                        source_used = f"cognitive:{question[:20]}"
                        break
                
                # Determine the next node
                next_node = None
                
                if decision_result and str(decision_result) in destinations:
                    next_node = destinations[str(decision_result)]
                else:
                    next_node = fallback
                    decision_result = "fallback"
                
                # Create decision metadata
                decision_meta = DecisionMetadata(
                    decision_id=decision_id,
                    timestamp=asyncio.get_event_loop().time(),
                    source_node=node_name,
                    decision_criteria=str(decision_result),
                    considered_options=list(destinations.keys()),
                    selected_option=next_node,
                    context_snapshot={
                        "status": state.get("status", "unknown"),
                        "source_used": source_used,
                        "context_keys": list(state.get("context", {}).keys())
                    }
                )
                
                # Store in decision history
                graph_id = state["context"].get("graph_id", "unknown")
                if graph_id not in self.decision_history:
                    self.decision_history[graph_id] = []
                self.decision_history[graph_id].append(decision_meta)
                
                # Add to state
                if "decisions" not in state["context"]:
                    state["context"]["decisions"] = []
                state["context"]["decisions"].append(decision_meta.to_dict())
                
                # Set next node and source used
                state["next_node"] = next_node
                state["context"]["last_decision_source"] = source_used
                
                # Publish event
                await event_bus.publish("workflow.decision.made", {
                    "graph_id": graph_id,
                    "decision_id": decision_id,
                    "source_node": node_name,
                    "decision": next_node,
                    "criterion_value": str(decision_result),
                    "source_used": source_used,
                    "options": list(destinations.keys()),
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                return state
                
            except Exception as e:
                logger.error(f"Error in adaptive decision node {node_name}: {e}")
                error_context = {
                    "component": "adaptive_decision_router",
                    "decision_node": node_name,
                    "criteria_sources": criteria_sources
                }
                asyncio.create_task(error_handler.handle_error(e, error_context))
                
                # Fallback handling
                state["next_node"] = fallback
                state["context"]["decision_error"] = str(e)
                
                return state
        
        # Add the node to the graph
        graph.add_node(node_name, adaptive_decision_node)
        
        # Add conditional edge based on next_node
        condition_func = self.create_condition_function("next_node")
        
        # Create mapping of possible values to destinations
        conditional_map = {dest: dest for dest in destinations.values()}
        conditional_map[fallback] = fallback  # Ensure fallback is included
        
        # Add the conditional edge
        graph.add_conditional_edges(
            node_name,
            condition_func,
            conditional_map
        )
        
        return graph
    
    def detect_decision_patterns(self, graph_id: str) -> Dict[str, Any]:
        """
        Detect patterns in decision history that might indicate issues or opportunities.
        
        Args:
            graph_id: The graph ID to analyze
            
        Returns:
            Dictionary of detected patterns
        """
        if graph_id not in self.decision_history:
            return {"status": "no_data"}
            
        decisions = self.decision_history[graph_id]
        patterns = {}
        
        # Look for oscillation (switching back and forth between options)
        oscillations = []
        for i in range(2, len(decisions)):
            if (decisions[i].source_node == decisions[i-2].source_node and
                decisions[i].selected_option != decisions[i-1].selected_option and 
                decisions[i].selected_option == decisions[i-2].selected_option):
                oscillations.append({
                    "node": decisions[i].source_node,
                    "options": [decisions[i-2].selected_option, decisions[i-1].selected_option],
                    "position": i
                })
        
        if oscillations:
            patterns["oscillation"] = oscillations
        
        # Look for repeated error paths
        error_paths = []
        for i in range(1, len(decisions)):
            if "error" in decisions[i].selected_option.lower():
                error_paths.append({
                    "from_node": decisions[i-1].source_node,
                    "to_node": decisions[i].source_node,
                    "criteria": decisions[i-1].decision_criteria
                })
        
        if error_paths:
            patterns["error_paths"] = error_paths
            
        # Look for fallback usage
        fallbacks = []
        for decision in decisions:
            if decision.decision_criteria == "fallback":
                fallbacks.append({
                    "node": decision.source_node,
                    "timestamp": decision.timestamp
                })
                
        if fallbacks:
            patterns["fallbacks"] = fallbacks
            
        # Check pattern significance
        if patterns:
            patterns["status"] = "patterns_detected"
            
            # Publish event about detected patterns
            asyncio.create_task(event_bus.publish("workflow.pattern_detected", {
                "graph_id": graph_id,
                "pattern_type": "decision_sequence",
                "pattern_data": patterns
            }))
        else:
            patterns["status"] = "no_patterns"
        
        return patterns
    
    def analyze_global_decision_trends(self) -> Dict[str, Any]:
        """
        Analyze global trends across all workflows.
        
        Returns:
            Analysis of global decision patterns
        """
        total_decisions = sum(len(decisions) for decisions in self.decision_history.values())
        
        # Aggregate node usage across workflows
        node_usage = {}
        option_selection = {}
        
        for graph_id, decisions in self.decision_history.items():
            for decision in decisions:
                # Count node usage
                if decision.source_node not in node_usage:
                    node_usage[decision.source_node] = 0
                node_usage[decision.source_node] += 1
                
                # Count option selection
                option_key = f"{decision.source_node}:{decision.selected_option}"
                if option_key not in option_selection:
                    option_selection[option_key] = 0
                option_selection[option_key] += 1
        
        # Find top nodes and options
        top_nodes = dict(sorted(node_usage.items(), key=lambda x: x[1], reverse=True)[:5])
        top_options = dict(sorted(option_selection.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Combine with existing global patterns
        return {
            "total_decisions": total_decisions,
            "total_workflows": len(self.decision_history),
            "top_decision_points": top_nodes,
            "top_selected_options": top_options,
            "frequent_paths": self.global_patterns["frequent_paths"],
            "error_triggers": self.global_patterns["error_triggers"]
        }


# Create singleton instance
advanced_decision_router = AdvancedDecisionRouter()


# Additional decision criteria functions

def check_context_contains(state: GraphState, key_path: str, value: Any = None) -> bool:
    """
    Check if context contains a specific key and optionally a specific value.
    
    Args:
        state: Current workflow state
        key_path: Path to the key in context (dot notation for nested)
        value: Optional value to match
        
    Returns:
        True if key exists and value matches (if provided), False otherwise
    """
    if "." in key_path:
        # Handle nested path
        parts = key_path.split(".")
        current = state.get("context", {})
        
        for part in parts[:-1]:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
                
        # Check final part
        last_part = parts[-1]
        if isinstance(current, dict) and last_part in current:
            if value is not None:
                return current[last_part] == value
            return True
        return False
    else:
        # Direct path
        if key_path in state.get("context", {}):
            if value is not None:
                return state["context"][key_path] == value
            return True
        return False

def check_system_supports(state: GraphState, capability: str) -> bool:
    """
    Check if the system supports a specific capability.
    
    Args:
        state: Current workflow state
        capability: Capability to check for
        
    Returns:
        True if system supports the capability, False otherwise
    """
    system_state = state["context"].get("system_state", {})
    capabilities = system_state.get("capabilities", [])
    return capability in capabilities

def route_by_model_capabilities(state: GraphState) -> str:
    """
    Route based on model capabilities required for the task.
    
    Args:
        state: Current workflow state
        
    Returns:
        Route based on required model capabilities
    """
    context = state.get("context", {})
    required_capabilities = context.get("required_capabilities", [])
    
    if "code_generation" in required_capabilities:
        return "code_gen_model"
    elif "reasoning" in required_capabilities:
        return "reasoning_model"
    elif "classification" in required_capabilities:
        return "classification_model"
    else:
        return "default_model"

def route_by_confidence(state: GraphState) -> str:
    """
    Route based on confidence level of previous step.
    
    Args:
        state: Current workflow state
        
    Returns:
        Route based on confidence level
    """
    context = state.get("context", {})
    last_result = context.get("last_result", {})
    confidence = last_result.get("confidence", 0)
    
    if confidence >= 0.8:
        return "high_confidence"
    elif confidence >= 0.5:
        return "medium_confidence"
    else:
        return "low_confidence"
