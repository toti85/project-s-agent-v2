"""
PROJECT-S V2 - Semantic Reasoning Engine
=======================================

This module provides advanced semantic reasoning capabilities for the v2 system,
building on the existing semantic_engine.py to provide enhanced understanding,
inference, and strategic planning using semantic embeddings and similarity matching.

Key Features:
1. Semantic similarity-based reasoning
2. Context-aware inference and decision making
3. Strategic planning with semantic understanding
4. Knowledge base integration with embeddings
5. Multi-language semantic reasoning support
6. Learning from reasoning outcomes

Legacy Integration:
- Maintains compatibility with existing semantic_engine functionality
- Enhances capabilities with reasoning and planning
- Provides migration path for existing semantic patterns
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from dataclasses import dataclass, field

from .base_cognitive import (
    BaseReasoningEngine,
    CognitiveContext,
    CognitiveResult
)

logger = logging.getLogger(__name__)

# Optional semantic embeddings support
try:
    from sentence_transformers import SentenceTransformer
    import torch
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence transformers not available - using fallback reasoning")

try:
    from scipy.spatial.distance import cosine
    import numpy as np
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("Scipy not available - using basic similarity calculations")


@dataclass
class SemanticMatch:
    """Represents a semantic similarity match"""
    text: str
    similarity: float
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)


@dataclass
class ReasoningPremise:
    """Represents a premise for reasoning"""
    statement: str
    confidence: float
    source: str
    semantic_embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningConclusion:
    """Represents a reasoning conclusion"""
    conclusion: str
    confidence: float
    supporting_premises: List[str]
    reasoning_path: List[str]
    semantic_support: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class SemanticReasoningEngine(BaseReasoningEngine):
    """
    Semantic reasoning engine with advanced understanding and inference capabilities.
    
    This engine provides sophisticated reasoning capabilities including:
    - Semantic similarity-based inference
    - Context-aware decision making
    - Strategic planning with semantic understanding
    - Knowledge integration and retrieval
    - Learning from reasoning outcomes
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the semantic reasoning engine"""
        super().__init__("semantic_reasoning_engine", config)
        
        # Configuration
        self.similarity_threshold = config.get("similarity_threshold", 0.7) if config else 0.7
        self.confidence_threshold = config.get("confidence_threshold", 0.6) if config else 0.6
        self.max_reasoning_depth = config.get("max_reasoning_depth", 5) if config else 5
        self.enable_semantic_embeddings = config.get("enable_semantic_embeddings", True) if config else True
        
        # Initialize semantic model if available
        self.semantic_model = None
        self.embeddings_cache = {}
        
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.enable_semantic_embeddings:
            try:
                # Use a lightweight model for better performance
                model_name = config.get("embedding_model", "all-MiniLM-L6-v2") if config else "all-MiniLM-L6-v2"
                self.semantic_model = SentenceTransformer(model_name)
                logger.info(f"✅ Semantic model loaded: {model_name}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load semantic model: {e}")
                self.semantic_model = None
        
        # Reasoning patterns and heuristics
        self.reasoning_patterns = {
            "causal": ["because", "since", "due to", "as a result", "therefore"],
            "conditional": ["if", "when", "unless", "provided that", "in case"],
            "comparative": ["better than", "worse than", "similar to", "different from"],
            "temporal": ["before", "after", "during", "while", "until"],
            "spatial": ["above", "below", "near", "far", "inside", "outside"]
        }
        
        # Knowledge domains
        self.knowledge_domains = {
            "file_operations": {},
            "system_administration": {},
            "code_analysis": {},
            "workflow_management": {},
            "problem_solving": {}
        }
        
        # Reasoning history for learning
        self.reasoning_history = []
        self.successful_patterns = {}
        
        logger.info("Semantic reasoning engine initialized")
    
    async def reason(self, premises: List[Dict[str, Any]], context: CognitiveContext) -> Dict[str, Any]:
        """
        Perform reasoning based on given premises using semantic understanding.
        
        Args:
            premises: List of premise statements to reason from
            context: Current cognitive context
            
        Returns:
            Reasoning result with conclusions and confidence
        """
        logger.info(f"🧠 Starting semantic reasoning with {len(premises)} premises")
        
        # Convert premises to structured format
        structured_premises = await self._structure_premises(premises)
        
        # Analyze premise relationships using semantic similarity
        relationships = await self._analyze_premise_relationships(structured_premises)
        
        # Generate inferences
        inferences = await self._generate_inferences(structured_premises, relationships, context)
        
        # Evaluate conclusions
        conclusions = await self._evaluate_conclusions(inferences, structured_premises, context)
        
        # Calculate overall confidence
        overall_confidence = await self._calculate_reasoning_confidence(conclusions, structured_premises)
        
        # Learn from reasoning
        await self._learn_from_reasoning(structured_premises, conclusions, overall_confidence)
        
        result = {
            "success": len(conclusions) > 0,
            "conclusions": [
                {
                    "conclusion": c.conclusion,
                    "confidence": c.confidence,
                    "supporting_premises": c.supporting_premises,
                    "reasoning_path": c.reasoning_path,
                    "semantic_support": c.semantic_support
                }
                for c in conclusions
            ],
            "overall_confidence": overall_confidence,
            "premise_relationships": relationships,
            "reasoning_metadata": {
                "premises_count": len(structured_premises),
                "inferences_generated": len(inferences),
                "conclusions_reached": len(conclusions),
                "semantic_model_used": self.semantic_model is not None,
                "reasoning_depth": max(len(c.reasoning_path) for c in conclusions) if conclusions else 0
            }
        }
        
        logger.info(f"✅ Reasoning completed: {len(conclusions)} conclusions with {overall_confidence:.2f} confidence")
        return result
    
    async def plan_strategy(self, goal: Dict[str, Any], constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Plan a strategy to achieve a given goal within constraints using semantic understanding.
        
        Args:
            goal: Goal specification to achieve
            constraints: List of constraints to consider
            
        Returns:
            Strategy plan with steps and alternatives
        """
        logger.info(f"🎯 Planning strategy for goal: {goal.get('description', 'Unknown goal')}")
        
        # Analyze goal semantically
        goal_analysis = await self._analyze_goal_semantically(goal)
        
        # Analyze constraints
        constraint_analysis = await self._analyze_constraints(constraints)
        
        # Find similar successful strategies
        similar_strategies = await self._find_similar_strategies(goal, constraints)
        
        # Generate strategy alternatives
        strategy_alternatives = await self._generate_strategy_alternatives(goal_analysis, constraint_analysis, similar_strategies)
        
        # Evaluate and rank strategies
        ranked_strategies = await self._evaluate_strategies(strategy_alternatives, goal, constraints)
        
        # Select best strategy
        best_strategy = ranked_strategies[0] if ranked_strategies else None
        
        result = {
            "success": best_strategy is not None,
            "primary_strategy": best_strategy,
            "alternative_strategies": ranked_strategies[1:3] if len(ranked_strategies) > 1 else [],
            "goal_analysis": goal_analysis,
            "constraint_analysis": constraint_analysis,
            "similar_strategies_found": len(similar_strategies),
            "strategy_metadata": {
                "alternatives_generated": len(strategy_alternatives),
                "semantic_analysis_used": self.semantic_model is not None,
                "planning_confidence": best_strategy.get("confidence", 0.0) if best_strategy else 0.0
            }
        }
        
        logger.info(f"📋 Strategy planning completed: {len(ranked_strategies)} strategies generated")
        return result
    
    # Core reasoning methods
    
    async def _structure_premises(self, premises: List[Dict[str, Any]]) -> List[ReasoningPremise]:
        """Structure premises for reasoning"""
        structured = []
        
        for i, premise in enumerate(premises):
            if isinstance(premise, str):
                premise = {"statement": premise, "confidence": 1.0, "source": "user"}
            
            statement = premise.get("statement", str(premise))
            confidence = premise.get("confidence", 0.8)
            source = premise.get("source", f"premise_{i}")
            
            # Generate semantic embedding if available
            embedding = None
            if self.semantic_model:
                try:
                    embedding = await self._get_embedding(statement)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for premise: {e}")
            
            structured.append(ReasoningPremise(
                statement=statement,
                confidence=confidence,
                source=source,
                semantic_embedding=embedding,
                metadata=premise.get("metadata", {})
            ))
        
        return structured
    
    async def _analyze_premise_relationships(self, premises: List[ReasoningPremise]) -> Dict[str, Any]:
        """Analyze relationships between premises using semantic similarity"""
        relationships = {
            "similarities": [],
            "contradictions": [],
            "implications": [],
            "clusters": []
        }
        
        if not self.semantic_model or len(premises) < 2:
            return relationships
        
        # Calculate pairwise similarities
        for i, premise1 in enumerate(premises):
            for j, premise2 in enumerate(premises[i+1:], i+1):
                if premise1.semantic_embedding is not None and premise2.semantic_embedding is not None:
                    similarity = await self._calculate_semantic_similarity(
                        premise1.semantic_embedding,
                        premise2.semantic_embedding
                    )
                    
                    if similarity > self.similarity_threshold:
                        relationships["similarities"].append({
                            "premise1": i,
                            "premise2": j,
                            "similarity": similarity,
                            "statements": [premise1.statement, premise2.statement]
                        })
                    
                    # Check for contradictions (high similarity but opposing meanings)
                    if similarity > 0.6 and await self._detect_contradiction(premise1.statement, premise2.statement):
                        relationships["contradictions"].append({
                            "premise1": i,
                            "premise2": j,
                            "similarity": similarity,
                            "statements": [premise1.statement, premise2.statement]
                        })
        
        # Detect implication patterns
        relationships["implications"] = await self._detect_implications(premises)
        
        # Cluster similar premises
        relationships["clusters"] = await self._cluster_premises(premises)
        
        return relationships
    
    async def _generate_inferences(self, premises: List[ReasoningPremise], relationships: Dict[str, Any], context: CognitiveContext) -> List[Dict[str, Any]]:
        """Generate inferences from premises and their relationships"""
        inferences = []
        
        # Generate inferences from similar premises
        for similarity in relationships["similarities"]:
            if similarity["similarity"] > 0.8:
                premise1 = premises[similarity["premise1"]]
                premise2 = premises[similarity["premise2"]]
                
                inference = await self._generate_similarity_inference(premise1, premise2, similarity["similarity"])
                if inference:
                    inferences.append(inference)
        
        # Generate inferences from implication patterns
        for implication in relationships["implications"]:
            inference = await self._generate_implication_inference(implication, premises)
            if inference:
                inferences.append(inference)
        
        # Generate causal inferences
        causal_inferences = await self._generate_causal_inferences(premises, context)
        inferences.extend(causal_inferences)
        
        # Generate conditional inferences
        conditional_inferences = await self._generate_conditional_inferences(premises, context)
        inferences.extend(conditional_inferences)
        
        return inferences
    
    async def _evaluate_conclusions(self, inferences: List[Dict[str, Any]], premises: List[ReasoningPremise], context: CognitiveContext) -> List[ReasoningConclusion]:
        """Evaluate inferences to form conclusions"""
        conclusions = []
        
        for inference in inferences:
            # Calculate confidence based on supporting premises
            confidence = await self._calculate_inference_confidence(inference, premises)
            
            if confidence >= self.confidence_threshold:
                # Calculate semantic support
                semantic_support = await self._calculate_semantic_support(inference, premises)
                
                conclusion = ReasoningConclusion(
                    conclusion=inference["statement"],
                    confidence=confidence,
                    supporting_premises=inference.get("supporting_premises", []),
                    reasoning_path=inference.get("reasoning_path", []),
                    semantic_support=semantic_support,
                    metadata={
                        "inference_type": inference.get("type", "unknown"),
                        "generated_at": datetime.now().isoformat()
                    }
                )
                conclusions.append(conclusion)
        
        # Sort by confidence
        conclusions.sort(key=lambda c: c.confidence, reverse=True)
        
        return conclusions
    
    async def _calculate_reasoning_confidence(self, conclusions: List[ReasoningConclusion], premises: List[ReasoningPremise]) -> float:
        """Calculate overall confidence in the reasoning process"""
        if not conclusions:
            return 0.0
        
        # Weight by conclusion confidence and premise quality
        total_confidence = sum(c.confidence for c in conclusions)
        premise_quality = sum(p.confidence for p in premises) / len(premises) if premises else 0.0
        
        # Factor in semantic support
        semantic_factor = sum(c.semantic_support for c in conclusions) / len(conclusions)
        
        overall = (total_confidence / len(conclusions)) * premise_quality * (1.0 + semantic_factor * 0.2)
        return min(1.0, overall)
    
    # Semantic similarity methods
    
    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get semantic embedding for text"""
        if not self.semantic_model:
            return None
        
        # Check cache first
        text_hash = hash(text)
        if text_hash in self.embeddings_cache:
            return self.embeddings_cache[text_hash]
        
        try:
            # Generate embedding
            embedding = self.semantic_model.encode([text])[0]
            
            # Cache for future use
            self.embeddings_cache[text_hash] = embedding
            
            return embedding
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None
    
    async def _calculate_semantic_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if not SCIPY_AVAILABLE:
            # Fallback to basic dot product similarity
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        try:
            return 1.0 - cosine(embedding1, embedding2)
        except Exception:
            return 0.0
    
    # Pattern detection methods
    
    async def _detect_contradiction(self, statement1: str, statement2: str) -> bool:
        """Detect if two statements contradict each other"""
        # Simple contradiction detection
        contradiction_indicators = [
            ("not", ""), ("isn't", "is"), ("cannot", "can"), ("never", "always"),
            ("impossible", "possible"), ("false", "true"), ("no", "yes")
        ]
        
        s1_lower = statement1.lower()
        s2_lower = statement2.lower()
        
        for neg, pos in contradiction_indicators:
            if (neg in s1_lower and pos in s2_lower) or (pos in s1_lower and neg in s2_lower):
                return True
        
        return False
    
    async def _detect_implications(self, premises: List[ReasoningPremise]) -> List[Dict[str, Any]]:
        """Detect implication patterns in premises"""
        implications = []
        
        for premise in premises:
            statement = premise.statement.lower()
            
            # Look for conditional patterns
            for pattern_type, patterns in self.reasoning_patterns.items():
                for pattern in patterns:
                    if pattern in statement:
                        implications.append({
                            "type": pattern_type,
                            "pattern": pattern,
                            "premise": premise.statement,
                            "confidence": premise.confidence * 0.8  # Reduce confidence for inferred implications
                        })
        
        return implications
    
    async def _cluster_premises(self, premises: List[ReasoningPremise]) -> List[List[int]]:
        """Cluster similar premises using semantic similarity"""
        if not self.semantic_model or len(premises) < 2:
            return []
        
        clusters = []
        used_indices = set()
        
        for i, premise1 in enumerate(premises):
            if i in used_indices or premise1.semantic_embedding is None:
                continue
            
            cluster = [i]
            used_indices.add(i)
            
            for j, premise2 in enumerate(premises[i+1:], i+1):
                if j in used_indices or premise2.semantic_embedding is None:
                    continue
                
                similarity = await self._calculate_semantic_similarity(
                    premise1.semantic_embedding,
                    premise2.semantic_embedding
                )
                
                if similarity > self.similarity_threshold:
                    cluster.append(j)
                    used_indices.add(j)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    # Inference generation methods
    
    async def _generate_similarity_inference(self, premise1: ReasoningPremise, premise2: ReasoningPremise, similarity: float) -> Optional[Dict[str, Any]]:
        """Generate inference from similar premises"""
        if similarity < 0.8:
            return None
        
        return {
            "statement": f"Based on similar premises, there is strong evidence supporting both {premise1.statement[:50]}... and {premise2.statement[:50]}...",
            "type": "similarity",
            "confidence": min(premise1.confidence, premise2.confidence) * similarity,
            "supporting_premises": [premise1.statement, premise2.statement],
            "reasoning_path": [f"Similar premises detected (similarity: {similarity:.2f})"]
        }
    
    async def _generate_implication_inference(self, implication: Dict[str, Any], premises: List[ReasoningPremise]) -> Optional[Dict[str, Any]]:
        """Generate inference from implication patterns"""
        if implication["confidence"] < 0.5:
            return None
        
        return {
            "statement": f"Pattern-based inference from {implication['type']} relationship",
            "type": "implication",
            "confidence": implication["confidence"],
            "supporting_premises": [implication["premise"]],
            "reasoning_path": [f"Detected {implication['pattern']} pattern"]
        }
    
    async def _generate_causal_inferences(self, premises: List[ReasoningPremise], context: CognitiveContext) -> List[Dict[str, Any]]:
        """Generate causal inferences from premises"""
        inferences = []
        
        causal_patterns = self.reasoning_patterns["causal"]
        
        for premise in premises:
            statement_lower = premise.statement.lower()
            
            for pattern in causal_patterns:
                if pattern in statement_lower:
                    # Extract cause and effect
                    parts = statement_lower.split(pattern)
                    if len(parts) == 2:
                        cause = parts[0].strip()
                        effect = parts[1].strip()
                        
                        inferences.append({
                            "statement": f"Causal relationship identified: {cause} leads to {effect}",
                            "type": "causal",
                            "confidence": premise.confidence * 0.7,
                            "supporting_premises": [premise.statement],
                            "reasoning_path": [f"Causal pattern '{pattern}' detected"]
                        })
        
        return inferences
    
    async def _generate_conditional_inferences(self, premises: List[ReasoningPremise], context: CognitiveContext) -> List[Dict[str, Any]]:
        """Generate conditional inferences from premises"""
        inferences = []
        
        conditional_patterns = self.reasoning_patterns["conditional"]
        
        for premise in premises:
            statement_lower = premise.statement.lower()
            
            for pattern in conditional_patterns:
                if pattern in statement_lower:
                    inferences.append({
                        "statement": f"Conditional relationship identified in: {premise.statement[:100]}...",
                        "type": "conditional",
                        "confidence": premise.confidence * 0.6,
                        "supporting_premises": [premise.statement],
                        "reasoning_path": [f"Conditional pattern '{pattern}' detected"]
                    })
        
        return inferences
    
    # Strategy planning methods
    
    async def _analyze_goal_semantically(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze goal using semantic understanding"""
        goal_text = goal.get("description", str(goal))
        
        analysis = {
            "goal_type": await self._classify_goal_type(goal_text),
            "complexity": await self._estimate_goal_complexity(goal_text),
            "semantic_embedding": await self._get_embedding(goal_text) if self.semantic_model else None,
            "keywords": await self._extract_goal_keywords(goal_text),
            "success_criteria": goal.get("success_criteria", [])
        }
        
        return analysis
    
    async def _analyze_constraints(self, constraints: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze constraints for strategy planning"""
        analysis = {
            "constraint_types": [],
            "severity_levels": [],
            "resource_constraints": [],
            "time_constraints": [],
            "quality_constraints": []
        }
        
        for constraint in constraints:
            constraint_text = constraint.get("description", str(constraint))
            constraint_type = await self._classify_constraint_type(constraint_text)
            severity = constraint.get("severity", "medium")
            
            analysis["constraint_types"].append(constraint_type)
            analysis["severity_levels"].append(severity)
            
            if "resource" in constraint_type or "memory" in constraint_type or "cpu" in constraint_type:
                analysis["resource_constraints"].append(constraint)
            elif "time" in constraint_type or "deadline" in constraint_type:
                analysis["time_constraints"].append(constraint)
            elif "quality" in constraint_type or "accuracy" in constraint_type:
                analysis["quality_constraints"].append(constraint)
        
        return analysis
    
    async def _find_similar_strategies(self, goal: Dict[str, Any], constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find similar successful strategies from history"""
        # Placeholder for strategy matching based on semantic similarity
        # In a full implementation, this would search through historical strategies
        return []
    
    async def _generate_strategy_alternatives(self, goal_analysis: Dict[str, Any], constraint_analysis: Dict[str, Any], similar_strategies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate alternative strategies"""
        strategies = []
        
        goal_type = goal_analysis.get("goal_type", "general")
        complexity = goal_analysis.get("complexity", "medium")
        
        # Generate strategies based on goal type
        if goal_type == "file_operation":
            strategies.extend(await self._generate_file_operation_strategies(goal_analysis, constraint_analysis))
        elif goal_type == "code_analysis":
            strategies.extend(await self._generate_code_analysis_strategies(goal_analysis, constraint_analysis))
        elif goal_type == "system_operation":
            strategies.extend(await self._generate_system_operation_strategies(goal_analysis, constraint_analysis))
        else:
            strategies.extend(await self._generate_generic_strategies(goal_analysis, constraint_analysis))
        
        return strategies
    
    async def _evaluate_strategies(self, strategies: List[Dict[str, Any]], goal: Dict[str, Any], constraints: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate and rank strategies"""
        evaluated = []
        
        for strategy in strategies:
            score = await self._calculate_strategy_score(strategy, goal, constraints)
            strategy["score"] = score
            strategy["confidence"] = score  # For now, use score as confidence
            evaluated.append(strategy)
        
        # Sort by score (highest first)
        evaluated.sort(key=lambda s: s["score"], reverse=True)
        
        return evaluated
    
    # Helper methods for strategy planning
    
    async def _classify_goal_type(self, goal_text: str) -> str:
        """Classify the type of goal"""
        goal_lower = goal_text.lower()
        
        if any(word in goal_lower for word in ["file", "create", "read", "write", "copy", "move"]):
            return "file_operation"
        elif any(word in goal_lower for word in ["analyze", "review", "code", "function", "class"]):
            return "code_analysis"
        elif any(word in goal_lower for word in ["install", "configure", "start", "stop", "system"]):
            return "system_operation"
        elif any(word in goal_lower for word in ["organize", "clean", "structure", "arrange"]):
            return "organization"
        else:
            return "general"
    
    async def _estimate_goal_complexity(self, goal_text: str) -> str:
        """Estimate the complexity of a goal"""
        word_count = len(goal_text.split())
        
        if word_count < 5:
            return "low"
        elif word_count < 15:
            return "medium"
        else:
            return "high"
    
    async def _extract_goal_keywords(self, goal_text: str) -> List[str]:
        """Extract keywords from goal text"""
        # Simple keyword extraction
        words = goal_text.lower().split()
        keywords = [word for word in words if len(word) > 3]
        return keywords[:10]  # Limit to top 10
    
    async def _classify_constraint_type(self, constraint_text: str) -> str:
        """Classify the type of constraint"""
        constraint_lower = constraint_text.lower()
        
        if any(word in constraint_lower for word in ["time", "deadline", "urgent", "fast"]):
            return "time_constraint"
        elif any(word in constraint_lower for word in ["memory", "cpu", "resource", "space"]):
            return "resource_constraint"
        elif any(word in constraint_lower for word in ["quality", "accuracy", "precision", "correct"]):
            return "quality_constraint"
        else:
            return "general_constraint"
    
    async def _generate_file_operation_strategies(self, goal_analysis: Dict[str, Any], constraint_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategies for file operations"""
        return [
            {
                "name": "Direct File Operation",
                "description": "Perform file operation directly with minimal steps",
                "steps": ["validate_path", "perform_operation", "verify_result"],
                "estimated_time": 3,
                "resource_usage": "low"
            },
            {
                "name": "Safe File Operation",
                "description": "Perform file operation with backup and validation",
                "steps": ["backup_if_exists", "validate_path", "perform_operation", "verify_result", "cleanup"],
                "estimated_time": 8,
                "resource_usage": "medium"
            }
        ]
    
    async def _generate_code_analysis_strategies(self, goal_analysis: Dict[str, Any], constraint_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategies for code analysis"""
        return [
            {
                "name": "Quick Code Scan",
                "description": "Perform basic syntax and structure analysis",
                "steps": ["parse_syntax", "analyze_structure", "generate_report"],
                "estimated_time": 5,
                "resource_usage": "low"
            },
            {
                "name": "Comprehensive Code Analysis",
                "description": "Perform detailed analysis including security and quality metrics",
                "steps": ["parse_syntax", "analyze_structure", "security_scan", "quality_metrics", "performance_analysis", "generate_report"],
                "estimated_time": 20,
                "resource_usage": "high"
            }
        ]
    
    async def _generate_system_operation_strategies(self, goal_analysis: Dict[str, Any], constraint_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate strategies for system operations"""
        return [
            {
                "name": "Direct System Command",
                "description": "Execute system command directly",
                "steps": ["validate_command", "execute", "check_result"],
                "estimated_time": 5,
                "resource_usage": "medium"
            },
            {
                "name": "Safe System Operation",
                "description": "Execute system command with precautions and rollback capability",
                "steps": ["backup_state", "validate_command", "execute", "verify_result", "rollback_if_failed"],
                "estimated_time": 15,
                "resource_usage": "high"
            }
        ]
    
    async def _generate_generic_strategies(self, goal_analysis: Dict[str, Any], constraint_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate generic strategies"""
        return [
            {
                "name": "Sequential Approach",
                "description": "Break down goal into sequential steps",
                "steps": ["analyze_goal", "plan_steps", "execute_sequentially", "validate_result"],
                "estimated_time": 10,
                "resource_usage": "medium"
            },
            {
                "name": "Parallel Approach",
                "description": "Execute independent parts in parallel",
                "steps": ["analyze_goal", "identify_parallel_parts", "execute_parallel", "combine_results"],
                "estimated_time": 7,
                "resource_usage": "high"
            }
        ]
    
    async def _calculate_strategy_score(self, strategy: Dict[str, Any], goal: Dict[str, Any], constraints: List[Dict[str, Any]]) -> float:
        """Calculate score for a strategy"""
        base_score = 0.5
        
        # Factor in estimated time vs constraints
        estimated_time = strategy.get("estimated_time", 10)
        if any("time" in str(c).lower() for c in constraints):
            if estimated_time < 10:
                base_score += 0.3
            elif estimated_time > 20:
                base_score -= 0.2
        
        # Factor in resource usage vs constraints
        resource_usage = strategy.get("resource_usage", "medium")
        if any("resource" in str(c).lower() for c in constraints):
            if resource_usage == "low":
                base_score += 0.2
            elif resource_usage == "high":
                base_score -= 0.1
        
        # Factor in number of steps (complexity)
        steps = strategy.get("steps", [])
        if len(steps) > 10:
            base_score -= 0.1
        elif len(steps) < 3:
            base_score -= 0.1
        
        return max(0.0, min(1.0, base_score))
    
    # Learning methods
    
    async def _learn_from_reasoning(self, premises: List[ReasoningPremise], conclusions: List[ReasoningConclusion], confidence: float) -> None:
        """Learn from reasoning outcomes"""
        reasoning_record = {
            "premises_count": len(premises),
            "conclusions_count": len(conclusions),
            "overall_confidence": confidence,
            "timestamp": datetime.now(),
            "semantic_model_used": self.semantic_model is not None
        }
        
        self.reasoning_history.append(reasoning_record)
        
        # Learn successful patterns
        if confidence > 0.8:
            for conclusion in conclusions:
                pattern_key = conclusion.metadata.get("inference_type", "unknown")
                if pattern_key not in self.successful_patterns:
                    self.successful_patterns[pattern_key] = []
                
                self.successful_patterns[pattern_key].append({
                    "conclusion": conclusion.conclusion,
                    "confidence": conclusion.confidence,
                    "semantic_support": conclusion.semantic_support
                })
        
        logger.debug(f"🎓 Learned from reasoning session: confidence={confidence:.2f}")
    
    # Utility methods
    
    async def _calculate_inference_confidence(self, inference: Dict[str, Any], premises: List[ReasoningPremise]) -> float:
        """Calculate confidence for an inference"""
        base_confidence = inference.get("confidence", 0.5)
        
        # Factor in supporting premises
        supporting_premises = inference.get("supporting_premises", [])
        if supporting_premises:
            premise_confidences = []
            for premise in premises:
                if premise.statement in supporting_premises:
                    premise_confidences.append(premise.confidence)
            
            if premise_confidences:
                avg_premise_confidence = sum(premise_confidences) / len(premise_confidences)
                base_confidence = (base_confidence + avg_premise_confidence) / 2
        
        return min(1.0, base_confidence)
    
    async def _calculate_semantic_support(self, inference: Dict[str, Any], premises: List[ReasoningPremise]) -> float:
        """Calculate semantic support for an inference"""
        if not self.semantic_model:
            return 0.0
        
        try:
            inference_embedding = await self._get_embedding(inference["statement"])
            if inference_embedding is None:
                return 0.0
            
            similarities = []
            for premise in premises:
                if premise.semantic_embedding is not None:
                    similarity = await self._calculate_semantic_similarity(
                        inference_embedding,
                        premise.semantic_embedding
                    )
                    similarities.append(similarity)
            
            return max(similarities) if similarities else 0.0
        except Exception as e:
            logger.warning(f"Failed to calculate semantic support: {e}")
            return 0.0


# Register the semantic reasoning engine
semantic_reasoning_engine = SemanticReasoningEngine()
from .base_cognitive import cognitive_registry
cognitive_registry.register_reasoning_engine(semantic_reasoning_engine)
