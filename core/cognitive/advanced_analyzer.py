"""
PROJECT-S V2 - Advanced Intent Analyzer
=======================================

This module provides advanced intent detection and analysis capabilities for the v2 system,
building on the existing intelligence_engine.py to provide enhanced natural language understanding
with confidence scoring, pattern matching, and semantic understanding.

Key Features:
1. Enhanced intent confidence scoring (0.0-1.0)
2. Fuzzy string matching for partial commands
3. Pattern strength analysis with context awareness
4. Multi-language support (Hungarian-English)
5. Learning from successful intent detections
6. Integration with semantic similarity engines

Legacy Integration:
- Maintains compatibility with existing intelligent_command_parser functionality
- Enhances capabilities while preserving behavior patterns
- Provides migration path for existing command patterns
"""

import logging
import re
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from difflib import SequenceMatcher
from dataclasses import dataclass, field

from .base_cognitive import (
    BaseIntentAnalyzer,
    IntentMatch,
    CognitiveContext
)

logger = logging.getLogger(__name__)


@dataclass
class PatternMatch:
    """Represents a pattern match with metadata"""
    pattern: str
    weight: float
    match_type: str  # "exact", "fuzzy", "semantic"
    position: int
    context_relevance: float = 0.0


class AdvancedIntentAnalyzer(BaseIntentAnalyzer):
    """
    Advanced intent analyzer with enhanced pattern matching and confidence scoring.
    
    This analyzer provides sophisticated intent detection capabilities including:
    - Multi-language pattern recognition
    - Confidence-based thresholding
    - Contextual understanding
    - Learning from successful detections
    - Fuzzy matching for partial commands
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced intent analyzer"""
        super().__init__("advanced_intent_analyzer", config)
        
        # Configuration
        self.confidence_threshold = config.get("confidence_threshold", 0.3) if config else 0.3
        self.fuzzy_match_threshold = config.get("fuzzy_match_threshold", 0.6) if config else 0.6
        self.max_intents_returned = config.get("max_intents_returned", 5) if config else 5
        
        # Enhanced pattern databases with weights and variations
        self.intent_patterns = {
            "DIRECTORY_ORGANIZATION": {
                "primary_patterns": [
                    ("rendszerezd", 1.0), ("szervezd", 1.0), ("organize", 1.0), 
                    ("rendezd", 0.9), ("kategorizáld", 0.9), ("clean up", 0.8),
                    ("tidy", 0.7), ("arrange", 0.7), ("strukturáld", 0.8)
                ],
                "context_words": [
                    ("mappát", 0.3), ("folder", 0.3), ("mappa", 0.3), 
                    ("directory", 0.3), ("files", 0.2), ("fájlok", 0.2),
                    ("downloads", 0.4), ("letöltések", 0.4)
                ],
                "negative_patterns": ["ne", "don't", "stop", "ne szervezd", "cancel"],
                "parameters": ["path", "strategy", "criteria"]
            },
            
            "FILE_OPERATION": {
                "create_patterns": [
                    ("hozz létre", 1.0), ("create file", 1.0), ("létrehozz", 0.9),
                    ("készíts", 0.9), ("new file", 0.8), ("make", 0.7),
                    ("generate", 0.6), ("build", 0.5), ("add", 0.4)
                ],
                "read_patterns": [
                    ("olvasd", 1.0), ("read file", 1.0), ("mutasd", 0.9),
                    ("tartalom", 0.8), ("show", 0.7), ("display", 0.7),
                    ("view", 0.6), ("cat", 0.8), ("type", 0.6), ("open", 0.5)
                ],
                "list_patterns": [
                    ("listázd", 1.0), ("list", 1.0), ("ls", 0.9), ("dir", 0.9),
                    ("show files", 0.8), ("fájlok", 0.7), ("contents", 0.6),
                    ("enumerate", 0.5)
                ],
                "copy_patterns": [
                    ("másold", 1.0), ("copy", 1.0), ("duplicate", 0.8),
                    ("clone", 0.7), ("replicate", 0.6)
                ],
                "move_patterns": [
                    ("mozgatd", 1.0), ("move", 1.0), ("relocate", 0.8),
                    ("transfer", 0.7), ("shift", 0.6)
                ],
                "delete_patterns": [
                    ("töröld", 1.0), ("delete", 1.0), ("remove", 0.9),
                    ("erase", 0.8), ("unlink", 0.7), ("destroy", 0.6)
                ],
                "context_words": [
                    ("fájlt", 0.3), ("file", 0.3), ("document", 0.2),
                    ("text", 0.2), (".txt", 0.4), (".py", 0.4), (".json", 0.4),
                    ("folder", 0.2), ("directory", 0.2)
                ],
                "parameters": ["path", "content", "destination", "options"]
            },
            
            "SHELL_COMMAND": {
                "primary_patterns": [
                    ("futtat", 1.0), ("execute", 1.0), ("run", 1.0),
                    ("powershell", 0.9), ("cmd", 0.9), ("command", 0.8),
                    ("terminal", 0.7), ("shell", 0.7), ("invoke", 0.6)
                ],
                "context_words": [
                    ("parancs", 0.2), ("script", 0.2), ("batch", 0.2),
                    ("executable", 0.3), ("program", 0.2)
                ],
                "parameters": ["command", "arguments", "working_directory"]
            },
            
            "CODE_ANALYSIS": {
                "primary_patterns": [
                    ("elemezd", 1.0), ("analyze", 1.0), ("review", 0.9),
                    ("check", 0.8), ("examine", 0.8), ("inspect", 0.7),
                    ("audit", 0.7), ("scan", 0.6), ("validate", 0.6)
                ],
                "context_words": [
                    ("kód", 0.4), ("code", 0.4), ("function", 0.3),
                    ("class", 0.3), ("method", 0.3), ("variable", 0.2),
                    ("syntax", 0.3), ("logic", 0.3), ("performance", 0.2)
                ],
                "analysis_types": [
                    ("security", 0.8), ("quality", 0.8), ("performance", 0.7),
                    ("syntax", 0.6), ("style", 0.5), ("complexity", 0.7)
                ],
                "parameters": ["path", "analysis_type", "language", "depth"]
            },
            
            "SYSTEM_OPERATION": {
                "primary_patterns": [
                    ("install", 1.0), ("telepíts", 1.0), ("setup", 0.9),
                    ("configure", 0.9), ("start", 0.8), ("stop", 0.8),
                    ("restart", 0.8), ("enable", 0.7), ("disable", 0.7)
                ],
                "context_words": [
                    ("service", 0.3), ("szolgáltatás", 0.3), ("process", 0.3),
                    ("application", 0.2), ("software", 0.2), ("package", 0.3)
                ],
                "parameters": ["target", "operation", "options", "configuration"]
            },
            
            "QUERY": {
                "primary_patterns": [
                    ("what", 1.0), ("mi", 1.0), ("how", 1.0), ("hogyan", 1.0),
                    ("why", 0.9), ("miért", 0.9), ("when", 0.8), ("mikor", 0.8),
                    ("where", 0.8), ("hol", 0.8), ("which", 0.7), ("melyik", 0.7),
                    ("explain", 0.9), ("magyarázd", 0.9), ("tell me", 0.8)
                ],
                "context_words": [
                    ("about", 0.2), ("regarding", 0.2), ("concerning", 0.2),
                    ("információ", 0.3), ("details", 0.2), ("explanation", 0.3)
                ],
                "parameters": ["topic", "detail_level", "context"]
            }
        }
        
        # Semantic similarity patterns (if available)
        self.semantic_patterns = {}
        
        # Learning data
        self.success_patterns = {}
        self.failure_patterns = {}
        self.context_patterns = {}
        
        logger.info("Advanced intent analyzer initialized")
    
    async def detect_intent(self, input_text: str, context: Optional[CognitiveContext] = None) -> List[IntentMatch]:
        """
        Detect intents in user input using advanced pattern matching.
        
        Args:
            input_text: Text to analyze
            context: Optional cognitive context for enhanced detection
            
        Returns:
            List of detected intents with confidence scores
        """
        if not input_text or not input_text.strip():
            return []
        
        intents = []
        input_clean = input_text.strip().lower()
        
        logger.debug(f"🔍 Analyzing intent for: '{input_text[:100]}...'")
        
        # Analyze against each intent type
        for intent_type, patterns in self.intent_patterns.items():
            intent_match = await self._analyze_intent_type(input_clean, intent_type, patterns, context)
            if intent_match and intent_match.confidence >= self.confidence_threshold:
                intents.append(intent_match)
        
        # Sort by total confidence (including context boost)
        intents.sort(key=lambda x: x.total_confidence, reverse=True)
        
        # Limit results
        intents = intents[:self.max_intents_returned]
        
        # Learn from high-confidence detections
        if intents and intents[0].total_confidence > 0.8:
            await self.learn_pattern(input_text, intents[0].intent_type, intents[0].total_confidence)
        
        logger.info(f"🎯 Detected {len(intents)} intents with confidence >= {self.confidence_threshold}")
        return intents
    
    async def extract_parameters(self, input_text: str, intent_type: str) -> Dict[str, Any]:
        """
        Extract parameters for a specific intent type.
        
        Args:
            input_text: Text to extract parameters from
            intent_type: Type of intent to extract parameters for
            
        Returns:
            Dictionary of extracted parameters
        """
        parameters = {}
        input_clean = input_text.strip().lower()
        
        # Get parameter patterns for this intent type
        intent_config = self.intent_patterns.get(intent_type, {})
        expected_params = intent_config.get("parameters", [])
        
        logger.debug(f"🔧 Extracting parameters for {intent_type}: {expected_params}")
        
        # Extract based on intent type
        if intent_type == "FILE_OPERATION":
            parameters.update(await self._extract_file_operation_params(input_text, input_clean))
        elif intent_type == "DIRECTORY_ORGANIZATION":
            parameters.update(await self._extract_directory_params(input_text, input_clean))
        elif intent_type == "CODE_ANALYSIS":
            parameters.update(await self._extract_code_analysis_params(input_text, input_clean))
        elif intent_type == "SHELL_COMMAND":
            parameters.update(await self._extract_shell_command_params(input_text, input_clean))
        elif intent_type == "SYSTEM_OPERATION":
            parameters.update(await self._extract_system_operation_params(input_text, input_clean))
        elif intent_type == "QUERY":
            parameters.update(await self._extract_query_params(input_text, input_clean))
        
        # Extract common parameters
        parameters.update(await self._extract_common_params(input_text, input_clean))
        
        logger.debug(f"📋 Extracted parameters: {parameters}")
        return parameters
    
    async def _analyze_intent_type(self, input_text: str, intent_type: str, patterns: Dict[str, Any], context: Optional[CognitiveContext]) -> Optional[IntentMatch]:
        """Analyze input against a specific intent type"""
        
        total_confidence = 0.0
        matched_patterns = []
        best_operation = "execute"
        
        # Check primary patterns
        primary_patterns = patterns.get("primary_patterns", [])
        if not primary_patterns:
            # Handle different pattern structures
            pattern_keys = [k for k in patterns.keys() if k.endswith("_patterns")]
            all_patterns = []
            for key in pattern_keys:
                all_patterns.extend(patterns[key])
                # Extract operation from pattern key
                if key in input_text:
                    best_operation = key.replace("_patterns", "")
            primary_patterns = all_patterns
        
        for pattern, weight in primary_patterns:
            confidence = await self._calculate_pattern_confidence(input_text, pattern, weight)
            if confidence > 0:
                total_confidence += confidence
                matched_patterns.append(pattern)
                
                # Extract operation from pattern
                if intent_type == "FILE_OPERATION":
                    if any(op in pattern for op in ["create", "hozz létre", "létrehozz"]):
                        best_operation = "create"
                    elif any(op in pattern for op in ["read", "olvasd", "mutasd"]):
                        best_operation = "read"
                    elif any(op in pattern for op in ["copy", "másold"]):
                        best_operation = "copy"
                    elif any(op in pattern for op in ["move", "mozgatd"]):
                        best_operation = "move"
                    elif any(op in pattern for op in ["delete", "töröld"]):
                        best_operation = "delete"
                    elif any(op in pattern for op in ["list", "listázd"]):
                        best_operation = "list"
        
        # Check context words for additional confidence
        context_words = patterns.get("context_words", [])
        for word, weight in context_words:
            if word in input_text:
                total_confidence += weight
                matched_patterns.append(f"context:{word}")
        
        # Check for negative patterns
        negative_patterns = patterns.get("negative_patterns", [])
        for neg_pattern in negative_patterns:
            if neg_pattern in input_text:
                total_confidence *= 0.3  # Significantly reduce confidence
                matched_patterns.append(f"negative:{neg_pattern}")
        
        # Apply fuzzy matching for partial matches
        fuzzy_bonus = await self._calculate_fuzzy_confidence(input_text, primary_patterns)
        total_confidence += fuzzy_bonus
        
        # Normalize confidence to [0, 1]
        total_confidence = min(1.0, total_confidence)
        
        # Calculate context boost
        context_boost = 0.0
        if context:
            context_boost = await self._calculate_context_boost(intent_type, context)
        
        # Check if confidence meets threshold
        if total_confidence < self.confidence_threshold and (total_confidence + context_boost) < self.confidence_threshold:
            return None
        
        # Extract parameters
        parameters = await self.extract_parameters(input_text, intent_type)
        
        return IntentMatch(
            intent_type=intent_type,
            confidence=total_confidence,
            operation=best_operation,
            parameters=parameters,
            patterns_matched=matched_patterns,
            context_boost=context_boost,
            metadata={
                "input_length": len(input_text),
                "pattern_count": len(matched_patterns),
                "fuzzy_bonus": fuzzy_bonus,
                "analyzed_at": datetime.now().isoformat()
            }
        )
    
    async def _calculate_pattern_confidence(self, input_text: str, pattern: str, weight: float) -> float:
        """Calculate confidence for a specific pattern match"""
        if pattern in input_text:
            # Exact match
            return weight * 1.0
        
        # Fuzzy match
        similarity = SequenceMatcher(None, input_text, pattern).ratio()
        if similarity >= self.fuzzy_match_threshold:
            return weight * similarity * 0.8  # Reduce weight for fuzzy matches
        
        return 0.0
    
    async def _calculate_fuzzy_confidence(self, input_text: str, patterns: List[Tuple[str, float]]) -> float:
        """Calculate additional confidence from fuzzy matching"""
        best_fuzzy = 0.0
        
        for pattern, weight in patterns:
            similarity = SequenceMatcher(None, input_text, pattern).ratio()
            if similarity >= self.fuzzy_match_threshold:
                fuzzy_score = weight * similarity * 0.3  # Lower weight for fuzzy
                best_fuzzy = max(best_fuzzy, fuzzy_score)
        
        return best_fuzzy
    
    async def _calculate_context_boost(self, intent_type: str, context: CognitiveContext) -> float:
        """Calculate confidence boost based on context"""
        boost = 0.0
        
        # Recent task history boost
        recent_tasks = list(context.completed_tasks)[-3:]  # Last 3 tasks
        if any(intent_type.lower() in str(task).lower() for task in recent_tasks):
            boost += 0.15
        
        # Current workspace context
        if intent_type == "FILE_OPERATION" and "current_directory" in context.workspace:
            boost += 0.1
        
        # Conversation context
        recent_messages = context.conversation_history[-2:]  # Last 2 messages
        for message in recent_messages:
            if intent_type.lower() in str(message).lower():
                boost += 0.1
                break
        
        return min(0.3, boost)  # Cap at 0.3
    
    # Parameter extraction methods
    
    async def _extract_file_operation_params(self, original_text: str, clean_text: str) -> Dict[str, Any]:
        """Extract parameters for file operations"""
        params = {}
        
        # Extract file paths
        # Look for quoted paths
        quoted_paths = re.findall(r'["\']([^"\']+)["\']', original_text)
        if quoted_paths:
            params["path"] = quoted_paths[0]
        else:
            # Look for file extensions or path indicators
            path_patterns = [
                r'(\S+\.\w{2,4})',  # Files with extensions
                r'([A-Za-z]:[\\\/]\S+)',  # Windows absolute paths
                r'(\/\S+)',  # Unix absolute paths
                r'(\.\S+)',  # Relative paths starting with .
            ]
            
            for pattern in path_patterns:
                matches = re.findall(pattern, original_text)
                if matches:
                    params["path"] = matches[0]
                    break
        
        # Extract destination for copy/move operations
        if any(op in clean_text for op in ["copy", "move", "másold", "mozgatd"]):
            # Look for "to" keyword followed by path
            to_match = re.search(r'\b(?:to|into|be|-ba|-be)\s+(["\']?[^"\']+["\']?)', original_text, re.IGNORECASE)
            if to_match:
                params["destination"] = to_match.group(1).strip('\'"')
        
        # Extract content for create operations
        if any(op in clean_text for op in ["create", "write", "hozz létre"]):
            # Look for "with content" or similar
            content_match = re.search(r'\b(?:with content|content|tartalom|szöveg)\s*[:=]?\s*(["\']?[^"\']+["\']?)', original_text, re.IGNORECASE)
            if content_match:
                params["content"] = content_match.group(1).strip('\'"')
        
        return params
    
    async def _extract_directory_params(self, original_text: str, clean_text: str) -> Dict[str, Any]:
        """Extract parameters for directory organization"""
        params = {}
        
        # Extract target directory
        if "path" not in params:
            # Look for directory indicators
            dir_patterns = [
                r'(?:mappát|folder|directory|mappa)\s+(["\']?[^"\']+["\']?)',
                r'([A-Za-z]:[\\\/][^\\\/\s]+)',  # Windows paths
                r'(\/[^\/\s]+)',  # Unix paths
            ]
            
            for pattern in dir_patterns:
                matches = re.findall(pattern, original_text, re.IGNORECASE)
                if matches:
                    params["path"] = matches[0].strip('\'"')
                    break
        
        # Extract organization strategy
        strategy_map = {
            "type": ["típus", "fajta", "extension", "kiterjesztés"],
            "date": ["dátum", "időpont", "created", "modified"],
            "size": ["méret", "nagyság"],
            "name": ["név", "alfabetical", "ábécé"],
            "auto": ["automatikus", "automatic", "smart", "okos"]
        }
        
        for strategy, keywords in strategy_map.items():
            if any(keyword in clean_text for keyword in keywords):
                params["strategy"] = strategy
                break
        
        if "strategy" not in params:
            params["strategy"] = "auto"  # Default
        
        return params
    
    async def _extract_code_analysis_params(self, original_text: str, clean_text: str) -> Dict[str, Any]:
        """Extract parameters for code analysis"""
        params = {}
        
        # Extract file/directory path
        code_patterns = [
            r'(\S+\.(?:py|js|ts|java|cpp|c|h|cs|php|rb|go|rs))',  # Code files
            r'(["\']?[^"\']*(?:src|source|code)[^"\']*["\']?)',  # Source directories
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, original_text, re.IGNORECASE)
            if matches:
                params["path"] = matches[0].strip('\'"')
                break
        
        # Extract analysis type
        analysis_types = {
            "security": ["security", "biztonság", "vulnerability", "sérülékenység"],
            "quality": ["quality", "minőség", "code quality", "style"],
            "performance": ["performance", "teljesítmény", "optimization", "optimalizálás"],
            "complexity": ["complexity", "komplexitás", "cyclomatic"],
            "syntax": ["syntax", "szintaxis", "grammar", "nyelvtan"]
        }
        
        for analysis_type, keywords in analysis_types.items():
            if any(keyword in clean_text for keyword in keywords):
                params["analysis_type"] = analysis_type
                break
        
        if "analysis_type" not in params:
            params["analysis_type"] = "general"
        
        return params
    
    async def _extract_shell_command_params(self, original_text: str, clean_text: str) -> Dict[str, Any]:
        """Extract parameters for shell commands"""
        params = {}
        
        # Extract command
        cmd_patterns = [
            r'(?:run|execute|futtat)\s+["\']?([^"\']+)["\']?',
            r'powershell\s+([^"\']+)',
            r'cmd\s+([^"\']+)',
        ]
        
        for pattern in cmd_patterns:
            matches = re.findall(pattern, original_text, re.IGNORECASE)
            if matches:
                params["command"] = matches[0].strip()
                break
        
        # Extract arguments
        if "command" in params:
            cmd_parts = params["command"].split()
            if len(cmd_parts) > 1:
                params["arguments"] = cmd_parts[1:]
                params["command"] = cmd_parts[0]
        
        return params
    
    async def _extract_system_operation_params(self, original_text: str, clean_text: str) -> Dict[str, Any]:
        """Extract parameters for system operations"""
        params = {}
        
        # Extract operation type
        operations = {
            "install": ["install", "telepíts", "setup"],
            "start": ["start", "indíts", "run"],
            "stop": ["stop", "állíts meg", "halt"],
            "restart": ["restart", "újraindíts"],
            "configure": ["configure", "beállíts", "setup"]
        }
        
        for operation, keywords in operations.items():
            if any(keyword in clean_text for keyword in keywords):
                params["operation"] = operation
                break
        
        # Extract target
        service_patterns = [
            r'(?:service|szolgáltatás)\s+(["\']?[^"\']+["\']?)',
            r'(?:process|folyamat)\s+(["\']?[^"\']+["\']?)',
            r'(?:application|alkalmazás)\s+(["\']?[^"\']+["\']?)',
        ]
        
        for pattern in service_patterns:
            matches = re.findall(pattern, original_text, re.IGNORECASE)
            if matches:
                params["target"] = matches[0].strip('\'"')
                break
        
        return params
    
    async def _extract_query_params(self, original_text: str, clean_text: str) -> Dict[str, Any]:
        """Extract parameters for queries"""
        params = {}
        
        # Extract topic/subject
        question_patterns = [
            r'(?:what|mi)\s+(?:is|van|about|regarding)\s+([^?]+)',
            r'(?:how|hogyan)\s+(?:to|do|can|tudom)\s+([^?]+)',
            r'(?:why|miért)\s+([^?]+)',
            r'(?:when|mikor)\s+([^?]+)',
            r'(?:where|hol)\s+([^?]+)',
        ]
        
        for pattern in question_patterns:
            matches = re.findall(pattern, original_text, re.IGNORECASE)
            if matches:
                params["topic"] = matches[0].strip()
                break
        
        # Determine detail level
        if any(word in clean_text for word in ["detailed", "részletes", "comprehensive", "átfogó"]):
            params["detail_level"] = "detailed"
        elif any(word in clean_text for word in ["brief", "rövid", "quick", "gyors", "summary", "összefoglaló"]):
            params["detail_level"] = "brief"
        else:
            params["detail_level"] = "normal"
        
        return params
    
    async def _extract_common_params(self, original_text: str, clean_text: str) -> Dict[str, Any]:
        """Extract common parameters that apply to multiple intent types"""
        params = {}
        
        # Extract options/flags
        flag_patterns = [
            r'--(\w+)',  # Long flags
            r'-(\w)',    # Short flags
        ]
        
        flags = []
        for pattern in flag_patterns:
            flags.extend(re.findall(pattern, original_text))
        
        if flags:
            params["flags"] = flags
        
        # Extract urgency/priority
        if any(word in clean_text for word in ["urgent", "sürgős", "critical", "kritikus", "asap"]):
            params["priority"] = "high"
        elif any(word in clean_text for word in ["low priority", "alacsony", "later", "később"]):
            params["priority"] = "low"
        else:
            params["priority"] = "normal"
        
        return params


# Register the advanced intent analyzer
advanced_intent_analyzer = AdvancedIntentAnalyzer()
from .base_cognitive import cognitive_registry
cognitive_registry.register_analyzer(advanced_intent_analyzer)
