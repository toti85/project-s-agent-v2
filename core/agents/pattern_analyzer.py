#!/usr/bin/env python3
"""
Pattern Analyzer for OverseerAgent
Analyzes workflow patterns and execution sequences
"""

import sqlite3
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class WorkflowPattern:
    """Represents a workflow pattern"""
    pattern_id: str
    tool_sequence: List[str]
    success_rate: float
    avg_execution_time: float
    frequency: int
    last_used: datetime
    optimization_score: float

class PatternAnalyzer:
    """
    Analyzes workflow patterns and execution sequences
    """
    
    def __init__(self, db_path: str = "overseer_memory.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the pattern analysis database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create patterns table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS workflow_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    tool_sequence TEXT,
                    success_rate REAL,
                    avg_execution_time REAL,
                    frequency INTEGER,
                    last_used TIMESTAMP,
                    optimization_score REAL
                )
            """)
            
            # Check if table was created
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='workflow_patterns'")
            result = cursor.fetchone()
            
            conn.commit()
            conn.close()
            
            if result:
                logger.info("[PATTERN_ANALYZER] Database initialized successfully")
            else:
                logger.error("[PATTERN_ANALYZER] Failed to create workflow_patterns table")
            
        except Exception as e:
            logger.error(f"[PATTERN_ANALYZER] Database init error: {e}")
            # Create database anyway if it doesn't exist
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS workflow_patterns (
                        pattern_id TEXT PRIMARY KEY,
                        tool_sequence TEXT,
                        success_rate REAL,
                        avg_execution_time REAL,
                        frequency INTEGER,
                        last_used TIMESTAMP,
                        optimization_score REAL
                    )
                """)
                conn.commit()
                conn.close()
                logger.info("[PATTERN_ANALYZER] Database created on retry")
            except:
                logger.error("[PATTERN_ANALYZER] Failed to create database on retry")
    
    def generate_pattern_id(self, tool_sequence: List[str]) -> str:
        """Generate unique pattern ID from tool sequence"""
        sequence_str = json.dumps(sorted(tool_sequence))
        return hashlib.md5(sequence_str.encode()).hexdigest()
    
    def analyze_execution_sequence(self, tool_sequence: List[str], 
                                 execution_time: float, 
                                 success: bool) -> Dict:
        """
        Analyze an execution sequence and update patterns
        """
        try:
            pattern_id = self.generate_pattern_id(tool_sequence)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if pattern exists
            cursor.execute("""
                SELECT * FROM workflow_patterns WHERE pattern_id = ?
            """, (pattern_id,))
            
            existing = cursor.fetchone()
            
            if existing:
                # Update existing pattern
                current_success_rate = existing[2]
                current_avg_time = existing[3]
                current_frequency = existing[4]
                
                # Calculate new values
                new_frequency = current_frequency + 1
                new_success_rate = ((current_success_rate * current_frequency) + 
                                  (1 if success else 0)) / new_frequency
                new_avg_time = ((current_avg_time * current_frequency) + 
                              execution_time) / new_frequency
                
                # Calculate optimization score
                optimization_score = self.calculate_optimization_score(
                    new_success_rate, new_avg_time, new_frequency
                )
                
                cursor.execute("""
                    UPDATE workflow_patterns 
                    SET success_rate = ?, avg_execution_time = ?, 
                        frequency = ?, last_used = ?, optimization_score = ?
                    WHERE pattern_id = ?
                """, (new_success_rate, new_avg_time, new_frequency, 
                     datetime.now(), optimization_score, pattern_id))
                
            else:
                # Create new pattern
                optimization_score = self.calculate_optimization_score(
                    1.0 if success else 0.0, execution_time, 1
                )
                
                cursor.execute("""
                    INSERT INTO workflow_patterns 
                    (pattern_id, tool_sequence, success_rate, avg_execution_time, 
                     frequency, last_used, optimization_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (pattern_id, json.dumps(tool_sequence), 
                     1.0 if success else 0.0, execution_time, 1, 
                     datetime.now(), optimization_score))
            
            conn.commit()
            conn.close()
            
            return {
                "pattern_id": pattern_id,
                "updated": True,
                "optimization_score": optimization_score
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_ANALYZER] Analysis error: {e}")
            return {"error": str(e)}
    
    def calculate_optimization_score(self, success_rate: float, 
                                   avg_time: float, 
                                   frequency: int) -> float:
        """
        Calculate optimization score for a pattern
        Higher score = better pattern
        """
        # Base score from success rate (0-1)
        base_score = success_rate
        
        # Time efficiency bonus (faster = better)
        time_bonus = max(0, (10 - avg_time) / 10)
        
        # Frequency bonus (more used = more reliable)
        frequency_bonus = min(frequency / 100, 0.5)
        
        # Combined score
        total_score = (base_score * 0.6) + (time_bonus * 0.3) + (frequency_bonus * 0.1)
        
        return min(max(total_score, 0.0), 1.0)
    
    def get_top_patterns(self, limit: int = 10) -> List[WorkflowPattern]:
        """Get top performing patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM workflow_patterns 
                ORDER BY optimization_score DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            patterns = []
            for row in rows:
                pattern = WorkflowPattern(
                    pattern_id=row[0],
                    tool_sequence=json.loads(row[1]),
                    success_rate=row[2],
                    avg_execution_time=row[3],
                    frequency=row[4],
                    last_used=datetime.fromisoformat(row[5]),
                    optimization_score=row[6]
                )
                patterns.append(pattern)
            
            return patterns
            
        except Exception as e:
            logger.error(f"[PATTERN_ANALYZER] Get patterns error: {e}")
            return []
    
    def find_similar_patterns(self, tool_sequence: List[str], 
                            similarity_threshold: float = 0.7) -> List[WorkflowPattern]:
        """Find patterns similar to given sequence"""
        try:
            all_patterns = self.get_top_patterns(100)
            similar_patterns = []
            
            for pattern in all_patterns:
                similarity = self.calculate_sequence_similarity(
                    tool_sequence, pattern.tool_sequence
                )
                
                if similarity >= similarity_threshold:
                    similar_patterns.append(pattern)
            
            # Sort by similarity score
            similar_patterns.sort(key=lambda p: p.optimization_score, reverse=True)
            
            return similar_patterns
            
        except Exception as e:
            logger.error(f"[PATTERN_ANALYZER] Similar patterns error: {e}")
            return []
    
    def calculate_sequence_similarity(self, seq1: List[str], 
                                    seq2: List[str]) -> float:
        """Calculate similarity between two tool sequences"""
        if not seq1 or not seq2:
            return 0.0
        
        # Convert to sets for comparison
        set1 = set(seq1)
        set2 = set(seq2)
        
        # Jaccard similarity
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_pattern_insights(self) -> Dict:
        """Get insights about patterns"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total patterns
            cursor.execute("SELECT COUNT(*) FROM workflow_patterns")
            total_patterns = cursor.fetchone()[0]
            
            # Average success rate
            cursor.execute("SELECT AVG(success_rate) FROM workflow_patterns")
            avg_success_rate = cursor.fetchone()[0] or 0.0
            
            # Most used tools
            cursor.execute("SELECT tool_sequence FROM workflow_patterns")
            all_sequences = cursor.fetchall()
            
            tool_usage = {}
            for seq_row in all_sequences:
                tools = json.loads(seq_row[0])
                for tool in tools:
                    tool_usage[tool] = tool_usage.get(tool, 0) + 1
            
            # Top tools
            top_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]
            
            conn.close()
            
            return {
                "total_patterns": total_patterns,
                "avg_success_rate": avg_success_rate,
                "top_tools": top_tools,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_ANALYZER] Insights error: {e}")
            return {"error": str(e)}
    
    def predict_execution_outcome(self, tool_sequence: List[str]) -> Dict:
        """Predict execution outcome for a sequence"""
        try:
            similar_patterns = self.find_similar_patterns(tool_sequence)
            
            if not similar_patterns:
                return {
                    "prediction": "unknown",
                    "confidence": 0.0,
                    "reason": "no_similar_patterns"
                }
            
            # Calculate weighted prediction
            total_weight = sum(p.frequency for p in similar_patterns)
            weighted_success = sum(p.success_rate * p.frequency for p in similar_patterns)
            weighted_time = sum(p.avg_execution_time * p.frequency for p in similar_patterns)
            
            predicted_success_rate = weighted_success / total_weight
            predicted_time = weighted_time / total_weight
            
            # Confidence based on number of similar patterns and their frequency
            confidence = min(len(similar_patterns) / 10, 1.0)
            
            return {
                "predicted_success_rate": predicted_success_rate,
                "predicted_execution_time": predicted_time,
                "confidence": confidence,
                "similar_patterns_count": len(similar_patterns),
                "recommendation": "execute" if predicted_success_rate > 0.7 else "review"
            }
            
        except Exception as e:
            logger.error(f"[PATTERN_ANALYZER] Prediction error: {e}")
            return {"error": str(e)}

# Test function
def test_pattern_analyzer():
    """Test the pattern analyzer"""
    print("[TEST] Testing Pattern Analyzer...")
    
    analyzer = PatternAnalyzer()
    
    # Test pattern analysis
    result = analyzer.analyze_execution_sequence(
        ["file_search", "read_file", "semantic_search"], 
        2.5, True
    )
    print(f"[TEST] Pattern analysis result: {result}")
    
    # Test top patterns
    top_patterns = analyzer.get_top_patterns(5)
    print(f"[TEST] Top patterns: {len(top_patterns)}")
    
    # Test insights
    insights = analyzer.get_pattern_insights()
    print(f"[TEST] Pattern insights: {insights}")
    
    print("[TEST] Pattern Analyzer test completed")

if __name__ == "__main__":
    test_pattern_analyzer()
