#!/usr/bin/env python3
"""
Optimization Engine for OverseerAgent
Generates optimization suggestions and performance improvements
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import uuid

logger = logging.getLogger(__name__)

@dataclass
class OptimizationSuggestion:
    """Represents an optimization suggestion"""
    suggestion_id: str
    type: str
    description: str
    expected_improvement: float
    confidence: float
    implementation_cost: float
    created_at: datetime
    status: str = "pending"

class OptimizationEngine:
    """
    Generates optimization suggestions and performance improvements
    """
    
    def __init__(self, db_path: str = "overseer_memory.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the optimization database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create optimization suggestions table
            cursor.execute("""
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
            """)
            
            # Create performance metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    metric_type TEXT,
                    value REAL,
                    context TEXT
                )
            """)
            
            # Check if tables were created
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            table_names = [table[0] for table in tables]
            
            conn.commit()
            conn.close()
            
            logger.info(f"[OPTIMIZATION_ENGINE] Database initialized with tables: {table_names}")
            
        except Exception as e:
            logger.error(f"[OPTIMIZATION_ENGINE] Database init error: {e}")
            # Create database anyway if it doesn't exist
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute("""
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
                """)
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP,
                        metric_type TEXT,
                        value REAL,
                        context TEXT
                    )
                """)
                conn.commit()
                conn.close()
                logger.info("[OPTIMIZATION_ENGINE] Database created on retry")
            except:
                logger.error("[OPTIMIZATION_ENGINE] Failed to create database on retry")
    
    def analyze_performance_metrics(self) -> Dict:
        """Analyze performance metrics and generate insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent performance metrics
            cursor.execute("""
                SELECT * FROM performance_metrics 
                WHERE timestamp > datetime('now', '-1 hour')
                ORDER BY timestamp DESC
            """)
            
            metrics = cursor.fetchall()
            conn.close()
            
            if not metrics:
                return {"status": "insufficient_data"}
            
            # Analyze metrics
            execution_times = [m[3] for m in metrics if m[2] == 'execution_time']
            success_rates = [m[3] for m in metrics if m[2] == 'success_rate']
            
            analysis = {
                "total_metrics": len(metrics),
                "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
                "avg_success_rate": sum(success_rates) / len(success_rates) if success_rates else 0,
                "slow_operations": len([t for t in execution_times if t > 5.0]),
                "failed_operations": len([s for s in success_rates if s < 0.8])
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"[OPTIMIZATION_ENGINE] Performance analysis error: {e}")
            return {"error": str(e)}
    
    def generate_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on current performance"""
        try:
            performance_analysis = self.analyze_performance_metrics()
            suggestions = []
            
            if "error" in performance_analysis:
                return suggestions
            
            # Generate suggestions based on performance analysis
            suggestions.extend(self._generate_performance_suggestions(performance_analysis))
            suggestions.extend(self._generate_workflow_suggestions())
            suggestions.extend(self._generate_resource_suggestions())
            
            # Store suggestions in database
            self._store_suggestions(suggestions)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"[OPTIMIZATION_ENGINE] Suggestion generation error: {e}")
            return []
    
    def _generate_performance_suggestions(self, analysis: Dict) -> List[OptimizationSuggestion]:
        """Generate performance-related suggestions"""
        suggestions = []
        
        # Slow operations suggestion
        if analysis.get("slow_operations", 0) > 3:
            suggestion = OptimizationSuggestion(
                suggestion_id=str(uuid.uuid4()),
                type="performance",
                description="Multiple slow operations detected. Consider implementing caching or optimizing heavy operations.",
                expected_improvement=0.3,
                confidence=0.8,
                implementation_cost=0.6,
                created_at=datetime.now()
            )
            suggestions.append(suggestion)
        
        # Failed operations suggestion
        if analysis.get("failed_operations", 0) > 2:
            suggestion = OptimizationSuggestion(
                suggestion_id=str(uuid.uuid4()),
                type="reliability",
                description="Multiple failed operations detected. Consider implementing retry logic or error handling improvements.",
                expected_improvement=0.25,
                confidence=0.7,
                implementation_cost=0.4,
                created_at=datetime.now()
            )
            suggestions.append(suggestion)
        
        # Execution time optimization
        if analysis.get("avg_execution_time", 0) > 3.0:
            suggestion = OptimizationSuggestion(
                suggestion_id=str(uuid.uuid4()),
                type="efficiency",
                description="Average execution time is high. Consider parallelizing operations or optimizing algorithms.",
                expected_improvement=0.4,
                confidence=0.6,
                implementation_cost=0.8,
                created_at=datetime.now()
            )
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_workflow_suggestions(self) -> List[OptimizationSuggestion]:
        """Generate workflow-related suggestions"""
        suggestions = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get workflow patterns
            cursor.execute("""
                SELECT * FROM workflow_patterns 
                WHERE success_rate < 0.7 AND frequency > 2
            """)
            
            low_success_patterns = cursor.fetchall()
            
            if low_success_patterns:
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    type="workflow",
                    description=f"Found {len(low_success_patterns)} workflow patterns with low success rates. Consider reviewing and optimizing these workflows.",
                    expected_improvement=0.35,
                    confidence=0.75,
                    implementation_cost=0.5,
                    created_at=datetime.now()
                )
                suggestions.append(suggestion)
            
            # Check for unused efficient patterns
            cursor.execute("""
                SELECT * FROM workflow_patterns 
                WHERE success_rate > 0.9 AND last_used < datetime('now', '-7 days')
            """)
            
            unused_patterns = cursor.fetchall()
            
            if unused_patterns:
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    type="workflow",
                    description=f"Found {len(unused_patterns)} highly successful workflow patterns that haven't been used recently. Consider promoting their usage.",
                    expected_improvement=0.2,
                    confidence=0.6,
                    implementation_cost=0.3,
                    created_at=datetime.now()
                )
                suggestions.append(suggestion)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"[OPTIMIZATION_ENGINE] Workflow suggestions error: {e}")
        
        return suggestions
    
    def _generate_resource_suggestions(self) -> List[OptimizationSuggestion]:
        """Generate resource-related suggestions"""
        suggestions = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check database size
            cursor.execute("SELECT COUNT(*) FROM performance_metrics")
            metrics_count = cursor.fetchone()[0]
            
            if metrics_count > 10000:
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    type="maintenance",
                    description="Performance metrics database has grown large. Consider implementing data retention policies or archiving old data.",
                    expected_improvement=0.15,
                    confidence=0.9,
                    implementation_cost=0.2,
                    created_at=datetime.now()
                )
                suggestions.append(suggestion)
            
            # Check for memory optimization opportunities
            cursor.execute("SELECT COUNT(*) FROM workflow_patterns")
            patterns_count = cursor.fetchone()[0]
            
            if patterns_count > 1000:
                suggestion = OptimizationSuggestion(
                    suggestion_id=str(uuid.uuid4()),
                    type="optimization",
                    description="Large number of workflow patterns detected. Consider implementing pattern consolidation or cleanup of unused patterns.",
                    expected_improvement=0.1,
                    confidence=0.7,
                    implementation_cost=0.4,
                    created_at=datetime.now()
                )
                suggestions.append(suggestion)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"[OPTIMIZATION_ENGINE] Resource suggestions error: {e}")
        
        return suggestions
    
    def _store_suggestions(self, suggestions: List[OptimizationSuggestion]):
        """Store optimization suggestions in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for suggestion in suggestions:
                cursor.execute("""
                    INSERT OR REPLACE INTO optimization_suggestions 
                    (suggestion_id, type, description, expected_improvement, 
                     confidence, implementation_cost, created_at, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    suggestion.suggestion_id,
                    suggestion.type,
                    suggestion.description,
                    suggestion.expected_improvement,
                    suggestion.confidence,
                    suggestion.implementation_cost,
                    suggestion.created_at.isoformat(),
                    suggestion.status
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"[OPTIMIZATION_ENGINE] Store suggestions error: {e}")
    
    def get_active_suggestions(self) -> List[OptimizationSuggestion]:
        """Get active optimization suggestions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM optimization_suggestions 
                WHERE status = 'pending' 
                ORDER BY expected_improvement DESC, confidence DESC
            """)
            
            rows = cursor.fetchall()
            conn.close()
            
            suggestions = []
            for row in rows:
                suggestion = OptimizationSuggestion(
                    suggestion_id=row[0],
                    type=row[1],
                    description=row[2],
                    expected_improvement=row[3],
                    confidence=row[4],
                    implementation_cost=row[5],
                    created_at=datetime.fromisoformat(row[6]),
                    status=row[7]
                )
                suggestions.append(suggestion)
            
            return suggestions
            
        except Exception as e:
            logger.error(f"[OPTIMIZATION_ENGINE] Get suggestions error: {e}")
            return []
    
    def mark_suggestion_implemented(self, suggestion_id: str) -> bool:
        """Mark a suggestion as implemented"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE optimization_suggestions 
                SET status = 'implemented' 
                WHERE suggestion_id = ?
            """, (suggestion_id,))
            
            conn.commit()
            conn.close()
            
            return True
            
        except Exception as e:
            logger.error(f"[OPTIMIZATION_ENGINE] Mark implemented error: {e}")
            return False
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total suggestions
            cursor.execute("SELECT COUNT(*) FROM optimization_suggestions")
            total_suggestions = cursor.fetchone()[0]
            
            # Implemented suggestions
            cursor.execute("SELECT COUNT(*) FROM optimization_suggestions WHERE status = 'implemented'")
            implemented_suggestions = cursor.fetchone()[0]
            
            # Pending suggestions
            cursor.execute("SELECT COUNT(*) FROM optimization_suggestions WHERE status = 'pending'")
            pending_suggestions = cursor.fetchone()[0]
            
            # Average improvement
            cursor.execute("SELECT AVG(expected_improvement) FROM optimization_suggestions WHERE status = 'implemented'")
            avg_improvement = cursor.fetchone()[0] or 0.0
            
            # Top suggestion types
            cursor.execute("""
                SELECT type, COUNT(*) as count 
                FROM optimization_suggestions 
                GROUP BY type 
                ORDER BY count DESC
            """)
            
            suggestion_types = cursor.fetchall()
            
            conn.close()
            
            return {
                "total_suggestions": total_suggestions,
                "implemented_suggestions": implemented_suggestions,
                "pending_suggestions": pending_suggestions,
                "implementation_rate": implemented_suggestions / total_suggestions if total_suggestions > 0 else 0,
                "avg_improvement": avg_improvement,
                "suggestion_types": suggestion_types
            }
            
        except Exception as e:
            logger.error(f"[OPTIMIZATION_ENGINE] Stats error: {e}")
            return {"error": str(e)}

# Test function
def test_optimization_engine():
    """Test the optimization engine"""
    print("[TEST] Testing Optimization Engine...")
    
    engine = OptimizationEngine()
    
    # Test performance analysis
    analysis = engine.analyze_performance_metrics()
    print(f"[TEST] Performance analysis: {analysis}")
    
    # Test suggestion generation
    suggestions = engine.generate_optimization_suggestions()
    print(f"[TEST] Generated {len(suggestions)} suggestions")
    
    # Test active suggestions
    active = engine.get_active_suggestions()
    print(f"[TEST] Active suggestions: {len(active)}")
    
    # Test stats
    stats = engine.get_optimization_stats()
    print(f"[TEST] Optimization stats: {stats}")
    
    print("[TEST] Optimization Engine test completed")

if __name__ == "__main__":
    test_optimization_engine()
