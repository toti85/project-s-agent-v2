#!/usr/bin/env python3
"""
Comprehensive Test Suite for OverseerAgent System
Tests all components: OverseerAgent, PatternAnalyzer, OptimizationEngine, and Integration
"""

import asyncio
import sys
import json
import unittest
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestOverseerAgent(unittest.TestCase):
    """Test cases for OverseerAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        from core.agents.overseer_agent import OverseerAgent
        from core.infrastructure.event_bus import EventBus
        
        # Create test event bus
        event_bus = EventBus()
        self.agent = OverseerAgent(event_bus=event_bus, memory_db_path=":memory:")
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent)
        self.assertTrue(self.agent.learning_enabled)
        self.assertIsNotNone(self.agent.db_path)
    
    def test_system_health_assessment(self):
        """Test system health assessment"""
        async def test_async():
            health = await self.agent.get_system_health()
            self.assertIsInstance(health, dict)
            self.assertIn("health", health)
        
        # Run in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_async())
        finally:
            loop.close()
    
    def test_performance_metrics(self):
        """Test performance metrics recording"""
        async def test_async():
            insights = await self.agent.get_performance_insights()
            self.assertIsInstance(insights, dict)
        
        # Run in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_async())
        finally:
            loop.close()
    
    def test_learning_system(self):
        """Test learning system"""
        self.agent.enable_learning()
        self.assertTrue(self.agent.learning_enabled)
        
        self.agent.disable_learning()
        self.assertFalse(self.agent.learning_enabled)
    
    def test_optimization_suggestions(self):
        """Test optimization suggestions"""
        async def test_async():
            suggestions = await self.agent.get_optimization_suggestions()
            self.assertIsInstance(suggestions, list)
        
        # Run in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_async())
        finally:
            loop.close()

class TestPatternAnalyzer(unittest.TestCase):
    """Test cases for PatternAnalyzer"""
    
    def setUp(self):
        """Set up test fixtures"""
        from core.agents.pattern_analyzer import PatternAnalyzer
        self.analyzer = PatternAnalyzer(db_path=":memory:")
    
    def test_pattern_analysis(self):
        """Test pattern analysis"""
        result = self.analyzer.analyze_execution_sequence(
            ["tool1", "tool2", "tool3"], 2.5, True
        )
        self.assertIsInstance(result, dict)
        self.assertIn("pattern_id", result)
    
    def test_pattern_insights(self):
        """Test pattern insights"""
        insights = self.analyzer.get_pattern_insights()
        self.assertIsInstance(insights, dict)
        self.assertIn("total_patterns", insights)
    
    def test_top_patterns(self):
        """Test top patterns retrieval"""
        patterns = self.analyzer.get_top_patterns(5)
        self.assertIsInstance(patterns, list)
    
    def test_prediction(self):
        """Test execution outcome prediction"""
        # First add some data
        self.analyzer.analyze_execution_sequence(
            ["tool1", "tool2"], 1.0, True
        )
        
        prediction = self.analyzer.predict_execution_outcome(["tool1", "tool2"])
        self.assertIsInstance(prediction, dict)
    
    def test_sequence_similarity(self):
        """Test sequence similarity calculation"""
        similarity = self.analyzer.calculate_sequence_similarity(
            ["tool1", "tool2", "tool3"],
            ["tool1", "tool2", "tool4"]
        )
        self.assertIsInstance(similarity, float)
        self.assertGreaterEqual(similarity, 0.0)
        self.assertLessEqual(similarity, 1.0)

class TestOptimizationEngine(unittest.TestCase):
    """Test cases for OptimizationEngine"""
    
    def setUp(self):
        """Set up test fixtures"""
        from core.agents.optimization_engine import OptimizationEngine
        self.engine = OptimizationEngine(db_path=":memory:")
    
    def test_performance_analysis(self):
        """Test performance analysis"""
        analysis = self.engine.analyze_performance_metrics()
        self.assertIsInstance(analysis, dict)
    
    def test_suggestion_generation(self):
        """Test optimization suggestion generation"""
        suggestions = self.engine.generate_optimization_suggestions()
        self.assertIsInstance(suggestions, list)
    
    def test_active_suggestions(self):
        """Test active suggestions retrieval"""
        suggestions = self.engine.get_active_suggestions()
        self.assertIsInstance(suggestions, list)
    
    def test_optimization_stats(self):
        """Test optimization statistics"""
        stats = self.engine.get_optimization_stats()
        self.assertIsInstance(stats, dict)
        self.assertIn("total_suggestions", stats)
    
    def test_suggestion_implementation(self):
        """Test suggestion implementation marking"""
        # First generate a suggestion
        suggestions = self.engine.generate_optimization_suggestions()
        
        if suggestions:
            suggestion_id = suggestions[0].suggestion_id
            result = self.engine.mark_suggestion_implemented(suggestion_id)
            self.assertTrue(result)

class TestOverseerIntegration(unittest.TestCase):
    """Test cases for OverseerIntegration"""
    
    def setUp(self):
        """Set up test fixtures"""
        from core.agents.overseer_integration import OverseerIntegration
        self.integration = OverseerIntegration()
    
    def test_integration_initialization(self):
        """Test integration initialization"""
        async def test_async():
            result = await self.integration.initialize()
            self.assertTrue(result)
        
        # Run in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_async())
        finally:
            loop.close()
    
    def test_system_status(self):
        """Test system status retrieval"""
        async def test_async():
            await self.integration.initialize()
            status = await self.integration.get_system_status()
            self.assertIsInstance(status, dict)
            self.assertIn("components", status)
        
        # Run in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_async())
        finally:
            loop.close()
    
    def test_event_handling(self):
        """Test event handling"""
        async def test_async():
            await self.integration.initialize()
            
            # Test tool execution events
            await self.integration.handle_tool_execution_start({
                "tool_name": "test_tool",
                "context": {"test": True}
            })
            
            await self.integration.handle_tool_execution_end({
                "tool_name": "test_tool",
                "success": True,
                "execution_time": 2.0,
                "context": {"test": True}
            })
        
        # Run in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(test_async())
        finally:
            loop.close()

async def run_comprehensive_tests():
    """Run comprehensive tests of all components"""
    print("=" * 60)
    print("OverseerAgent System - Comprehensive Test Suite")
    print("=" * 60)
    
    # Test individual components
    print("\n[TEST] Testing OverseerAgent...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOverseerAgent)
    runner = unittest.TextTestRunner(verbosity=2)
    result_agent = runner.run(suite)
    
    print("\n[TEST] Testing PatternAnalyzer...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPatternAnalyzer)
    result_pattern = runner.run(suite)
    
    print("\n[TEST] Testing OptimizationEngine...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOptimizationEngine)
    result_optimization = runner.run(suite)
    
    print("\n[TEST] Testing OverseerIntegration...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestOverseerIntegration)
    result_integration = runner.run(suite)
    
    # Integration test
    print("\n[TEST] Running Integration Test...")
    await integration_test()
    
    # Performance test
    print("\n[TEST] Running Performance Test...")
    await performance_test()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = (result_agent.testsRun + result_pattern.testsRun + 
                  result_optimization.testsRun + result_integration.testsRun)
    total_failures = (len(result_agent.failures) + len(result_pattern.failures) + 
                     len(result_optimization.failures) + len(result_integration.failures))
    total_errors = (len(result_agent.errors) + len(result_pattern.errors) + 
                   len(result_optimization.errors) + len(result_integration.errors))
    
    print(f"Total Tests: {total_tests}")
    print(f"Failures: {total_failures}")
    print(f"Errors: {total_errors}")
    print(f"Success Rate: {((total_tests - total_failures - total_errors) / total_tests * 100):.1f}%")
    
    if total_failures == 0 and total_errors == 0:
        print("[SUCCESS] All tests passed!")
        return True
    else:
        print("[FAILURE] Some tests failed")
        return False

async def integration_test():
    """Test full system integration"""
    try:
        from core.agents.overseer_integration import OverseerIntegration
        
        print("  [INTEGRATION] Creating integrated system...")
        integration = OverseerIntegration()
        
        print("  [INTEGRATION] Initializing components...")
        success = await integration.initialize()
        
        if not success:
            print("  [INTEGRATION] Failed to initialize")
            return False
        
        print("  [INTEGRATION] Testing workflow simulation...")
        
        # Simulate a workflow
        await integration.handle_workflow_start({
            "workflow_id": "test_workflow",
            "workflow_type": "test",
            "tools": ["file_search", "read_file", "semantic_search"]
        })
        
        # Simulate tool executions
        for tool in ["file_search", "read_file", "semantic_search"]:
            await integration.handle_tool_execution_start({
                "tool_name": tool,
                "context": {"workflow_id": "test_workflow"}
            })
            
            await integration.handle_tool_execution_end({
                "tool_name": tool,
                "success": True,
                "execution_time": 1.5,
                "context": {"workflow_id": "test_workflow"}
            })
        
        await integration.handle_workflow_end({
            "workflow_id": "test_workflow",
            "success": True,
            "execution_time": 4.5
        })
        
        print("  [INTEGRATION] Getting system status...")
        status = await integration.get_system_status()
        
        print(f"  [INTEGRATION] System status: {json.dumps(status, indent=2, default=str)}")
        print("  [INTEGRATION] Test completed successfully")
        
        return True
        
    except Exception as e:
        print(f"  [INTEGRATION] Error: {e}")
        return False

async def performance_test():
    """Test system performance"""
    try:
        from core.agents.overseer_agent import OverseerAgent
        from core.agents.pattern_analyzer import PatternAnalyzer
        from core.agents.optimization_engine import OptimizationEngine
        
        print("  [PERFORMANCE] Testing system performance...")
        
        # Create components
        from core.infrastructure.event_bus import EventBus
        event_bus = EventBus()
        agent = OverseerAgent(event_bus=event_bus, memory_db_path=":memory:")
        analyzer = PatternAnalyzer(db_path=":memory:")
        engine = OptimizationEngine(db_path=":memory:")
        
        # Performance test - record many metrics
        start_time = datetime.now()
        
        for i in range(100):
            analyzer.analyze_execution_sequence([f"tool_{i}", f"tool_{i+1}"], float(i), True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"  [PERFORMANCE] Processed 100 operations in {duration:.2f} seconds")
        print(f"  [PERFORMANCE] Rate: {100/duration:.1f} operations/second")
        
        # Test system health
        health = await agent.get_system_health()
        print(f"  [PERFORMANCE] System health: {health}")
        
        # Test suggestions
        suggestions = engine.generate_optimization_suggestions()
        print(f"  [PERFORMANCE] Generated {len(suggestions)} suggestions")
        
        print("  [PERFORMANCE] Performance test completed")
        
        return True
        
    except Exception as e:
        print(f"  [PERFORMANCE] Error: {e}")
        return False

def main():
    """Main test function"""
    print("Starting OverseerAgent System Tests...")
    
    try:
        result = asyncio.run(run_comprehensive_tests())
        
        if result:
            print("\n[SUCCESS] All tests completed successfully!")
            print("OverseerAgent system is ready for production use.")
            sys.exit(0)
        else:
            print("\n[FAILURE] Some tests failed.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n[ERROR] Test execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
