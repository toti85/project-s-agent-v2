#!/usr/bin/env python3
"""
Integrated Execution Manager - Project-S V2
Combines tool execution, evaluation, and self-improvement in a unified pipeline.
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.evaluator import ExecutionEvaluator
from core.self_dev_agent import SelfDevelopmentAgent
from tools.registry.tool_registry_golden import ToolRegistry
from integrations.ai_models.multi_model_ai_client import MultiModelAIClient
from tools.registry.smart_orchestrator import SmartToolOrchestrator

logger = logging.getLogger(__name__)

class IntegratedExecutionManager:
    """
    Unified execution manager that handles:
    1. Tool execution
    2. Result evaluation
    3. Failure recovery
    4. Self-improvement
    """
    
    def __init__(
        self,
        ai_client: MultiModelAIClient,
        tool_registry: ToolRegistry,
        config_dir: Optional[Path] = None
    ):
        """Initialize the integrated execution manager."""
        self.ai_client = ai_client
        self.tool_registry = tool_registry
        self.config_dir = config_dir or Path("config")
        
        # Initialize components
        self.evaluator = ExecutionEvaluator(ai_client=ai_client)
        self.self_dev_agent = SelfDevelopmentAgent(ai_client, tool_registry, config_dir)
        self.orchestrator = SmartToolOrchestrator(tool_registry, ai_client)
        
        # Execution tracking
        self.execution_history = []
        self.max_retry_attempts = 3
        self.max_improvement_attempts = 2
        
    async def execute_task_with_intelligence(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None,
        expected_outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a task with full intelligence pipeline:
        AI reasoning → tool execution → evaluation → improvement if needed
        """
        
        execution_id = f"exec_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        logger.info(f"🚀 Starting intelligent task execution: {task_description}")
        logger.info(f"📋 Execution ID: {execution_id}")
        
        execution_context = {
            "execution_id": execution_id,
            "task_description": task_description,
            "context": context or {},
            "expected_outcome": expected_outcome,
            "start_time": datetime.now().isoformat(),
            "attempts": []
        }
        
        try:
            # Execute with retry and improvement logic
            final_result = await self._execute_with_intelligence_loop(
                execution_context,
                max_attempts=self.max_retry_attempts
            )
            
            execution_context["final_result"] = final_result
            execution_context["end_time"] = datetime.now().isoformat()
            execution_context["success"] = final_result.get("success", False)
            
            # Track execution history
            self.execution_history.append(execution_context)
            
            logger.info(f"✅ Task execution completed: {final_result.get('success', False)}")
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Task execution failed critically: {e}")
            execution_context["critical_error"] = str(e)
            execution_context["success"] = False
            execution_context["end_time"] = datetime.now().isoformat()
            
            self.execution_history.append(execution_context)
            
            return {
                "success": False,
                "error": str(e),
                "execution_id": execution_id,
                "execution_context": execution_context
            }
    
    async def _execute_with_intelligence_loop(
        self,
        execution_context: Dict[str, Any],
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """Main intelligence loop with retry and improvement."""
        
        task_description = execution_context["task_description"]
        
        for attempt_num in range(1, max_attempts + 1):
            logger.info(f"🔄 Attempt {attempt_num}/{max_attempts}")
            
            attempt_context = {
                "attempt_number": attempt_num,
                "start_time": datetime.now().isoformat()
            }
            
            try:
                # Step 1: AI reasoning and tool selection
                logger.info("🧠 AI reasoning and tool selection...")
                tool_selection = await self._ai_reasoning_phase(task_description, execution_context)
                attempt_context["tool_selection"] = tool_selection
                
                if not tool_selection.get("success"):
                    raise ValueError(f"AI reasoning failed: {tool_selection.get('error')}")
                
                # Step 2: Tool execution
                logger.info(f"🛠️ Executing tool: {tool_selection['tool_name']}")
                execution_result = await self._execute_tool_phase(tool_selection)
                attempt_context["execution_result"] = execution_result
                
                # Step 3: Result evaluation
                logger.info("🔍 Evaluating execution result...")
                evaluation = await self._evaluation_phase(
                    task_description,
                    tool_selection,
                    execution_result,
                    execution_context.get("expected_outcome")
                )
                attempt_context["evaluation"] = evaluation
                
                # Step 4: Handle evaluation outcome
                if evaluation["status"] == "success":
                    logger.info("✅ Execution successful!")
                    attempt_context["outcome"] = "success"
                    execution_context["attempts"].append(attempt_context)
                    
                    return {
                        "success": True,
                        "result": execution_result,
                        "evaluation": evaluation,
                        "attempts": attempt_num,
                        "execution_id": execution_context["execution_id"]
                    }
                
                elif evaluation["status"] == "failure":
                    logger.info(f"⚠️ Execution failed: {evaluation['reason']}")
                    attempt_context["outcome"] = "failure"
                    
                    # Handle different failure strategies
                    if evaluation["suggestion"] == "retry" and attempt_num < max_attempts:
                        logger.info("🔄 Retrying with same approach...")
                        attempt_context["strategy"] = "retry"
                        
                    elif evaluation["suggestion"] == "replan" and attempt_num < max_attempts:
                        logger.info("🎯 Replanning approach...")
                        attempt_context["strategy"] = "replan"
                        # Add failure context for next attempt
                        execution_context["previous_failures"] = execution_context.get("previous_failures", [])
                        execution_context["previous_failures"].append({
                            "attempt": attempt_num,
                            "tool": tool_selection["tool_name"],
                            "error": evaluation["reason"]
                        })
                        
                    elif evaluation["suggestion"] == "implement_tool":
                        logger.info("🛠️ Attempting self-improvement...")
                        improvement_result = await self._self_improvement_phase(
                            task_description,
                            tool_selection,
                            evaluation,
                            execution_result
                        )
                        attempt_context["improvement_attempt"] = improvement_result
                        
                        if improvement_result.get("success"):
                            logger.info("✨ Self-improvement successful, retrying...")
                            attempt_context["strategy"] = "improved_retry"
                        else:
                            logger.warning("⚠️ Self-improvement failed")
                            attempt_context["strategy"] = "improvement_failed"
                    
                else:  # partial success
                    logger.info("⚡ Partial success - continuing with result")
                    attempt_context["outcome"] = "partial"
                    execution_context["attempts"].append(attempt_context)
                    
                    return {
                        "success": True,
                        "result": execution_result,
                        "evaluation": evaluation,
                        "attempts": attempt_num,
                        "warning": "Partial success",
                        "execution_id": execution_context["execution_id"]
                    }
                
            except Exception as e:
                logger.error(f"❌ Attempt {attempt_num} failed with exception: {e}")
                attempt_context["exception"] = str(e)
                attempt_context["outcome"] = "exception"
            
            attempt_context["end_time"] = datetime.now().isoformat()
            execution_context["attempts"].append(attempt_context)
        
        # All attempts exhausted
        logger.error(f"❌ All {max_attempts} attempts exhausted")
        return {
            "success": False,
            "error": "All execution attempts failed",
            "attempts": max_attempts,
            "execution_context": execution_context,
            "execution_id": execution_context["execution_id"]
        }
    
    async def _ai_reasoning_phase(
        self,
        task_description: str,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """AI reasoning phase to understand task and select appropriate tool."""
        
        try:
            # Get available tools
            available_tools = self.tool_registry.list_tools()
            
            # Include previous failure context if any
            failure_context = ""
            if "previous_failures" in execution_context:
                failure_context = f"\nPREVIOUS FAILED ATTEMPTS:\n"
                for failure in execution_context["previous_failures"]:
                    failure_context += f"- Attempt {failure['attempt']}: Tool '{failure['tool']}' failed: {failure['error']}\n"
            
            reasoning_prompt = f"""
Analyze this task and select the most appropriate tool for execution:

TASK: {task_description}

AVAILABLE TOOLS:
{self._format_available_tools(available_tools)}
{failure_context}

CONTEXT: {execution_context.get('context', {})}

Please analyze the task and respond in JSON format:
{{
    "reasoning": "step-by-step analysis of what the task requires",
    "tool_name": "most_appropriate_tool_name",
    "tool_args": {{
        "arg1": "value1",
        "arg2": "value2"
    }},
    "confidence": 0.0-1.0,
    "alternative_tools": ["tool2", "tool3"],
    "success": true
}}
"""
            
            response = await self.ai_client.generate_response(
                prompt=reasoning_prompt,
                model="qwen3-235b",
                temperature=0.3
            )
            
            import json
            reasoning_result = json.loads(response)
            
            # Validate tool exists
            tool_name = reasoning_result.get("tool_name")
            if tool_name not in available_tools:
                raise ValueError(f"Selected tool '{tool_name}' is not available")
            
            return reasoning_result
            
        except Exception as e:
            logger.error(f"❌ AI reasoning phase failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "reasoning": "AI reasoning failed"
            }
    
    async def _execute_tool_phase(self, tool_selection: Dict[str, Any]) -> Any:
        """Execute the selected tool with provided arguments."""
        
        tool_name = tool_selection["tool_name"]
        tool_args = tool_selection.get("tool_args", {})
        
        try:
            tool = self.tool_registry.get_tool(tool_name)
            if not tool:
                raise ValueError(f"Tool '{tool_name}' not found in registry")
            
            result = await tool.execute(**tool_args)
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_name": tool_name,
                "tool_args": tool_args
            }
    
    async def _evaluation_phase(
        self,
        task_description: str,
        tool_selection: Dict[str, Any],
        execution_result: Any,
        expected_outcome: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate the execution result."""
        
        return await self.evaluator.evaluate_execution_result(
            task_description=task_description,
            tool_name=tool_selection["tool_name"],
            tool_args=tool_selection.get("tool_args", {}),
            execution_result=execution_result,
            expected_outcome=expected_outcome
        )
    
    async def _self_improvement_phase(
        self,
        task_description: str,
        tool_selection: Dict[str, Any],
        evaluation: Dict[str, Any],
        execution_result: Any
    ) -> Dict[str, Any]:
        """Attempt self-improvement when standard tools fail."""
        
        failure_context = {
            "task_description": task_description,
            "tool_name": tool_selection["tool_name"],
            "tool_args": tool_selection.get("tool_args", {}),
            "execution_result": execution_result,
            "evaluation": evaluation
        }
        
        return await self.self_dev_agent.autonomous_improvement_cycle(failure_context)
    
    def _format_available_tools(self, available_tools: Dict[str, Any]) -> str:
        """Format available tools for AI prompt."""
        
        formatted = ""
        for tool_name, tool_info in available_tools.items():
            formatted += f"- {tool_name}: {tool_info.get('description', 'No description')}\n"
            if tool_info.get('required_args'):
                formatted += f"  Required args: {tool_info['required_args']}\n"
        
        return formatted
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics and performance metrics."""
        
        if not self.execution_history:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_attempts": 0.0
            }
        
        total = len(self.execution_history)
        successful = sum(1 for exec_ctx in self.execution_history if exec_ctx.get("success"))
        total_attempts = sum(len(exec_ctx.get("attempts", [])) for exec_ctx in self.execution_history)
        
        # Get evaluator stats
        evaluator_stats = await self.evaluator.get_evaluation_stats()
        
        return {
            "total_executions": total,
            "successful_executions": successful,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_attempts": total_attempts / total if total > 0 else 0.0,
            "evaluation_stats": evaluator_stats,
            "recent_executions": self.execution_history[-5:]  # Last 5
        }


# Convenience function for quick intelligent execution
async def execute_task_intelligently(
    task_description: str,
    ai_client: MultiModelAIClient,
    tool_registry: ToolRegistry,
    context: Optional[Dict[str, Any]] = None,
    expected_outcome: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for intelligent task execution."""
    
    manager = IntegratedExecutionManager(ai_client, tool_registry)
    return await manager.execute_task_with_intelligence(
        task_description, context, expected_outcome
    )
