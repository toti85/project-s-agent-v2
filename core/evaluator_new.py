"""
Evaluator Agent - Assesses tool execution results and triggers replanning
This component evaluates whether tool execution achieved the intended goal
and can trigger replanning or self-improvement actions when needed.
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .infrastructure.event_bus import event_bus
from .infrastructure.error_handler import error_handler
from ..integrations.ai_models.multi_model_ai_client import MultiModelAIClient
from ..utils.performance_monitor import monitor_performance, EnhancedLogger

class Evaluator:
    """
    Strategic evaluation agent that assesses execution results and determines
    if goals were achieved or if replanning/self-improvement is needed.
    """
    
    def __init__(self, ai_client: Optional[MultiModelAIClient] = None):
        """Initialize the evaluator with AI client and logger."""
        self.ai_client = ai_client or MultiModelAIClient()
        self.logger = EnhancedLogger("evaluator")
        self.evaluation_history: List[Dict[str, Any]] = []
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Subscribe to execution events
        event_bus.subscribe("tool.execution.completed", self._on_execution_completed)
        event_bus.subscribe("execution.failed", self._on_execution_failed)
        
        self.logger.info("Evaluator initialized and subscribed to execution events")
    
    @monitor_performance
    async def evaluate_execution_result(
        self, 
        original_goal: str,
        tool_used: str,
        tool_params: Dict[str, Any],
        execution_result: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate if the execution result achieved the original goal.
        
        Args:
            original_goal: The original user intent/goal
            tool_used: Name of the tool that was executed
            tool_params: Parameters passed to the tool
            execution_result: Result from tool execution
            context: Additional context information
            
        Returns:
            Dict containing evaluation results and recommendations
        """
        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        try:
            # Prepare evaluation prompt
            evaluation_prompt = self._build_evaluation_prompt(
                original_goal, tool_used, tool_params, execution_result, context
            )
            
            # Get AI evaluation
            ai_evaluation = await self.ai_client.get_response(
                evaluation_prompt,
                model_type="primary"  # Use primary model (Qwen3 235B)
            )
            
            # Parse AI evaluation
            evaluation_result = self._parse_evaluation_response(ai_evaluation)
            
            # Add metadata
            evaluation_result.update({
                "evaluation_id": evaluation_id,
                "timestamp": datetime.now().isoformat(),
                "original_goal": original_goal,
                "tool_used": tool_used,
                "tool_params": tool_params,
                "execution_result": execution_result,
                "context": context or {}
            })
            
            # Store evaluation in history
            self.evaluation_history.append(evaluation_result)
            
            # Log evaluation
            await self._log_evaluation(evaluation_result)
            
            # Trigger actions based on evaluation
            await self._handle_evaluation_result(evaluation_result)
            
            return evaluation_result
            
        except Exception as e:
            error_context = {
                "component": "evaluator",
                "operation": "evaluate_execution_result",
                "evaluation_id": evaluation_id,
                "original_goal": original_goal,
                "tool_used": tool_used
            }
            return await error_handler.handle_error(e, error_context)
    
    def _build_evaluation_prompt(
        self,
        original_goal: str,
        tool_used: str,
        tool_params: Dict[str, Any],
        execution_result: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """Build the evaluation prompt for AI assessment."""
        
        return f"""
EVALUATION REQUEST:
Please evaluate whether the tool execution successfully achieved the original goal.

ORIGINAL GOAL: {original_goal}

TOOL EXECUTED: {tool_used}
TOOL PARAMETERS: {json.dumps(tool_params, indent=2)}

EXECUTION RESULT:
{json.dumps(execution_result, indent=2)}

ADDITIONAL CONTEXT: {json.dumps(context or {}, indent=2)}

Please provide your evaluation in the following JSON format:
{{
    "success": true/false,
    "confidence": 0.0-1.0,
    "goal_achieved": true/false,
    "assessment": "detailed explanation of why the goal was/wasn't achieved",
    "issues_found": ["list", "of", "any", "issues"],
    "recommendations": {{
        "action": "continue|replan|improve_tool|create_new_tool",
        "reason": "explanation of recommended action",
        "details": "specific details about what should be done"
    }},
    "suggested_improvements": ["list", "of", "suggested", "improvements"]
}}

Focus on:
1. Did the execution produce the expected outcome?
2. Are there any errors or unexpected results?
3. Could the tool or process be improved?
4. Is additional action needed to fully achieve the goal?
"""
    
    def _parse_evaluation_response(self, ai_response: str) -> Dict[str, Any]:
        """Parse the AI evaluation response into structured data."""
        try:
            # Try to extract JSON from the response
            if "```json" in ai_response:
                json_start = ai_response.find("```json") + 7
                json_end = ai_response.find("```", json_start)
                json_str = ai_response[json_start:json_end].strip()
            elif "{" in ai_response:
                json_start = ai_response.find("{")
                json_end = ai_response.rfind("}") + 1
                json_str = ai_response[json_start:json_end]
            else:
                raise ValueError("No JSON found in AI response")
            
            parsed_result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["success", "confidence", "goal_achieved", "assessment"]
            for field in required_fields:
                if field not in parsed_result:
                    parsed_result[field] = None
            
            # Ensure recommendations structure
            if "recommendations" not in parsed_result:
                parsed_result["recommendations"] = {
                    "action": "continue",
                    "reason": "No specific recommendation provided",
                    "details": ""
                }
            
            return parsed_result
            
        except Exception as e:
            self.logger.error(f"Failed to parse AI evaluation response: {e}")
            # Return default evaluation
            return {
                "success": False,
                "confidence": 0.0,
                "goal_achieved": False,
                "assessment": f"Failed to parse evaluation response: {ai_response}",
                "issues_found": ["evaluation_parsing_failed"],
                "recommendations": {
                    "action": "replan",
                    "reason": "Could not properly evaluate the result",
                    "details": "Manual review needed"
                },
                "suggested_improvements": ["improve_evaluation_parsing"]
            }
    
    async def _log_evaluation(self, evaluation_result: Dict[str, Any]):
        """Log the evaluation result to strategic decisions log."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "evaluation",
            "evaluation_id": evaluation_result.get("evaluation_id"),
            "goal_achieved": evaluation_result.get("goal_achieved"),
            "success": evaluation_result.get("success"),
            "confidence": evaluation_result.get("confidence"),
            "assessment": evaluation_result.get("assessment"),
            "recommendations": evaluation_result.get("recommendations"),
            "original_goal": evaluation_result.get("original_goal"),
            "tool_used": evaluation_result.get("tool_used")
        }
        
        # Append to strategic decisions log
        strategic_log_path = Path("logs/strategic_decisions.md")
        with open(strategic_log_path, "a", encoding="utf-8") as f:
            f.write(f"\n## Evaluation Result - {evaluation_result.get('evaluation_id')}\n")
            f.write(f"**Timestamp:** {log_entry['timestamp']}\n")
            f.write(f"**Original Goal:** {log_entry['original_goal']}\n")
            f.write(f"**Tool Used:** {log_entry['tool_used']}\n")
            f.write(f"**Goal Achieved:** {log_entry['goal_achieved']}\n")
            f.write(f"**Success:** {log_entry['success']}\n")
            f.write(f"**Confidence:** {log_entry['confidence']}\n")
            f.write(f"**Assessment:** {log_entry['assessment']}\n")
            f.write(f"**Recommended Action:** {log_entry['recommendations'].get('action', 'none')}\n")
            f.write(f"**Reason:** {log_entry['recommendations'].get('reason', 'none')}\n\n")
        
        self.logger.info(f"Logged evaluation result: {evaluation_result.get('evaluation_id')}")
    
    async def _handle_evaluation_result(self, evaluation_result: Dict[str, Any]):
        """Handle the evaluation result by triggering appropriate actions."""
        recommendations = evaluation_result.get("recommendations", {})
        action = recommendations.get("action", "continue")
        
        if action == "continue":
            # Goal achieved, no further action needed
            await event_bus.publish("evaluation.success", evaluation_result)
            
        elif action == "replan":
            # Goal not achieved, trigger replanning
            await event_bus.publish("evaluation.replan_needed", evaluation_result)
            
        elif action == "improve_tool":
            # Tool worked but could be improved
            await event_bus.publish("evaluation.tool_improvement_needed", evaluation_result)
            
        elif action == "create_new_tool":
            # Current tool insufficient, need new tool
            await event_bus.publish("evaluation.new_tool_needed", evaluation_result)
        
        self.logger.info(f"Handled evaluation result with action: {action}")
    
    async def _on_execution_completed(self, event_data: Dict[str, Any]):
        """Handle tool execution completion events."""
        if event_data and "auto_evaluate" in event_data and event_data["auto_evaluate"]:
            # Auto-evaluate the result
            await self.evaluate_execution_result(
                original_goal=event_data.get("original_goal", "Unknown"),
                tool_used=event_data.get("tool_name", "Unknown"),
                tool_params=event_data.get("tool_params", {}),
                execution_result=event_data.get("result", {}),
                context=event_data.get("context", {})
            )
    
    async def _on_execution_failed(self, event_data: Dict[str, Any]):
        """Handle execution failure events."""
        # Create failure evaluation
        failure_evaluation = {
            "success": False,
            "confidence": 1.0,
            "goal_achieved": False,
            "assessment": f"Execution failed: {event_data.get('error', 'Unknown error')}",
            "issues_found": ["execution_failure"],
            "recommendations": {
                "action": "replan",
                "reason": "Execution failed, need to retry with different approach",
                "details": event_data.get("error_details", "")
            }
        }
        
        await self._handle_evaluation_result(failure_evaluation)
    
    def get_evaluation_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent evaluation history."""
        if limit:
            return self.evaluation_history[-limit:]
        return self.evaluation_history.copy()
    
    def get_success_rate(self) -> float:
        """Calculate success rate of recent evaluations."""
        if not self.evaluation_history:
            return 0.0
        
        successful = sum(1 for eval_result in self.evaluation_history 
                        if eval_result.get("goal_achieved", False))
        return successful / len(self.evaluation_history)

    # Legacy method for backward compatibility
    async def evaluate_result(self, result, expected_outcome):
        """Legacy method - use evaluate_execution_result instead."""
        return await self.evaluate_execution_result(
            original_goal=str(expected_outcome),
            tool_used="unknown",
            tool_params={},
            execution_result=result
        )

# Create singleton instance
evaluator = Evaluator()

# Backward compatibility
ExecutionEvaluator = Evaluator
