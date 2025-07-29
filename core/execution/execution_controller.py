"""
Project-S V2 Execution Controller
=================================
Bridge between AI reasoning and tool execution.
This is the critical component that makes Project-S actually DO things.
"""

import asyncio
import logging
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ExecutionController:
    """
    Main execution controller that bridges AI reasoning to tool execution.
    This is where the magic happens - AI thinks, Controller executes.
    """
    
    def __init__(self):
        """Initialize the execution controller."""
        self.tools = {}
        self.execution_history = []
        self.current_task_id = None
        
        # Load basic tools
        self._load_basic_tools()
        
        logger.info("🎯 ExecutionController initialized")
    
    def _load_basic_tools(self):
        """Load basic tool set."""
        try:
            from tools.implementations.basic_toolset import BASIC_TOOLS, TOOL_CAPABILITIES
            
            for tool_name, tool_class in BASIC_TOOLS.items():
                self.tools[tool_name] = {
                    "class": tool_class,
                    "capabilities": TOOL_CAPABILITIES.get(tool_name, []),
                    "instance": tool_class()
                }
            
            logger.info(f"✅ Loaded {len(self.tools)} basic tools")
            for tool_name in self.tools.keys():
                logger.info(f"  - {tool_name}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load basic tools: {e}")
    
    async def execute_task(self, task_description: str, use_ai_decomposition: bool = True) -> Dict[str, Any]:
        """
        Execute a natural language task.
        This is the main entry point for task execution.
        """
        task_id = f"task_{int(datetime.now().timestamp())}"
        self.current_task_id = task_id
        
        logger.info(f"🎯 Executing task {task_id}: {task_description}")
        
        try:
            if use_ai_decomposition:
                # Use AI to break down the task
                steps = await self._decompose_task_with_ai(task_description)
            else:
                # Use simple pattern matching
                steps = self._decompose_task_simple(task_description)
            
            logger.info(f"📋 Task decomposed into {len(steps)} steps")
            
            # Execute each step
            results = []
            for i, step in enumerate(steps, 1):
                logger.info(f"🔄 Step {i}/{len(steps)}: {step.get('description', step)}")
                
                step_result = await self._execute_step(step)
                results.append({
                    "step": i,
                    "step_data": step,
                    "result": step_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                if step_result.get("status") == "error":
                    logger.warning(f"⚠️ Step {i} failed, continuing...")
            
            # Compile final result
            execution_result = {
                "task_id": task_id,
                "description": task_description,
                "steps_total": len(steps),
                "steps_completed": len([r for r in results if r["result"].get("status") == "success"]),
                "results": results,
                "status": "completed",
                "timestamp": datetime.now().isoformat()
            }
            
            self.execution_history.append(execution_result)
            
            logger.info(f"✅ Task {task_id} completed: {execution_result['steps_completed']}/{execution_result['steps_total']} steps successful")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"❌ Task execution failed: {e}")
            return {
                "task_id": task_id,
                "description": task_description,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def _decompose_task_with_ai(self, task_description: str) -> List[Dict[str, Any]]:
        """Use AI to decompose task into executable steps."""
        try:
            from integrations.ai_models.multi_model_ai_client import AIClient
            ai_client = AIClient()
            
            prompt = f"""
            Break down this task into specific executable steps using these available tools:
            
            Available Tools:
            - file_system_utility: create_directory, create_file, write_to_file, read_file, list_directory
            - date_utility: get_current_date, get_current_datetime
            - system_command_utility: execute_command
            
            Task: {task_description}
            
            Please respond with a JSON array of steps. Each step should have:
            - tool: the tool name
            - action: the specific method to call
            - parameters: the parameters to pass
            - description: human readable description
            
            Example format:
            [
                {{
                    "tool": "file_system_utility",
                    "action": "create_directory", 
                    "parameters": {{"path": "reports"}},
                    "description": "Create reports directory"
                }}
            ]
            """
            
            response = await ai_client.generate_response(
                prompt=prompt,
                model="qwen3-235b",
                system_message="You are a task decomposition AI. Always respond with valid JSON arrays of executable steps."
            )
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                steps_json = json_match.group()
                steps = json.loads(steps_json)
                logger.info(f"🤖 AI decomposed task into {len(steps)} steps")
                return steps
            else:
                logger.warning("⚠️ Could not extract JSON from AI response, falling back to simple decomposition")
                return self._decompose_task_simple(task_description)
                
        except Exception as e:
            logger.error(f"❌ AI decomposition failed: {e}, falling back to simple decomposition")
            return self._decompose_task_simple(task_description)
    
    def _decompose_task_simple(self, task_description: str) -> List[Dict[str, Any]]:
        """Simple pattern-based task decomposition."""
        steps = []
        
        # Simple pattern matching for common tasks
        if "create" in task_description.lower() and "file" in task_description.lower():
            steps.append({
                "tool": "file_system_utility",
                "action": "create_file",
                "parameters": {"path": "example.txt", "content": "Generated by Project-S V2"},
                "description": "Create file as requested"
            })
        
        if "directory" in task_description.lower() and "create" in task_description.lower():
            steps.append({
                "tool": "file_system_utility", 
                "action": "create_directory",
                "parameters": {"path": "new_directory"},
                "description": "Create directory as requested"
            })
        
        if not steps:
            # Default action for unknown tasks
            steps.append({
                "tool": "date_utility",
                "action": "get_current_datetime",
                "parameters": {},
                "description": "Get current time (default action)"
            })
        
        return steps
    
    async def _execute_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single step."""
        try:
            tool_name = step.get("tool")
            action = step.get("action")
            parameters = step.get("parameters", {})
            
            if tool_name not in self.tools:
                return {"status": "error", "message": f"Tool '{tool_name}' not found"}
            
            tool_instance = self.tools[tool_name]["instance"]
            
            if not hasattr(tool_instance, action):
                return {"status": "error", "message": f"Action '{action}' not found in tool '{tool_name}'"}
            
            # Execute the tool method
            method = getattr(tool_instance, action)
            result = await method(**parameters)
            
            logger.info(f"✅ Executed {tool_name}.{action} successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Step execution failed: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_available_tools(self) -> Dict[str, Any]:
        """Get list of available tools and their capabilities."""
        return {
            tool_name: {
                "capabilities": tool_data["capabilities"]
            }
            for tool_name, tool_data in self.tools.items()
        }
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.execution_history
