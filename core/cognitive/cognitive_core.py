import logging
import asyncio
import json
import uuid
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import uuid

# Temporarily mock missing dependencies for clean V2 startup
class MockEventBus:
    def subscribe(self, event, handler): pass

class MockExecutor:
    async def execute(self, task): return {"status": "mock_success"}

class MockErrorHandler:
    async def handle_error(self, error, context=None): 
        print(f"Error handled: {error}")
        return {"status": "error_handled", "error": str(error)}

class MockMonitor:
    def monitor_performance(self, func): return func

# Use mocks until we implement these properly in V2
event_bus = MockEventBus()
executor = MockExecutor()
error_handler = MockErrorHandler()

def monitor_performance(func):
    return func

# Mock LangGraph integration for now
cognitive_core_langgraph = None
CognitiveCoreWithLangGraph = None

logger = logging.getLogger(__name__)

class CognitiveCore:
    """
    Cognitive core for the Project-S agent system.
    
    Responsible for:
    1. Maintaining context between commands
    2. Breaking down complex tasks into simpler steps
    3. Learning from past interactions and results
    4. Suggesting next actions based on current context
    """
    
    def __init__(self):
        """Initialize the cognitive core with empty context and task history."""
        # Current context of the agent's operations
        self.context = {
            "current_working_directory": None,
            "last_command": None,
            "last_result": None,
            "active_project": None,
            "session_history": [],
            "learned_patterns": {},
            "user_preferences": {}
        }
        
        # Task decomposition and planning
        self.task_history = []
        self.current_plan = None
        self.plan_step = 0
        
        # Context awareness and learning
        self.cognitive_patterns = {}
        self.success_patterns = {}
        self.failure_patterns = {}
        
        # Current context of the agent's operations
        self.context = {
            "conversation": [],
            "tasks": {},
            "entities": {},
            "workspace": {},
            "session_start": datetime.now().isoformat()
        }
        
        # Task processing state
        self.active_tasks = set()
        self.completed_tasks = set()
        self.task_dependencies = {}
        self.task_results = {}
        
        # Register for events
        event_bus.subscribe("command.completed", self._on_command_completed)
        event_bus.subscribe("command.error", self._on_command_error)
        
        logger.info("🧠 Cognitive Core initialized")
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a high-level task by breaking it into steps and executing them.
        
        Args:
            task (Dict[str, Any]): The high-level task specification
            
        Returns:
            Dict[str, Any]: The result of the task processing
        """
        try:
            task_id = task.get("id", f"task_{len(self.context['tasks']) + 1}")
            logger.info(f"Processing task: {task_id}")
            
            # Add to active tasks
            self.active_tasks.add(task_id)
            self.context["tasks"][task_id] = task
            
            # Break down the task into steps
            steps = await self._break_down_task(task)
            logger.info(f"Task {task_id} broken down into {len(steps)} steps")
            
            # Execute each step
            results = []
            for step_num, step in enumerate(steps, 1):
                step_id = f"{task_id}_step_{step_num}"
                logger.info(f"Executing step {step_id}: {step.get('description', '')}")
                
                # Create command from step
                command = self._create_command_from_step(step, task_context=task)
                
                # Execute the command
                try:
                    step_result = await executor.execute(command)
                    results.append({
                        "step_id": step_id,
                        "step": step,
                        "result": step_result,
                        "status": "completed"
                    })
                except Exception as e:
                    error_context = {"component": "cognitive_core", "task_id": task_id, "step_id": step_id}
                    await error_handler.handle_error(e, error_context)
                    results.append({
                        "step_id": step_id,
                        "step": step,
                        "error": str(e),
                        "status": "failed"
                    })
                    
                    # Check if this is a critical step
                    if step.get("critical", False):
                        logger.error(f"Critical step {step_id} failed, aborting task {task_id}")
                        break
            
            # Move from active to completed
            self.active_tasks.remove(task_id)
            self.completed_tasks.add(task_id)
            
            # Store the results
            task_result = {
                "task_id": task_id,
                "steps": results,
                "status": "completed",
                "completed_at": datetime.now().isoformat()
            }
            self.task_results[task_id] = task_result
            
            # Update the context with task results
            self._update_context_from_task(task_id, task_result)
            
            return task_result
            
        except Exception as e:
            logger.error(f"Error processing task: {str(e)}")
            error_context = {"component": "cognitive_core", "operation": "process_task"}
            await error_handler.handle_error(e, error_context)
            
            if task_id in self.active_tasks:
                self.active_tasks.remove(task_id)
            
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _break_down_task(self, task: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Break down a complex task into simpler steps that can be executed.
        
        Args:
            task (Dict[str, Any]): The task to break down
            
        Returns:
            List[Dict[str, Any]]: List of step definitions
        """
        task_type = task.get("type", "").lower()
        steps = []
        
        # If steps are already defined, use them
        if "steps" in task and isinstance(task["steps"], list):
            return task["steps"]
        
        # Otherwise, determine steps based on task type
        if task_type == "query":
            # Simple query task
            steps = [{
                "type": "ASK",
                "command": task.get("query", ""),
                "description": "Process query"
            }]
        
        elif task_type == "file_operation":
            # File operation task
            action = task.get("action", "")
            path = task.get("path", "")
            
            steps = [{
                "type": "FILE",
                "command": {
                    "action": action,
                    "path": path
                },
                "description": f"Perform {action} operation on {path}"
            }]
        
        elif task_type == "code_analysis":
            # Code analysis task - break into retrieve and analyze steps
            code_path = task.get("path", "")
            
            # Step 1: Read the file
            steps.append({
                "type": "FILE",
                "command": {
                    "action": "read",
                    "path": code_path
                },
                "description": f"Read code file {code_path}",
                "critical": True  # Mark as critical - if this fails, abort the task
            })
            
            # Step 2: Analyze the code
            steps.append({
                "type": "CODE",
                "command": {
                    "action": "analyze",
                    "code": "{step_1_result}"  # Will be replaced with actual result
                },
                "description": "Analyze the code",
                "depends_on": "step_1"
            })
        
        else:
            # Default to a single step that passes through the task
            steps = [{
                "type": task.get("command_type", "ASK"),
                "command": task.get("command", ""),
                "description": "Execute command"
            }]
        
        return steps
    
    def _create_command_from_step(self, step: Dict[str, Any], task_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a command object from a step definition.
        
        Args:
            step (Dict[str, Any]): The step definition
            task_context (Dict[str, Any]): The context of the parent task
            
        Returns:
            Dict[str, Any]: The command to execute
        """
        command = {
            "type": step.get("type", "ASK"),
            "command": step.get("command", "")
        }
        
        # Handle template substitution for dynamic values
        if isinstance(command["command"], str) and "{" in command["command"]:
            # Simple template substitution
            for key, value in task_context.items():
                placeholder = "{" + key + "}"
                if placeholder in command["command"]:
                    command["command"] = command["command"].replace(placeholder, str(value))
        
        # Copy any additional fields from the step
        for key, value in step.items():
            if key not in ["type", "command", "description", "critical", "depends_on"]:
                command[key] = value
        
        return command
    
    def _update_context_from_task(self, task_id: str, task_result: Dict[str, Any]) -> None:
        """
        Update the context with information from a completed task.
        
        Args:
            task_id (str): The ID of the completed task
            task_result (Dict[str, Any]): The result of the task execution
        """
        # Store the task result
        self.context["tasks"][task_id] = {
            **self.context["tasks"].get(task_id, {}),
            "result": task_result,
            "status": task_result.get("status", "unknown")
        }
        
        # Extract entities if available
        for step_result in task_result.get("steps", []):
            result = step_result.get("result", {})
            
            # For code analysis, store code entities
            if step_result.get("step", {}).get("type") == "CODE" and "analysis" in result:
                # Extract entities from analysis (simplified example)
                self._extract_entities_from_analysis(result.get("analysis", ""))
    
    def _extract_entities_from_analysis(self, analysis: str) -> None:
        """
        Extract entities from code analysis results and add to context.
        
        Args:
            analysis (str): The code analysis text
        """
        # Simple entity extraction (in a real system, this would be more sophisticated)
        # Example: Look for mentions of classes, functions, variables
        # This is a placeholder implementation
        entities = set()
        
        # Very naive extraction for demonstration purposes only
        lines = analysis.split("\n")
        for line in lines:
            line = line.strip()
            # Look for mentions of classes
            if "class " in line:
                parts = line.split("class ")[1].split("(")[0].split(":")
                class_name = parts[0].strip()
                entities.add(("class", class_name))
            
            # Look for mentions of functions
            if "function " in line or "method " in line:
                for marker in ["function ", "method "]:
                    if marker in line:
                        parts = line.split(marker)[1].split("(")[0]
                        func_name = parts.strip()
                        entities.add(("function", func_name))
        
        # Add entities to context
        for entity_type, entity_name in entities:
            if entity_type not in self.context["entities"]:
                self.context["entities"][entity_type] = []
            
            if entity_name not in self.context["entities"][entity_type]:
                self.context["entities"][entity_type].append(entity_name)
    
    async def _on_command_completed(self, event_data: Any) -> None:
        """
        Event handler for command.completed events.
        Update context based on command results.
        
        Args:
            event_data (Any): The event data containing command and result
        """
        if not isinstance(event_data, dict):
            return
            
        command = event_data.get("command", {})
        result = event_data.get("result", {})
        
        # Add to conversation history
        self.context["conversation"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "command",
            "command": command,
            "result": result
        })
        
        # Update workspace information for FILE commands
        if command.get("type") == "FILE":
            self._update_workspace_info(command, result)
    
    async def _on_command_error(self, event_data: Any) -> None:
        """
        Event handler for command.error events.
        Update context with error information.
        
        Args:
            event_data (Any): The event data containing command and error
        """
        if not isinstance(event_data, dict):
            return
            
        command = event_data.get("command", {})
        error = event_data.get("error", "Unknown error")
        
        # Add to conversation history
        self.context["conversation"].append({
            "timestamp": datetime.now().isoformat(),
            "type": "error",
            "command": command,
            "error": error
        })
    
    def _update_workspace_info(self, command: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Update workspace information based on file operations.
        
        Args:
            command (Dict[str, Any]): The FILE command
            result (Dict[str, Any]): The result of the command
        """
        # Extract command details
        cmd_obj = command.get("command", {})
        if isinstance(cmd_obj, str):
            # Try to parse JSON string
            try:
                cmd_obj = json.loads(cmd_obj)
            except:
                cmd_obj = {"action": "unknown", "path": cmd_obj}
        
        action = cmd_obj.get("action", "")
        path = cmd_obj.get("path", "")
        
        if not path:
            return
            
        # Update workspace based on the action
        if action == "read" and "content" in result:
            self.context["workspace"][path] = {
                "last_read": datetime.now().isoformat(),
                "content_preview": result["content"][:100] + "..." if len(result["content"]) > 100 else result["content"]
            }
        elif action == "write":
            self.context["workspace"][path] = {
                "last_write": datetime.now().isoformat()
            }
        elif action == "list" and "files" in result:
            directory = path
            for file in result["files"]:
                file_path = f"{directory}/{file}" if directory != "." else file
                if file_path not in self.context["workspace"]:
                    self.context["workspace"][file_path] = {
                        "discovered": datetime.now().isoformat()
                    }
    
    @monitor_performance
    async def suggest_next_action(self) -> Optional[Dict[str, Any]]:
        """
        Suggest the next action based on current context.
        
        Returns:
            Optional[Dict[str, Any]]: A suggested action or None
        """
        # This is a simple implementation that could be expanded with more intelligence
        
        # If there are active tasks, don't suggest anything
        if self.active_tasks:
            return None
        
        # Analyze recent conversation
        recent_items = self.context["conversation"][-5:] if len(self.context["conversation"]) > 5 else self.context["conversation"]
        
        # Simple rule-based suggestions
        for item in reversed(recent_items):
            if item["type"] == "command" and item["command"].get("type") == "ASK":
                # After an ASK command, suggest a follow-up
                return {
                    "type": "suggestion",
                    "action": {
                        "type": "ASK",
                        "command": "Would you like me to explain this in more detail?"
                    },
                    "confidence": 0.7,
                    "reason": "Follow-up to previous query"
                }
            
            if item["type"] == "command" and item["command"].get("type") == "FILE" and item["command"].get("action") == "read":
                # After reading a file, suggest analysis
                return {
                    "type": "suggestion",
                    "action": {
                        "type": "CODE",
                        "command": {
                            "action": "analyze",
                            "code": "The previously read file content"
                        }
                    },
                    "confidence": 0.8,
                    "reason": "Analyze the file you just read"
                }
        
        return None
    
    def get_context(self) -> Dict[str, Any]:
        """
        Get the current context.
        
        Returns:
            Dict[str, Any]: The current context
        """
        return self.context
    
    def clear_context(self) -> None:
        """Clear the current context."""
        self.context = {
            "conversation": [],
            "tasks": {},
            "entities": {},
            "workspace": {},
            "session_start": datetime.now().isoformat()
        }
        self.active_tasks = set()
        self.completed_tasks = set()
        self.task_dependencies = {}
        self.task_results = {}
        
        logger.info("Context cleared")
    
    async def parse_task(self, task_description: str) -> Dict[str, Any]:
        """
        Parse and analyze a natural language task description.
        
        Args:
            task_description: Natural language description of the task
            
        Returns:
            Dictionary containing parsed task information and analysis
        """
        logger.info(f"🧠 Parsing task: {task_description[:100]}...")
        
        try:
            # Basic task analysis
            task_analysis = {
                "original_description": task_description,
                "task_id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
                "complexity": "unknown",
                "estimated_steps": 1,
                "required_resources": [],
                "success_criteria": [],
                "risks": []
            }
            
            # Analyze task complexity and type
            complexity_analysis = self._analyze_task_complexity(task_description)
            task_analysis.update(complexity_analysis)
            
            # Break down into steps if complex
            if task_analysis["complexity"] in ["moderate", "high"]:
                steps = self._decompose_task(task_description)
                task_analysis["decomposed_steps"] = steps
                task_analysis["estimated_steps"] = len(steps)
            
            # Identify required resources and tools
            resources = self._identify_required_resources(task_description)
            task_analysis["required_resources"] = resources
            
            # Define success criteria
            success_criteria = self._define_success_criteria(task_description)
            task_analysis["success_criteria"] = success_criteria
            
            # Identify potential risks
            risks = self._identify_risks(task_description)
            task_analysis["risks"] = risks
            
            # Store in context for future reference
            self.context["last_parsed_task"] = task_analysis
            
            logger.info(f"✅ Task parsed successfully: {task_analysis['complexity']} complexity, {task_analysis['estimated_steps']} steps")
            return task_analysis
            
        except Exception as e:
            logger.error(f"❌ Task parsing failed: {e}")
            return {
                "original_description": task_description,
                "status": "parsing_failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_task_complexity(self, task_description: str) -> Dict[str, Any]:
        """Analyze the complexity of a task."""
        
        task_lower = task_description.lower()
        word_count = len(task_description.split())
        
        # Basic complexity indicators
        complexity_indicators = {
            "high": ["integrate", "complex", "multiple", "comprehensive", "sophisticated", "advanced"],
            "moderate": ["create", "build", "develop", "implement", "design", "configure"],
            "low": ["show", "display", "list", "check", "simple", "quick"]
        }
        
        complexity = "low"
        for level, indicators in complexity_indicators.items():
            if any(indicator in task_lower for indicator in indicators):
                complexity = level
                break
        
        # Adjust based on length
        if word_count > 20:
            complexity = "high" if complexity == "moderate" else "moderate" if complexity == "low" else complexity
        
        return {
            "complexity": complexity,
            "word_count": word_count,
            "complexity_factors": [indicator for level, indicators in complexity_indicators.items() 
                                 for indicator in indicators if indicator in task_lower]
        }
    
    def _decompose_task(self, task_description: str) -> List[Dict[str, Any]]:
        """Break down complex tasks into smaller steps."""
        
        # Basic task decomposition logic
        steps = []
        task_lower = task_description.lower()
        
        # Common task patterns
        if "create" in task_lower or "build" in task_lower:
            steps.extend([
                {"step": 1, "action": "analyze_requirements", "description": "Analyze task requirements"},
                {"step": 2, "action": "plan_implementation", "description": "Plan implementation approach"},
                {"step": 3, "action": "implement_solution", "description": "Implement the solution"},
                {"step": 4, "action": "test_and_validate", "description": "Test and validate the result"}
            ])
        
        elif "analyze" in task_lower or "examine" in task_lower:
            steps.extend([
                {"step": 1, "action": "gather_data", "description": "Gather relevant data"},
                {"step": 2, "action": "perform_analysis", "description": "Perform detailed analysis"},
                {"step": 3, "action": "compile_results", "description": "Compile analysis results"}
            ])
        
        else:
            # Generic decomposition
            steps.extend([
                {"step": 1, "action": "understand_task", "description": "Understand task requirements"},
                {"step": 2, "action": "execute_task", "description": "Execute the task"},
                {"step": 3, "action": "verify_completion", "description": "Verify task completion"}
            ])
        
        return steps
    
    def _identify_required_resources(self, task_description: str) -> List[str]:
        """Identify resources required for the task."""
        
        resources = []
        task_lower = task_description.lower()
        
        # File system resources
        if any(keyword in task_lower for keyword in ["file", "folder", "directory", "create", "write"]):
            resources.append("file_system_access")
        
        # Programming languages
        programming_languages = ["python", "javascript", "java", "html", "css", "sql"]
        for lang in programming_languages:
            if lang in task_lower:
                resources.append(f"{lang}_environment")
        
        # Network resources
        if any(keyword in task_lower for keyword in ["api", "web", "http", "download", "upload"]):
            resources.append("network_access")
        
        # External tools
        if any(keyword in task_lower for keyword in ["git", "docker", "database"]):
            resources.append("external_tools")
        
        return resources
    
    def _define_success_criteria(self, task_description: str) -> List[str]:
        """Define success criteria for the task."""
        
        criteria = []
        task_lower = task_description.lower()
        
        # Common success patterns
        if "create" in task_lower or "build" in task_lower:
            criteria.extend([
                "Solution is implemented correctly",
                "Code runs without errors",
                "Requirements are met"
            ])
        
        elif "analyze" in task_lower:
            criteria.extend([
                "Analysis is comprehensive",
                "Results are accurate",
                "Insights are provided"
            ])
        
        else:
            criteria.append("Task is completed successfully")
        
        return criteria
    
    def _identify_risks(self, task_description: str) -> List[str]:
        """Identify potential risks in task execution."""
        
        risks = []
        task_lower = task_description.lower()
        
        # Common risk patterns
        if "complex" in task_lower or "multiple" in task_lower:
            risks.append("High complexity may lead to implementation challenges")
        
        if any(keyword in task_lower for keyword in ["file", "delete", "modify"]):
            risks.append("File operations may affect existing data")
        
        if "api" in task_lower or "network" in task_lower:
            risks.append("Network dependencies may cause failures")
        
        if not risks:
            risks.append("Minimal risks identified")
        
        return risks

# Create a default cognitive core instance
legacy_cognitive_core = CognitiveCore()

# Create the enhanced LangGraph-based cognitive core
# This is now the default cognitive core used throughout the system
cognitive_core = cognitive_core_langgraph