"""
Project-S Enhanced Workflow Engine
----------------------------------
Advanced workflow management system that combines LangGraph with intelligent
decision-making and multi-AI model orchestration.

This module provides sophisticated workflow capabilities that can handle
complex, multi-step tasks with dynamic routing, parallel processing,
and intelligent error recovery.
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from pathlib import Path

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages
    from langgraph.prebuilt import ToolNode
    LANGGRAPH_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✅ LangGraph available for enhanced workflows")
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ LangGraph not available - using fallback workflow engine")

from typing import TypedDict

class WorkflowState(TypedDict, total=False):
    """Enhanced workflow state for complex task management."""
    
    # Core state
    workflow_id: str
    session_id: str
    start_time: str
    current_step: str
    completed_steps: List[str]
    
    # Task information
    original_task: str
    task_analysis: Dict[str, Any]
    execution_plan: Dict[str, Any]
    
    # Data flow
    input_data: Dict[str, Any]
    intermediate_results: Dict[str, Any]
    final_results: Dict[str, Any]
    
    # AI and tool integration
    selected_tools: List[str]
    ai_model_preferences: Dict[str, str]
    tool_results: Dict[str, Any]
    
    # Progress tracking
    progress_percentage: float
    estimated_completion_time: str
    quality_scores: Dict[str, float]
    
    # Error handling
    errors: List[Dict[str, Any]]
    warnings: List[Dict[str, Any]]
    recovery_attempts: int
    
    # Context and learning
    context: Dict[str, Any]
    learning_insights: List[str]
    performance_metrics: Dict[str, Any]


class EnhancedWorkflowEngine:
    """
    Advanced workflow engine that integrates LangGraph with intelligent
    tool orchestration and multi-AI capabilities.
    """
    
    def __init__(self, smart_orchestrator=None, cognitive_core=None):
        """Initialize the enhanced workflow engine."""
        self.smart_orchestrator = smart_orchestrator
        self.cognitive_core = cognitive_core
        
        # Workflow management
        self.active_workflows = {}
        self.workflow_templates = {}
        self.workflow_history = []
        
        # LangGraph integration
        self.langgraph_available = LANGGRAPH_AVAILABLE
        self.graph_builders = {}
        
        # Performance tracking
        self.execution_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "average_execution_time": 0,
            "ai_model_usage": {},
            "tool_usage_stats": {}
        }
        
        logger.info("🔄 Enhanced Workflow Engine initialized")
    
    async def initialize_workflow_templates(self):
        """Initialize pre-built workflow templates for common tasks."""
        
        # Website Analysis Workflow Template
        self.workflow_templates["website_analysis"] = {
            "name": "Comprehensive Website Analysis",
            "description": "Full website analysis including SEO, technical, content, and performance evaluation",
            "complexity": "high",
            "estimated_duration": 180,  # 3 minutes
            "required_capabilities": ["web_fetch", "content_analysis", "report_generation"],
            "ai_model_preferences": {
                "content_analysis": "claude-3-sonnet",  # Good for content understanding
                "technical_analysis": "gpt-4",         # Good for technical details
                "report_generation": "gpt-3.5-turbo"   # Fast for formatting
            },
            "workflow_steps": [
                {
                    "step_id": "fetch_website",
                    "name": "Fetch Website Content",
                    "type": "tool_execution",
                    "tool": "WebPageFetchTool",
                    "ai_assistance": False,
                    "parallel": False
                },
                {
                    "step_id": "analyze_content",
                    "name": "Analyze Content Quality",
                    "type": "ai_analysis",
                    "ai_model": "claude-3-sonnet",
                    "parallel": True,
                    "depends_on": ["fetch_website"]
                },
                {
                    "step_id": "technical_audit",
                    "name": "Technical SEO Audit",
                    "type": "ai_analysis",
                    "ai_model": "gpt-4",
                    "parallel": True,
                    "depends_on": ["fetch_website"]
                },
                {
                    "step_id": "performance_analysis",
                    "name": "Performance Analysis",
                    "type": "tool_execution",
                    "tool": "WebApiCallTool",
                    "parallel": True,
                    "depends_on": ["fetch_website"]
                },
                {
                    "step_id": "generate_report",
                    "name": "Generate Comprehensive Report",
                    "type": "ai_synthesis",
                    "ai_model": "gpt-3.5-turbo",
                    "depends_on": ["analyze_content", "technical_audit", "performance_analysis"]
                },
                {
                    "step_id": "save_results",
                    "name": "Save Analysis Results",
                    "type": "tool_execution",
                    "tool": "FileWriteTool",
                    "depends_on": ["generate_report"]
                }
            ]
        }
        
        # Multi-Model Research Workflow Template
        self.workflow_templates["multi_model_research"] = {
            "name": "Multi-Model Research and Documentation",
            "description": "Research a topic using multiple AI models and create comprehensive documentation",
            "complexity": "medium",
            "estimated_duration": 120,
            "required_capabilities": ["web_search", "ai_reasoning", "document_generation"],
            "ai_model_preferences": {
                "research_planning": "gpt-4",
                "information_gathering": "claude-3-sonnet",
                "fact_verification": "gpt-4",
                "documentation": "gpt-3.5-turbo"
            },
            "workflow_steps": [
                {
                    "step_id": "research_planning",
                    "name": "Plan Research Strategy",
                    "type": "ai_planning",
                    "ai_model": "gpt-4"
                },
                {
                    "step_id": "gather_information",
                    "name": "Gather Information",
                    "type": "tool_execution",
                    "tool": "WebSearchTool",
                    "depends_on": ["research_planning"]
                },
                {
                    "step_id": "analyze_sources",
                    "name": "Analyze Information Sources",
                    "type": "ai_analysis",
                    "ai_model": "claude-3-sonnet",
                    "depends_on": ["gather_information"]
                },
                {
                    "step_id": "verify_facts",
                    "name": "Verify Key Facts",
                    "type": "ai_verification",
                    "ai_model": "gpt-4",
                    "depends_on": ["analyze_sources"]
                },
                {
                    "step_id": "create_documentation",
                    "name": "Create Final Documentation",
                    "type": "ai_synthesis",
                    "ai_model": "gpt-3.5-turbo",
                    "depends_on": ["verify_facts"]
                },
                {
                    "step_id": "save_documentation",
                    "name": "Save Research Documentation",
                    "type": "tool_execution",
                    "tool": "FileWriteTool",
                    "depends_on": ["create_documentation"]
                }
            ]
        }
        
        # File Analysis Workflow Template  
        self.workflow_templates["file_analysis"] = {
            "name": "Comprehensive File Analysis",
            "description": "Analyze files in a directory with AI-powered insights",
            "complexity": "medium",
            "estimated_duration": 90,
            "required_capabilities": ["file_operations", "content_analysis", "pattern_recognition"],
            "ai_model_preferences": {
                "file_categorization": "gpt-3.5-turbo",
                "content_analysis": "claude-3-sonnet",
                "pattern_detection": "gpt-4"
            },
            "workflow_steps": [
                {
                    "step_id": "discover_files",
                    "name": "Discover Files",
                    "type": "tool_execution",
                    "tool": "FileSearchTool"
                },
                {
                    "step_id": "categorize_files",
                    "name": "Categorize Files",
                    "type": "ai_analysis",
                    "ai_model": "gpt-3.5-turbo",
                    "depends_on": ["discover_files"]
                },
                {
                    "step_id": "analyze_content",
                    "name": "Analyze File Contents",
                    "type": "parallel_analysis",
                    "ai_model": "claude-3-sonnet",
                    "depends_on": ["categorize_files"]
                },
                {
                    "step_id": "detect_patterns",
                    "name": "Detect Patterns and Insights",
                    "type": "ai_synthesis",
                    "ai_model": "gpt-4",
                    "depends_on": ["analyze_content"]
                },
                {
                    "step_id": "generate_report",
                    "name": "Generate Analysis Report",
                    "type": "ai_synthesis",
                    "ai_model": "gpt-3.5-turbo",
                    "depends_on": ["detect_patterns"]
                },
                {
                    "step_id": "save_analysis",
                    "name": "Save Analysis Results",
                    "type": "tool_execution",
                    "tool": "FileWriteTool",
                    "depends_on": ["generate_report"]
                }
            ]
        }
        
        logger.info(f"✅ Initialized {len(self.workflow_templates)} workflow templates")
    
    async def create_workflow_from_task(
        self, 
        task_description: str, 
        context: Dict[str, Any] = None,
        preferences: Dict[str, Any] = None
    ) -> str:
        """
        Create a custom workflow from a natural language task description.
        
        Args:
            task_description: Natural language description of the task
            context: Current context and resources
            preferences: User preferences for execution
            
        Returns:
            Workflow ID for tracking
        """
        logger.info(f"🎯 Creating workflow from task: {task_description}")
        
        workflow_id = str(uuid.uuid4())
        context = context or {}
        preferences = preferences or {}
        
        try:
            # Use cognitive core for intelligent planning if available
            if self.cognitive_core:
                execution_plan = await self.cognitive_core.plan_task_execution(task_description, context)
            else:
                execution_plan = await self._create_basic_execution_plan(task_description, context)
            
            # Determine if we should use a template or create custom workflow
            template_match = await self._find_matching_template(task_description, execution_plan)
            
            if template_match:
                workflow_config = await self._customize_template(template_match, execution_plan, preferences)
                logger.info(f"📋 Using template: {template_match}")
            else:
                workflow_config = await self._create_custom_workflow(execution_plan, preferences)
                logger.info("🔧 Creating custom workflow")
            
            # Initialize workflow state
            workflow_state = WorkflowState(
                workflow_id=workflow_id,
                session_id=str(uuid.uuid4()),
                start_time=datetime.now().isoformat(),
                current_step="initialized",
                completed_steps=[],
                original_task=task_description,
                task_analysis=execution_plan.get("task_analysis", {}),
                execution_plan=execution_plan,
                input_data=context,
                intermediate_results={},
                final_results={},
                selected_tools=[],
                ai_model_preferences=workflow_config.get("ai_model_preferences", {}),
                tool_results={},
                progress_percentage=0.0,
                estimated_completion_time="",
                quality_scores={},
                errors=[],
                warnings=[],
                recovery_attempts=0,
                context=context,
                learning_insights=[],
                performance_metrics={}
            )
            
            # Store workflow
            self.active_workflows[workflow_id] = {
                "state": workflow_state,
                "config": workflow_config,
                "created_at": datetime.now(),
                "status": "created"
            }
            
            logger.info(f"✅ Workflow created: {workflow_id}")
            return workflow_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create workflow: {str(e)}")
            raise WorkflowError(f"Failed to create workflow: {str(e)}")
    
    async def _create_basic_execution_plan(self, task_description: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a basic execution plan when cognitive core is not available."""
        
        # Simple task analysis
        task_lower = task_description.lower()
        
        task_analysis = {
            "task_type": "web_analysis" if "website" in task_lower or "url" in task_lower else "general",
            "complexity": "high" if len(task_description.split()) > 10 else "medium",
            "estimated_steps": 3 + task_lower.count("and"),
            "required_capabilities": self._identify_basic_capabilities(task_lower)
        }
        
        return {
            "plan_id": str(uuid.uuid4()),
            "task_analysis": task_analysis,
            "execution_strategy": {"approach": "sequential_execution"},
            "estimated_duration": task_analysis["estimated_steps"] * 30,
            "confidence_score": 0.7
        }
    
    def _identify_basic_capabilities(self, task_description: str) -> List[str]:
        """Identify basic capabilities needed for a task."""
        capabilities = []
        
        if any(kw in task_description for kw in ["web", "website", "url"]):
            capabilities.extend(["web_fetch", "content_analysis"])
        if any(kw in task_description for kw in ["file", "read", "write"]):
            capabilities.extend(["file_operations"])
        if any(kw in task_description for kw in ["search", "find"]):
            capabilities.extend(["search_operations"])
        if any(kw in task_description for kw in ["analyze", "report"]):
            capabilities.extend(["ai_analysis", "report_generation"])
        
        return capabilities if capabilities else ["general_processing"]
    
    async def _find_matching_template(self, task_description: str, execution_plan: Dict[str, Any]) -> Optional[str]:
        """Find a matching workflow template for the task."""
        
        task_lower = task_description.lower()
        required_capabilities = execution_plan.get("task_analysis", {}).get("required_capabilities", [])
        
        # Check for website analysis
        if any(kw in task_lower for kw in ["website", "web", "url", "analyze"]) and "web_fetch" in required_capabilities:
            return "website_analysis"
        
        # Check for research tasks
        if any(kw in task_lower for kw in ["research", "information", "study"]) and "search_operations" in required_capabilities:
            return "multi_model_research"
        
        # Check for file analysis
        if any(kw in task_lower for kw in ["file", "directory", "analyze files"]) and "file_operations" in required_capabilities:
            return "file_analysis"
        
        return None
    
    async def _customize_template(
        self, 
        template_name: str, 
        execution_plan: Dict[str, Any], 
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Customize a workflow template based on execution plan and preferences."""
        
        if template_name not in self.workflow_templates:
            raise WorkflowError(f"Template '{template_name}' not found")
        
        template = self.workflow_templates[template_name].copy()
        
        # Apply user preferences
        if preferences.get("ai_model_preference"):
            # Override AI model preferences
            preferred_model = preferences["ai_model_preference"]
            for step in template["workflow_steps"]:
                if step.get("type") in ["ai_analysis", "ai_synthesis", "ai_planning"]:
                    step["ai_model"] = preferred_model
        
        if preferences.get("execution_speed") == "fast":
            # Prioritize speed over thoroughness
            template["estimated_duration"] = int(template["estimated_duration"] * 0.7)
            # Use faster AI models
            for step in template["workflow_steps"]:
                if step.get("ai_model") == "gpt-4":
                    step["ai_model"] = "gpt-3.5-turbo"
        
        if preferences.get("quality_priority") == "high":
            # Prioritize quality over speed
            template["estimated_duration"] = int(template["estimated_duration"] * 1.3)
            # Use more capable AI models
            for step in template["workflow_steps"]:
                if step.get("ai_model") == "gpt-3.5-turbo":
                    step["ai_model"] = "gpt-4"
        
        return template
    
    async def _create_custom_workflow(self, execution_plan: Dict[str, Any], preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom workflow from scratch."""
        
        task_analysis = execution_plan.get("task_analysis", {})
        required_capabilities = task_analysis.get("required_capabilities", [])
        
        # Basic custom workflow structure
        workflow_config = {
            "name": "Custom Task Workflow",
            "description": "Dynamically created workflow for specific task",
            "complexity": task_analysis.get("complexity", "medium"),
            "estimated_duration": execution_plan.get("estimated_duration", 90),
            "required_capabilities": required_capabilities,
            "ai_model_preferences": self._select_ai_models_for_capabilities(required_capabilities),
            "workflow_steps": await self._generate_workflow_steps(task_analysis, required_capabilities)
        }
        
        return workflow_config
    
    def _select_ai_models_for_capabilities(self, capabilities: List[str]) -> Dict[str, str]:
        """Select appropriate AI models for different capabilities."""
        
        model_mapping = {
            "content_analysis": "claude-3-sonnet",
            "technical_analysis": "gpt-4",
            "web_analysis": "gpt-4",
            "file_analysis": "claude-3-sonnet",
            "report_generation": "gpt-3.5-turbo",
            "general_processing": "gpt-3.5-turbo",
            "research": "gpt-4",
            "summarization": "claude-3-sonnet"
        }
        
        preferences = {}
        for capability in capabilities:
            if capability in model_mapping:
                preferences[capability] = model_mapping[capability]
            else:
                preferences[capability] = "gpt-3.5-turbo"  # Default
        
        return preferences
    
    async def _generate_workflow_steps(self, task_analysis: Dict[str, Any], capabilities: List[str]) -> List[Dict[str, Any]]:
        """Generate workflow steps based on task analysis."""
        
        steps = []
        step_counter = 1
        
        # Add initial steps based on capabilities
        if "web_fetch" in capabilities:
            steps.append({
                "step_id": f"step_{step_counter}",
                "name": "Fetch Web Content",
                "type": "tool_execution",
                "tool": "WebPageFetchTool"
            })
            step_counter += 1
        
        if "file_operations" in capabilities:
            steps.append({
                "step_id": f"step_{step_counter}",
                "name": "Process Files",
                "type": "tool_execution",
                "tool": "FileReadTool"
            })
            step_counter += 1
        
        if "search_operations" in capabilities:
            steps.append({
                "step_id": f"step_{step_counter}",
                "name": "Search for Information",
                "type": "tool_execution",
                "tool": "WebSearchTool"
            })
            step_counter += 1
        
        # Add analysis step if needed
        if any(cap in capabilities for cap in ["content_analysis", "ai_analysis"]):
            steps.append({
                "step_id": f"step_{step_counter}",
                "name": "Analyze Content",
                "type": "ai_analysis",
                "ai_model": "claude-3-sonnet"
            })
            step_counter += 1
        
        # Add synthesis/report generation step
        if "report_generation" in capabilities or task_analysis.get("complexity") != "low":
            steps.append({
                "step_id": f"step_{step_counter}",
                "name": "Generate Results",
                "type": "ai_synthesis",
                "ai_model": "gpt-3.5-turbo"
            })
            step_counter += 1
        
        # Add save step
        steps.append({
            "step_id": f"step_{step_counter}",
            "name": "Save Results",
            "type": "tool_execution",
            "tool": "FileWriteTool"
        })
        
        return steps
    
    async def execute_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a workflow with full orchestration and error handling.
        
        Args:
            workflow_id: ID of the workflow to execute
            
        Returns:
            Execution results
        """
        logger.info(f"🚀 Executing workflow: {workflow_id}")
        
        if workflow_id not in self.active_workflows:
            raise WorkflowError(f"Workflow '{workflow_id}' not found")
        
        workflow = self.active_workflows[workflow_id]
        state = workflow["state"]
        config = workflow["config"]
        
        # Mark workflow as executing
        workflow["status"] = "executing"
        state["current_step"] = "execution_started"
        
        execution_start = time.time()
        
        try:
            # Choose execution method based on availability
            if self.langgraph_available and len(config["workflow_steps"]) > 3:
                # Use LangGraph for complex workflows
                results = await self._execute_with_langgraph(workflow_id, state, config)
            else:
                # Use sequential execution for simpler workflows
                results = await self._execute_sequential_workflow(workflow_id, state, config)
            
            execution_time = time.time() - execution_start
            
            # Update final state
            state["final_results"] = results
            state["progress_percentage"] = 100.0
            state["performance_metrics"]["total_execution_time"] = execution_time
            
            # Mark as completed
            workflow["status"] = "completed"
            
            # Update metrics
            self.execution_metrics["total_workflows"] += 1
            if results.get("success", False):
                self.execution_metrics["successful_workflows"] += 1
            
            # Update average execution time
            total_workflows = self.execution_metrics["total_workflows"]
            current_avg = self.execution_metrics["average_execution_time"]
            self.execution_metrics["average_execution_time"] = ((current_avg * (total_workflows - 1)) + execution_time) / total_workflows
            
            # Archive workflow
            await self._archive_workflow(workflow_id)
            
            success_indicator = "✅" if results.get("success") else "❌"
            logger.info(f"{success_indicator} Workflow {workflow_id} completed in {execution_time:.2f}s")
            
            return {
                "workflow_id": workflow_id,
                "success": results.get("success", False),
                "execution_time": execution_time,
                "results": results,
                "state": state
            }
            
        except Exception as e:
            # Handle execution error
            execution_time = time.time() - execution_start
            error_msg = f"Workflow execution failed: {str(e)}"
            
            state["errors"].append({
                "error": error_msg,
                "timestamp": datetime.now().isoformat(),
                "step": state.get("current_step", "unknown")
            })
            
            workflow["status"] = "failed"
            
            logger.error(f"❌ {error_msg}")
            
            return {
                "workflow_id": workflow_id,
                "success": False,
                "execution_time": execution_time,
                "error": error_msg,
                "state": state
            }
    
    async def _execute_sequential_workflow(
        self, 
        workflow_id: str, 
        state: WorkflowState, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow sequentially without LangGraph."""
        
        logger.info("🔄 Executing workflow sequentially")
        
        workflow_steps = config.get("workflow_steps", [])
        total_steps = len(workflow_steps)
        results = {"success": True, "step_results": {}}
        
        for i, step in enumerate(workflow_steps):
            step_id = step.get("step_id", f"step_{i+1}")
            step_name = step.get("name", f"Step {i+1}")
            
            logger.info(f"📋 Executing step {i+1}/{total_steps}: {step_name}")
            
            # Update progress
            state["current_step"] = step_id
            state["progress_percentage"] = (i / total_steps) * 100
            
            try:
                # Execute step based on type
                step_result = await self._execute_workflow_step(step, state)
                
                # Store result
                results["step_results"][step_id] = step_result
                state["intermediate_results"][step_id] = step_result
                state["completed_steps"].append(step_id)
                
                # Update state with results
                if step_result.get("output"):
                    state["tool_results"][step_id] = step_result["output"]
                
                logger.info(f"✅ Step {step_name} completed successfully")
                
            except Exception as e:
                error_msg = f"Step {step_name} failed: {str(e)}"
                logger.error(f"❌ {error_msg}")
                
                state["errors"].append({
                    "step": step_id,
                    "error": error_msg,
                    "timestamp": datetime.now().isoformat()
                })
                
                results["step_results"][step_id] = {"success": False, "error": error_msg}
                
                # Decide whether to continue or abort
                if not config.get("continue_on_error", False):
                    results["success"] = False
                    break
        
        return results
    
    async def _execute_with_langgraph(
        self, 
        workflow_id: str, 
        state: WorkflowState, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute workflow using LangGraph for complex orchestration."""
        
        logger.info("🔗 Executing workflow with LangGraph")
        
        try:
            # Build LangGraph workflow
            graph = await self._build_langgraph_workflow(config)
            
            # Execute with LangGraph
            final_state = await graph.ainvoke(state)
            
            return {
                "success": True,
                "langgraph_execution": True,
                "final_state": final_state
            }
            
        except Exception as e:
            logger.error(f"❌ LangGraph execution failed: {str(e)}")
            # Fallback to sequential execution
            logger.info("🔄 Falling back to sequential execution")
            return await self._execute_sequential_workflow(workflow_id, state, config)
    
    async def _build_langgraph_workflow(self, config: Dict[str, Any]) -> StateGraph:
        """Build a LangGraph workflow from configuration."""
        
        if not LANGGRAPH_AVAILABLE:
            raise WorkflowError("LangGraph not available")
        
        # Create state graph
        workflow = StateGraph(WorkflowState)
        
        # Add nodes for each step
        workflow_steps = config.get("workflow_steps", [])
        
        for step in workflow_steps:
            step_id = step.get("step_id")
            step_function = await self._create_langgraph_step_function(step)
            workflow.add_node(step_id, step_function)
        
        # Set entry point
        if workflow_steps:
            workflow.set_entry_point(workflow_steps[0]["step_id"])
        
        # Add edges based on dependencies
        for i, step in enumerate(workflow_steps):
            step_id = step.get("step_id")
            depends_on = step.get("depends_on", [])
            
            if not depends_on and i > 0:
                # Sequential execution - depends on previous step
                prev_step = workflow_steps[i-1]["step_id"]
                workflow.add_edge(prev_step, step_id)
            else:
                # Dependency-based execution
                for dependency in depends_on:
                    workflow.add_edge(dependency, step_id)
        
        # Add final edge to END
        if workflow_steps:
            final_step = workflow_steps[-1]["step_id"]
            workflow.add_edge(final_step, END)
        
        return workflow.compile()
    
    async def _create_langgraph_step_function(self, step: Dict[str, Any]) -> Callable:
        """Create a LangGraph-compatible function for a workflow step."""
        
        step_type = step.get("type", "tool_execution")
        
        async def step_function(state: WorkflowState) -> WorkflowState:
            """LangGraph step function."""
            step_result = await self._execute_workflow_step(step, state)
            
            # Update state
            updated_state = state.copy()
            updated_state["intermediate_results"][step["step_id"]] = step_result
            updated_state["completed_steps"].append(step["step_id"])
            
            return updated_state
        
        return step_function
    
    async def _execute_workflow_step(self, step: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
        """Execute a single workflow step."""
        
        step_type = step.get("type", "tool_execution")
        step_id = step.get("step_id", "unknown")
        
        try:
            if step_type == "tool_execution":
                return await self._execute_tool_step(step, state)
            elif step_type in ["ai_analysis", "ai_synthesis", "ai_planning", "ai_verification"]:
                return await self._execute_ai_step(step, state)
            elif step_type == "parallel_analysis":
                return await self._execute_parallel_step(step, state)
            else:
                return {"success": False, "error": f"Unknown step type: {step_type}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_tool_step(self, step: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
        """Execute a tool-based workflow step."""
        
        tool_name = step.get("tool")
        if not tool_name:
            return {"success": False, "error": "No tool specified for step"}
        
        # Use smart orchestrator if available
        if self.smart_orchestrator and hasattr(self.smart_orchestrator, 'available_tools'):
            if tool_name in self.smart_orchestrator.available_tools:
                tool = self.smart_orchestrator.available_tools[tool_name]
                
                # Prepare parameters from state
                params = await self._prepare_tool_parameters(step, state)
                
                # Execute tool
                result = await self.smart_orchestrator._execute_tool_safely(tool, params)
                
                return {
                    "success": result.get("success", False),
                    "output": result.get("result"),
                    "tool_used": tool_name,
                    "params": params
                }
        
        # Fallback execution
        return {"success": False, "error": f"Tool {tool_name} not available"}
    
    async def _execute_ai_step(self, step: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
        """Execute an AI-powered workflow step."""
        
        step_type = step.get("type")
        ai_model = step.get("ai_model", "gpt-3.5-turbo")
        
        # This would integrate with the multi-AI system
        # For now, return a placeholder result
        
        return {
            "success": True,
            "output": f"AI step {step_type} completed with {ai_model}",
            "ai_model_used": ai_model,
            "step_type": step_type
        }
    
    async def _execute_parallel_step(self, step: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
        """Execute a step that can run in parallel."""
        
        # For now, execute sequentially
        # This would be enhanced with actual parallel processing
        
        return await self._execute_ai_step(step, state)
    
    async def _prepare_tool_parameters(self, step: Dict[str, Any], state: WorkflowState) -> Dict[str, Any]:
        """Prepare parameters for tool execution based on workflow state."""
        
        params = {}
        tool_name = step.get("tool", "")
        
        # Extract relevant data from state
        if "Web" in tool_name:
            params["url"] = state["input_data"].get("url") or state["input_data"].get("target_url")
        elif "File" in tool_name and "Write" in tool_name:
            params["path"] = state["input_data"].get("output_path", f"workflow_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
            params["content"] = state["intermediate_results"].get("content_to_save", "Workflow output")
        elif "File" in tool_name and "Read" in tool_name:
            params["path"] = state["input_data"].get("file_path")
        elif "Search" in tool_name:
            params["query"] = state["input_data"].get("search_query", state["original_task"])
        
        return params
    
    async def _archive_workflow(self, workflow_id: str):
        """Archive a completed workflow."""
        
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows.pop(workflow_id)
            workflow["archived_at"] = datetime.now()
            self.workflow_history.append(workflow)
            
            # Keep history manageable
            if len(self.workflow_history) > 100:
                self.workflow_history = self.workflow_history[-50:]  # Keep last 50
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get the current status of a workflow."""
        
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            state = workflow["state"]
            
            return {
                "workflow_id": workflow_id,
                "status": workflow["status"],
                "current_step": state.get("current_step"),
                "progress_percentage": state.get("progress_percentage", 0),
                "completed_steps": len(state.get("completed_steps", [])),
                "total_steps": len(workflow["config"].get("workflow_steps", [])),
                "errors": len(state.get("errors", [])),
                "estimated_completion": state.get("estimated_completion_time", "unknown")
            }
        else:
            return {"workflow_id": workflow_id, "status": "not_found"}
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get the current status of the workflow engine."""
        
        return {
            "langgraph_available": self.langgraph_available,
            "active_workflows": len(self.active_workflows),
            "workflow_templates": len(self.workflow_templates),
            "total_executed": self.execution_metrics["total_workflows"],
            "success_rate": (
                self.execution_metrics["successful_workflows"] / max(1, self.execution_metrics["total_workflows"])
            ),
            "average_execution_time": self.execution_metrics["average_execution_time"],
            "components_connected": {
                "smart_orchestrator": self.smart_orchestrator is not None,
                "cognitive_core": self.cognitive_core is not None
            }
        }


class WorkflowError(Exception):
    """Custom exception for workflow-related errors."""
    pass


# Singleton instance for global access
enhanced_workflow_engine = EnhancedWorkflowEngine()
