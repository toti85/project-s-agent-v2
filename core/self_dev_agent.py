#!/usr/bin/env python3
"""
Self-Development Agent - Project-S V2 Strategic Component
Autonomous tool discovery, implementation, and system improvement.
"""

import asyncio
import logging
import json
import ast
import importlib.util
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path
import hashlib

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from tools.tool_interface import BaseTool
from tools.registry.tool_registry_golden import ToolRegistry

logger = logging.getLogger(__name__)

class SelfDevelopmentAgent:
    """
    Autonomous agent for system self-improvement and tool development.
    Discovers missing capabilities and implements new tools dynamically.
    """
    
    def __init__(self, ai_client, tool_registry: ToolRegistry, config_dir: Path = None):
        """Initialize the self-development agent."""
        self.ai_client = ai_client
        self.tool_registry = tool_registry
        self.config_dir = Path(config_dir) if config_dir else Path("config")
        
        # Self-development tracking
        self.dev_log_dir = Path("logs/self_development")
        self.dev_log_dir.mkdir(parents=True, exist_ok=True)
        
        self.tools_dir = Path("tools/implementations/generated")
        self.tools_dir.mkdir(parents=True, exist_ok=True)
        
        # Track development history
        self.dev_registry_file = self.dev_log_dir / "self_dev_registry.md"
        self.strategic_decisions_file = self.dev_log_dir / "strategic_decisions.md"
        
        # Initialize registry files if they don't exist
        self._initialize_registry_files()
        
    def _initialize_registry_files(self):
        """Initialize the self-development registry files."""
        if not self.dev_registry_file.exists():
            registry_content = """# Self-Development Registry - Project-S V2

## Generated Tools

| Tool Name | Created | Purpose | Status | Location |
|-----------|---------|---------|--------|----------|

## Tool Implementation History

"""
            self.dev_registry_file.write_text(registry_content)
            
        if not self.strategic_decisions_file.exists():
            decisions_content = """# Strategic Decisions - Project-S V2

## Self-Improvement Planning

"""
            self.strategic_decisions_file.write_text(decisions_content)
    
    async def analyze_capability_gap(
        self, 
        failed_task_description: str,
        requested_tool: str,
        failure_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze why a task failed and identify missing capabilities.
        
        Returns:
        {
            "gap_type": "missing_tool" | "inadequate_tool" | "configuration_issue",
            "recommended_action": "implement_tool" | "enhance_tool" | "fix_config",
            "tool_specification": {...},
            "priority": "high" | "medium" | "low"
        }
        """
        
        logger.info(f"🔍 Analyzing capability gap for failed task: {failed_task_description}")
        
        analysis_prompt = f"""
Analyze this task failure and identify what capability is missing from Project-S:

FAILED TASK: {failed_task_description}
REQUESTED TOOL: {requested_tool}
FAILURE CONTEXT: {json.dumps(failure_context, indent=2)}

CURRENT AVAILABLE TOOLS:
{self._get_available_tools_summary()}

Please analyze:
1. What specific capability is missing?
2. Is this a missing tool, inadequate existing tool, or configuration issue?
3. What should be implemented to solve this problem?
4. How critical is this capability for Project-S operation?

Respond in JSON format:
{{
    "gap_type": "missing_tool|inadequate_tool|configuration_issue",
    "recommended_action": "implement_tool|enhance_tool|fix_config",
    "missing_capability": "detailed description",
    "tool_specification": {{
        "name": "proposed_tool_name",
        "purpose": "what this tool should do",
        "inputs": ["input1", "input2"],
        "outputs": "expected output format",
        "dependencies": ["required packages"],
        "complexity": "simple|medium|complex"
    }},
    "priority": "high|medium|low",
    "reasoning": "why this is needed and how it helps"
}}
"""
        
        try:
            response = await self.ai_client.generate_response(
                prompt=analysis_prompt,
                model="qwen3-235b",
                temperature=0.3
            )
            
            analysis = json.loads(response)
            
            # Log the analysis
            await self._log_strategic_decision(
                "capability_gap_analysis",
                {
                    "failed_task": failed_task_description,
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            logger.info(f"✅ Gap analysis complete: {analysis['gap_type']} - {analysis['recommended_action']}")
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Capability gap analysis failed: {e}")
            return {
                "gap_type": "unknown",
                "recommended_action": "manual_review",
                "error": str(e)
            }
    
    async def propose_tool_implementation(
        self, 
        tool_specification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a complete tool implementation based on specification.
        
        Returns:
        {
            "tool_code": "complete python code",
            "test_code": "test code",
            "success": bool,
            "tool_file_path": "path to generated tool"
        }
        """
        
        tool_name = tool_specification["name"]
        logger.info(f"🛠️ Proposing implementation for tool: {tool_name}")
        
        implementation_prompt = f"""
Create a complete, production-ready tool implementation for Project-S V2.

TOOL SPECIFICATION:
{json.dumps(tool_specification, indent=2)}

REQUIREMENTS:
1. Must inherit from BaseTool
2. Must be async and use proper error handling
3. Must include comprehensive docstrings
4. Must handle edge cases gracefully
5. Must follow Project-S V2 coding standards

TEMPLATE TO FOLLOW:
```python
#!/usr/bin/env python3
\"\"\"
{tool_name} - Automatically Generated Tool for Project-S V2
Generated by Self-Development Agent
\"\"\"

import asyncio
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path

from ..tool_interface import BaseTool

logger = logging.getLogger(__name__)

class {tool_name.replace('_', '').title()}Tool(BaseTool):
    \"\"\"
    {tool_specification.get('purpose', 'Auto-generated tool')}
    \"\"\"
    
    def __init__(self):
        super().__init__()
        self.name = "{tool_name}"
        self.description = "{tool_specification.get('purpose', 'Auto-generated tool')}"
        self.required_args = {tool_specification.get('inputs', [])}
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        \"\"\"Execute the tool with given parameters.\"\"\"
        try:
            # Implementation here
            pass
        except Exception as e:
            logger.error(f"❌ {self.name} execution failed: {{e}}")
            return {{"success": False, "error": str(e)}}
```

Please provide the COMPLETE implementation including:
1. Full class definition
2. All necessary imports
3. Proper error handling
4. Input validation
5. Return value formatting

Only return the Python code, no explanations.
"""
        
        try:
            tool_code = await self.ai_client.generate_response(
                prompt=implementation_prompt,
                model="qwen3-235b",
                temperature=0.2
            )
            
            # Clean up the code (remove markdown formatting if present)
            if "```python" in tool_code:
                tool_code = tool_code.split("```python")[1].split("```")[0].strip()
            elif "```" in tool_code:
                tool_code = tool_code.split("```")[1].strip()
            
            # Generate test code
            test_code = await self._generate_test_code(tool_name, tool_specification)
            
            # Save the tool implementation
            tool_file_path = self.tools_dir / f"{tool_name}.py"
            tool_file_path.write_text(tool_code)
            
            # Save test code
            test_file_path = self.tools_dir / f"test_{tool_name}.py"
            test_file_path.write_text(test_code)
            
            # Validate the generated code
            validation_result = await self._validate_tool_code(tool_file_path, tool_code)
            
            result = {
                "tool_code": tool_code,
                "test_code": test_code,
                "tool_file_path": str(tool_file_path),
                "test_file_path": str(test_file_path),
                "validation": validation_result,
                "success": validation_result["is_valid"]
            }
            
            # Log the implementation
            await self._log_tool_implementation(tool_name, tool_specification, result)
            
            logger.info(f"✅ Tool implementation proposed: {tool_name}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Tool implementation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool_code": None
            }
    
    async def register_new_tool(
        self, 
        tool_file_path: str, 
        tool_name: str
    ) -> Dict[str, Any]:
        """
        Dynamically load and register a newly implemented tool.
        """
        
        logger.info(f"📝 Registering new tool: {tool_name}")
        
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(tool_name, tool_file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find the tool class
            tool_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, BaseTool) and 
                    attr != BaseTool):
                    tool_class = attr
                    break
            
            if not tool_class:
                raise ValueError(f"No BaseTool subclass found in {tool_file_path}")
            
            # Instantiate and register the tool
            tool_instance = tool_class()
            registration_result = await self.tool_registry.register_tool(tool_instance)
            
            if registration_result.get("success"):
                # Update the registry file
                await self._update_dev_registry(
                    tool_name, 
                    tool_instance.description,
                    str(tool_file_path),
                    "registered"
                )
                
                logger.info(f"✅ Tool registered successfully: {tool_name}")
                return {
                    "success": True,
                    "tool_name": tool_name,
                    "tool_instance": tool_instance,
                    "registration_result": registration_result
                }
            else:
                raise ValueError(f"Tool registration failed: {registration_result}")
                
        except Exception as e:
            logger.error(f"❌ Tool registration failed: {e}")
            await self._update_dev_registry(
                tool_name,
                "Failed registration",
                str(tool_file_path),
                f"failed: {str(e)}"
            )
            return {
                "success": False,
                "error": str(e)
            }
    
    async def autonomous_improvement_cycle(
        self,
        failure_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Complete autonomous improvement cycle: analyze -> implement -> register.
        """
        
        logger.info("🚀 Starting autonomous improvement cycle")
        
        try:
            # 1. Analyze capability gap
            gap_analysis = await self.analyze_capability_gap(
                failure_context.get("task_description", "Unknown task"),
                failure_context.get("tool_name", "unknown"),
                failure_context
            )
            
            if gap_analysis.get("recommended_action") != "implement_tool":
                logger.info(f"⏭️ Not implementing tool: {gap_analysis.get('recommended_action')}")
                return gap_analysis
            
            # 2. Propose tool implementation
            tool_spec = gap_analysis.get("tool_specification", {})
            if not tool_spec:
                raise ValueError("No tool specification provided in gap analysis")
            
            implementation = await self.propose_tool_implementation(tool_spec)
            
            if not implementation.get("success"):
                raise ValueError(f"Tool implementation failed: {implementation.get('error')}")
            
            # 3. Register the new tool
            registration = await self.register_new_tool(
                implementation["tool_file_path"],
                tool_spec["name"]
            )
            
            if not registration.get("success"):
                raise ValueError(f"Tool registration failed: {registration.get('error')}")
            
            # 4. Return complete result
            result = {
                "success": True,
                "improvement_type": "new_tool_implemented",
                "tool_name": tool_spec["name"],
                "gap_analysis": gap_analysis,
                "implementation": implementation,
                "registration": registration,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"✅ Autonomous improvement cycle completed: {tool_spec['name']}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Autonomous improvement cycle failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _get_available_tools_summary(self) -> str:
        """Get a summary of currently available tools."""
        try:
            available_tools = self.tool_registry.list_tools()
            summary = "Available tools:\n"
            for tool_name, tool_info in available_tools.items():
                summary += f"- {tool_name}: {tool_info.get('description', 'No description')}\n"
            return summary
        except Exception:
            return "Unable to retrieve available tools"
    
    async def _generate_test_code(
        self, 
        tool_name: str, 
        tool_specification: Dict[str, Any]
    ) -> str:
        """Generate test code for a tool."""
        
        test_prompt = f"""
Generate comprehensive test code for this tool:

TOOL NAME: {tool_name}
SPECIFICATION: {json.dumps(tool_specification, indent=2)}

Requirements:
1. Use pytest framework
2. Test normal operation
3. Test error conditions
4. Test edge cases
5. Mock external dependencies

Only return the Python test code, no explanations.
"""
        
        try:
            test_code = await self.ai_client.generate_response(
                prompt=test_prompt,
                model="qwen3-235b",
                temperature=0.2
            )
            
            # Clean up the code
            if "```python" in test_code:
                test_code = test_code.split("```python")[1].split("```")[0].strip()
            elif "```" in test_code:
                test_code = test_code.split("```")[1].strip()
                
            return test_code
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to generate test code: {e}")
            return f'# Test code generation failed: {e}\n\ndef test_{tool_name}():\n    assert True  # Placeholder'
    
    async def _validate_tool_code(self, tool_file_path: Path, tool_code: str) -> Dict[str, Any]:
        """Validate the generated tool code."""
        
        try:
            # Parse the code to check for syntax errors
            ast.parse(tool_code)
            
            # Check for required components
            has_class = "class " in tool_code and "ToolInterface" in tool_code
            has_execute = "async def execute" in tool_code
            has_init = "def __init__" in tool_code
            
            validation_result = {
                "is_valid": True,
                "syntax_valid": True,
                "has_required_class": has_class,
                "has_execute_method": has_execute,
                "has_init_method": has_init,
                "issues": []
            }
            
            if not has_class:
                validation_result["issues"].append("Missing ToolInterface subclass")
                validation_result["is_valid"] = False
                
            if not has_execute:
                validation_result["issues"].append("Missing async execute method")
                validation_result["is_valid"] = False
                
            if not has_init:
                validation_result["issues"].append("Missing __init__ method")
                validation_result["is_valid"] = False
            
            return validation_result
            
        except SyntaxError as e:
            return {
                "is_valid": False,
                "syntax_valid": False,
                "syntax_error": str(e),
                "issues": [f"Syntax error: {e}"]
            }
        except Exception as e:
            return {
                "is_valid": False,
                "validation_error": str(e),
                "issues": [f"Validation failed: {e}"]
            }
    
    async def _log_tool_implementation(
        self, 
        tool_name: str, 
        specification: Dict[str, Any], 
        result: Dict[str, Any]
    ) -> None:
        """Log tool implementation details."""
        
        try:
            log_entry = {
                "tool_name": tool_name,
                "specification": specification,
                "implementation_result": result,
                "timestamp": datetime.now().isoformat(),
                "implementation_id": hashlib.md5(f"{tool_name}_{datetime.now()}".encode()).hexdigest()[:8]
            }
            
            log_file = self.dev_log_dir / f"implementation_{tool_name}_{log_entry['implementation_id']}.json"
            log_file.write_text(json.dumps(log_entry, indent=2, default=str))
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log tool implementation: {e}")
    
    async def _update_dev_registry(
        self, 
        tool_name: str, 
        purpose: str, 
        location: str, 
        status: str
    ) -> None:
        """Update the self-development registry."""
        
        try:
            registry_content = self.dev_registry_file.read_text()
            
            # Add new entry to the table
            new_entry = f"| {tool_name} | {datetime.now().strftime('%Y-%m-%d %H:%M')} | {purpose} | {status} | {location} |\n"
            
            # Insert after the header
            lines = registry_content.split('\n')
            header_end = -1
            for i, line in enumerate(lines):
                if line.startswith('|--------'):
                    header_end = i
                    break
            
            if header_end >= 0:
                lines.insert(header_end + 1, new_entry.rstrip())
                self.dev_registry_file.write_text('\n'.join(lines))
            else:
                # Append if header not found
                self.dev_registry_file.write_text(registry_content + new_entry)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to update dev registry: {e}")
    
    async def _log_strategic_decision(
        self, 
        decision_type: str, 
        decision_data: Dict[str, Any]
    ) -> None:
        """Log strategic decisions for improvement planning."""
        
        try:
            decisions_content = self.strategic_decisions_file.read_text()
            
            new_decision = f"""
## {decision_type.replace('_', ' ').title()} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

```json
{json.dumps(decision_data, indent=2, default=str)}
```

"""
            
            self.strategic_decisions_file.write_text(decisions_content + new_decision)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to log strategic decision: {e}")


# Convenience function for triggering self-improvement
async def trigger_self_improvement(
    ai_client,
    tool_registry: ToolRegistry,
    failure_context: Dict[str, Any]
) -> Dict[str, Any]:
    """Convenience function for triggering autonomous self-improvement."""
    agent = SelfDevelopmentAgent(ai_client, tool_registry)
    return await agent.autonomous_improvement_cycle(failure_context)
