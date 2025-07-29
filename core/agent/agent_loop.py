#!/usr/bin/env python3
"""
Project-S V2 Autonomous Agent Loop
==================================

Autonomous background agent that monitors for tasks and triggers 
full AI → TOOL → EVALUATION cycles automatically.

Features:
- File watcher monitoring tasks/inbox/ directory
- Full AI cognitive processing pipeline
- Tool orchestration and execution
- Comprehensive evaluation and feedback
- Automated task processing with error handling
- Graceful shutdown and logging

Author: Project-S V2 Architecture
Date: July 2, 2025
"""
import os
import sys
import time
import logging
import asyncio
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import Project-S V2 components
from core.cognitive.cognitive_core import CognitiveCore
from tools.registry.smart_orchestrator import SmartToolOrchestrator
from core.evaluator import Evaluator
from integrations.ai_models.multi_model_ai_client import AIClient

class AgentLoop:
    """
    Autonomous Agent Loop for Project-S V2
    
    Monitors inbox directory for new tasks and automatically processes them
    through the complete AI → Tool → Evaluation pipeline.
    """
    
    def __init__(self):
        """Initialize the autonomous agent loop system."""
        
        # Setup logging
        self._setup_logging()
        
        # Initialize directories
        self.project_root = project_root
        self.tasks_dir = self.project_root / "tasks"
        self.inbox_dir = self.tasks_dir / "inbox"
        self.outbox_dir = self.tasks_dir / "outbox"
        self.processed_dir = self.tasks_dir / "processed"
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Initialize Project-S V2 components
        self.logger.info("[INIT] Initializing Project-S V2 components...")
        self._initialize_components()
        
        # Track processed files to avoid reprocessing
        self.processed_files = set()
        
        # Running state
        self.is_running = False
        
        self.logger.info("[OK] Autonomous Agent Loop initialized successfully!")
    
    def _setup_logging(self):
        """Setup comprehensive logging for the agent loop."""
        
        # Create logs directory
        logs_dir = project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Setup main logger
        self.logger = logging.getLogger("AgentLoop")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create formatters
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler for general logging
        log_file = logs_dir / "agent_loop.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for real-time monitoring with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        # Set encoding to handle Unicode characters
        if hasattr(console_handler.stream, 'reconfigure'):
            try:
                console_handler.stream.reconfigure(encoding='utf-8', errors='replace')
            except:
                pass  # Fallback if reconfigure fails
        self.logger.addHandler(console_handler)
        
        # Setup failure logger
        self.failure_logger = logging.getLogger("AgentFailures")
        self.failure_logger.setLevel(logging.ERROR)
        
        failure_file = logs_dir / "agent_failures.log"
        failure_handler = logging.FileHandler(failure_file)
        failure_handler.setLevel(logging.ERROR)
        failure_handler.setFormatter(formatter)
        self.failure_logger.addHandler(failure_handler)
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        
        directories = [
            self.tasks_dir,
            self.inbox_dir,
            self.outbox_dir,
            self.processed_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.info(f"[DIR] Directory ready: {directory}")
    
    def _initialize_components(self):
        """Initialize all Project-S V2 components."""
        
        try:
            # Initialize AI Client
            self.ai_client = AIClient()
            self.logger.info("[OK] AI Client initialized")
            
            # Initialize Cognitive Core
            self.cognitive_core = CognitiveCore()
            self.logger.info("[OK] Cognitive Core initialized")
            
            # Initialize Smart Tool Orchestrator
            self.tool_orchestrator = SmartToolOrchestrator()
            self.logger.info("[OK] Smart Tool Orchestrator initialized")
            
            # Initialize Evaluator
            self.evaluator = Evaluator(self.ai_client)
            self.logger.info("[OK] Evaluator initialized")
            
            self.logger.info("[READY] All Project-S V2 components ready!")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize components: {e}")
            raise
    
    def scan_inbox(self) -> List[Path]:
        """
        Scan the inbox directory for new task files.
        
        Returns:
            List of new .txt files to process
        """
        
        try:
            new_files = []
            
            if not self.inbox_dir.exists():
                return new_files
            
            # Get all .txt files in inbox
            for file_path in self.inbox_dir.glob("*.txt"):
                if file_path.name not in self.processed_files:
                    new_files.append(file_path)
                    self.processed_files.add(file_path.name)
            
            return new_files
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error scanning inbox: {e}")
            return []
    
    async def process_task_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Process a single task file through the complete AI pipeline.
        
        Args:
            file_path: Path to the task file
            
        Returns:
            Processing result dictionary
        """
        
        task_name = file_path.stem
        self.logger.info(f"[TARGET] Processing task: {task_name}")
        
        try:
            # Read task content
            task_content = file_path.read_text(encoding='utf-8').strip()
            self.logger.info(f"[CONTENT] Task content: {task_content[:100]}...")
            
            # PHASE 1: Cognitive Processing
            self.logger.info("[PHASE 1] COGNITIVE: Cognitive analysis...")
            cognitive_result = await self._cognitive_analysis(task_content)
            
            # PHASE 2: Tool Orchestration
            self.logger.info("[PHASE 2] ORCHESTRATION: Tool orchestration...")
            tool_result = await self._tool_orchestration(cognitive_result, task_content)
            
            # PHASE 3: Evaluation
            self.logger.info("[PHASE 3] EVALUATION: Result evaluation...")
            evaluation_result = await self._evaluate_result(tool_result, task_content)
            
            # Compile complete result
            complete_result = {
                "task_name": task_name,
                "task_content": task_content,
                "cognitive_result": cognitive_result,
                "tool_result": tool_result,
                "evaluation_result": evaluation_result,
                "status": "SUCCESS",
                "timestamp": datetime.now().isoformat()
            }
            
            # Write result to outbox
            await self._write_result(task_name, complete_result)
            
            # Move processed file
            self._move_to_processed(file_path)
            
            self.logger.info(f"[SUCCESS] Task completed successfully: {task_name}")
            return complete_result
            
        except Exception as e:
            self.logger.error(f"[ERROR] Task processing failed: {task_name} - {e}")
            self.failure_logger.error(f"Task: {task_name}, Error: {e}, Traceback: {traceback.format_exc()}")
            
            # Create failure result
            failure_result = {
                "task_name": task_name,
                "task_content": task_content if 'task_content' in locals() else "Unknown",
                "status": "FAILED",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            # Write failure result
            await self._write_result(task_name, failure_result, is_failure=True)
            
            # Move failed file to processed (with error marker)
            self._move_to_processed(file_path, failed=True)
            
            return failure_result
    
    async def _cognitive_analysis(self, task_content: str) -> Dict[str, Any]:
        """Run cognitive analysis on the task."""
        
        try:
            # Use cognitive core to parse and understand the task
            analysis = await self.cognitive_core.parse_task(task_content)
            
            # Enhanced analysis with AI client
            ai_analysis = await self.ai_client.generate_response(
                prompt=f"""
                Analyze this task for autonomous processing:
                
                Task: {task_content}
                
                Provide analysis in JSON format:
                {{
                    "task_type": "programming|system|analysis|creative|other",
                    "complexity": "simple|moderate|complex",
                    "required_tools": ["tool1", "tool2"],
                    "expected_outcome": "description",
                    "processing_strategy": "step by step approach"
                }}
                """,
                model=None,
                task_type="analysis",
                temperature=0.3
            )
            
            return {
                "cognitive_core_result": analysis,
                "ai_analysis": ai_analysis,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Cognitive analysis failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _tool_orchestration(self, cognitive_result: Dict[str, Any], task_content: str) -> Dict[str, Any]:
        """Orchestrate tool selection and execution."""
        
        try:
            # Use smart orchestrator to find and execute appropriate tools
            orchestration_result = await self.tool_orchestrator.orchestrate_task(
                task_description=task_content,
                cognitive_analysis=cognitive_result
            )
            
            return {
                "orchestration_result": orchestration_result,
                "tools_used": orchestration_result.get("tools_used", []),
                "execution_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Tool orchestration failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _evaluate_result(self, tool_result: Dict[str, Any], original_task: str) -> Dict[str, Any]:
        """Evaluate the execution result."""
        
        try:
            # Use evaluator to assess the result
            evaluation = await self.evaluator.evaluate_result(
                result=tool_result,
                expected_outcome=original_task
            )
            
            return {
                "evaluation": evaluation,
                "success": evaluation.get("success", False),
                "score": evaluation.get("score", 0),
                "recommendations": evaluation.get("recommendations", []),
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"[ERROR] Evaluation failed: {e}")
            return {"error": str(e), "status": "failed"}
    
    async def _write_result(self, task_name: str, result: Dict[str, Any], is_failure: bool = False):
        """Write processing result to outbox."""
        
        try:
            # Create result filename
            suffix = "_FAILED" if is_failure else "_result"
            result_filename = f"{task_name}{suffix}.txt"
            result_path = self.outbox_dir / result_filename
            
            # Format result for output
            output_content = self._format_result_output(result)
            
            # Write to file
            result_path.write_text(output_content, encoding='utf-8')
            
            self.logger.info(f"[OUTPUT] Result written: {result_filename}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to write result: {e}")
    
    def _format_result_output(self, result: Dict[str, Any]) -> str:
        """Format the result for human-readable output."""
        
        output = f"""
# AUTONOMOUS AGENT PROCESSING RESULT

**Task:** {result.get('task_name', 'Unknown')}
**Status:** {result.get('status', 'Unknown')}
**Timestamp:** {result.get('timestamp', 'Unknown')}

## ORIGINAL TASK
{result.get('task_content', 'N/A')}

"""
        
        if result.get('status') == 'SUCCESS':
            output += f"""
## COGNITIVE ANALYSIS
{self._format_dict_section(result.get('cognitive_result', {}))}

## TOOL EXECUTION
{self._format_dict_section(result.get('tool_result', {}))}

## EVALUATION
{self._format_dict_section(result.get('evaluation_result', {}))}

## SUMMARY
- Success: {result.get('evaluation_result', {}).get('success', 'Unknown')}
- Score: {result.get('evaluation_result', {}).get('score', 'N/A')}
- Tools Used: {', '.join(result.get('tool_result', {}).get('tools_used', []))}
"""
        else:
            output += f"""
## ERROR DETAILS
{result.get('error', 'Unknown error occurred')}

## RECOMMENDATIONS
- Check task format and content
- Verify system components are properly initialized
- Review logs for detailed error information
"""
        
        return output
    
    def _format_dict_section(self, data: Dict[str, Any]) -> str:
        """Format a dictionary section for output."""
        
        if not data:
            return "No data available"
        
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"**{key}:**")
                for sub_key, sub_value in value.items():
                    lines.append(f"  - {sub_key}: {sub_value}")
            elif isinstance(value, list):
                lines.append(f"**{key}:** {', '.join(map(str, value))}")
            else:
                lines.append(f"**{key}:** {value}")
        
        return '\n'.join(lines)
    
    def _move_to_processed(self, file_path: Path, failed: bool = False):
        """Move processed file to processed directory."""
        
        try:
            # Create new filename with status
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            status = "FAILED" if failed else "SUCCESS"
            new_name = f"{file_path.stem}_{status}_{timestamp}.txt"
            new_path = self.processed_dir / new_name
            
            # Move file
            file_path.rename(new_path)
            self.logger.info(f"[MOVED] File moved to processed: {new_name}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to move file: {e}")
    
    async def run_cycle(self):
        """Run a single monitoring and processing cycle."""
        
        # Scan for new files
        new_files = self.scan_inbox()
        
        if new_files:
            self.logger.info(f"[FOUND] Found {len(new_files)} new task(s)")
            
            # Process each file
            for file_path in new_files:
                await self.process_task_file(file_path)
        
        # Small delay between files to prevent overwhelming
        if new_files:
            await asyncio.sleep(1)
    
    async def run(self):
        """
        Main agent loop - runs continuously until interrupted.
        """
        
        self.logger.info("[START] Starting Autonomous Agent Loop...")
        self.logger.info(f"[MONITOR] Monitoring directory: {self.inbox_dir}")
        self.logger.info("[CONTROL] Press Ctrl+C to stop gracefully")
        
        self.is_running = True
        
        try:
            while self.is_running:
                # Run processing cycle
                await self.run_cycle()
                
                # Wait before next cycle (5 seconds as specified)
                await asyncio.sleep(5)
                
        except KeyboardInterrupt:
            self.logger.info("[SIGNAL] Shutdown signal received...")
            await self.shutdown()
            
        except Exception as e:
            self.logger.error(f"[CRASH] Unexpected error in main loop: {e}")
            self.failure_logger.error(f"Main loop error: {e}, Traceback: {traceback.format_exc()}")
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown of the agent loop."""
        
        self.logger.info("[SHUTDOWN] Shutting down Autonomous Agent Loop...")
        self.is_running = False
        
        # Cleanup any pending operations
        self.logger.info("[CLEANUP] Cleanup completed")
        self.logger.info("[STOPPED] Autonomous Agent Loop stopped")

def main():
    """Main entry point for the autonomous agent loop."""
    
    print("[PROJECT-S] PROJECT-S V2 AUTONOMOUS AGENT LOOP")
    print("=" * 50)
    print("[MONITOR] Monitoring for tasks and executing AI cycles...")
    print("[INBOX] Drop .txt files in tasks/inbox/ to trigger processing")
    print("[OUTBOX] Results will appear in tasks/outbox/")
    print("[CONTROL] Press Ctrl+C to stop")
    print("=" * 50)
    
    try:
        # Create and run agent loop
        agent = AgentLoop()
        asyncio.run(agent.run())
        
    except KeyboardInterrupt:
        print("\n[STOPPED] Agent loop stopped by user")
    except Exception as e:
        print(f"\n[ERROR] Agent loop crashed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
