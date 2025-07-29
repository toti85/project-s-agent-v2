import time
import functools
from datetime import datetime
from typing import Callable, Any, TypeVar, cast
import logging
import traceback
import json
import os

# Type variable for the decorated function
F = TypeVar('F', bound=Callable[..., Any])

class EnhancedLogger:
    def __init__(self, name, log_dir="logs"):
        self.name = name
        self.log_dir = log_dir
        
        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)
        
        # File handler - regular logs
        log_file = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # Error file handler - only errors
        error_file = os.path.join(log_dir, f"{name}_errors.log")
        error_handler = logging.FileHandler(error_file)
        error_handler.setLevel(logging.ERROR)
        error_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        error_handler.setFormatter(error_format)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(error_handler)
        
    def debug(self, message):
        self.logger.debug(message)
        
    def info(self, message):
        self.logger.info(message)
        
    def warning(self, message):
        self.logger.warning(message)
        
    def error(self, message, exc_info=None):
        if (exc_info):
            self.logger.error(message, exc_info=exc_info)
            self._save_error_details(message, exc_info)
        else:
            self.logger.error(message)
            
    def critical(self, message, exc_info=None):
        if (exc_info):
            self.logger.critical(message, exc_info=exc_info)
            self._save_error_details(message, exc_info, critical=True)
        else:
            self.logger.critical(message)
            
    def _save_error_details(self, message, exc_info, critical=False):
        """Save detailed error information to a structured file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            level = "CRITICAL" if critical else "ERROR"
            error_file = os.path.join(self.log_dir, f"error_{level}_{timestamp}.json")
            
            error_type = exc_info[0].__name__ if exc_info and len(exc_info) > 0 else "Unknown"
            error_value = str(exc_info[1]) if exc_info and len(exc_info) > 1 else "No details"
            
            tb = traceback.format_exception(*exc_info) if exc_info else []
            
            error_data = {
                "timestamp": timestamp,
                "level": level,
                "component": self.name,
                "message": message,
                "error_type": error_type,
                "error_value": error_value,
                "traceback": tb,
                "context": {}  # Can be extended with system state information
            }
            
            with open(error_file, 'w') as f:
                json.dump(error_data, f, indent=2)
                
        except Exception as e:
            # Fallback to basic logging if error details saving fails
            self.logger.error(f"Failed to save error details: {str(e)}")

def monitor_performance(func: F) -> F:
    """
    Decorator to monitor the execution time of async functions.
    
    Measures and prints the execution time of the decorated async function.
    Can be applied to any async function in the Project-S system.
    
    Args:
        func (Callable): The async function to be decorated
        
    Returns:
        Callable: The wrapped function with performance monitoring
        
    Example:
        @monitor_performance
        async def some_slow_function(param1, param2):
            # function implementation
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Record start time
        start_time = time.time()
        
        # Get function details for better logging
        func_name = func.__name__
        module_name = func.__module__
        
        # Format current time for log readability
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"[{current_time}] Starting execution of {module_name}.{func_name}")
        
        try:
            # Execute the actual function
            result = await func(*args, **kwargs)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Print performance information
            print(f"[{current_time}] Completed {module_name}.{func_name} in {execution_time:.4f} seconds")
            
            return result
        except Exception as e:
            # Calculate execution time until exception
            execution_time = time.time() - start_time
            
            # Print performance information with exception
            print(f"[{current_time}] Error in {module_name}.{func_name} after {execution_time:.4f} seconds: {str(e)}")
            
            # Re-raise the exception
            raise
    
    # Cast is used to maintain the type signature for better IDE support
    return cast(F, wrapper)

# Example usage:
# 
# @monitor_performance
# async def generate_response(prompt: str) -> str:
#     # Some time-consuming operation
#     await asyncio.sleep(2)  # Simulate work
#     return "Generated response"
