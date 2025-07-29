import logging
import traceback
import sys
import json
import os
from typing import Any, Dict, Optional, Union
from datetime import datetime

class ErrorHandler:
    """
    Centralized error handler for the Project-S system.
    
    Provides a unified way to handle, log, and process errors throughout the system.
    All system components (executor, plugins, LLM clients) should use this for exception handling.
    """
    
    def __init__(self, logger=None):
        """
        Initialize the error handler with an optional custom logger.
        
        Args:
            logger (logging.Logger, optional): Custom logger instance.
                If not provided, a default 'project_s' logger will be created.
        """
        if logger:
            self.logger = logger
        else:
            # Create default logger
            self.logger = logging.getLogger('project_s')
            
            # Configure default logger if it has no handlers yet
            if not self.logger.handlers:
                self._setup_default_logger()
    
    def _setup_default_logger(self):
        """Set up the default logger with console and file handlers."""
        self.logger.setLevel(logging.INFO)
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
        console_handler.setFormatter(console_format)
        
        # Create file handler
        file_handler = logging.FileHandler('logs/system.log')
        file_handler.setLevel(logging.ERROR)  # Only log errors and above to file
        file_format = logging.Formatter('[%(asctime)s] [%(levelname)s] %(name)s - %(message)s')
        file_handler.setFormatter(file_format)
        
        # Add handlers to logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    async def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Handle an error with optional context information.
        
        Args:
            error (Exception): The exception to handle
            context (Dict[str, Any], optional): Additional context about where/how the error occurred
        
        Returns:
            Dict[str, Any]: Error information in a standardized format
        """
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "error_type": error.__class__.__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc()
        }
        
        # Add context if provided
        if context:
            error_info["context"] = context
            
            # Get component info if available
            component = context.get("component", "unknown")
            operation = context.get("operation", "unknown")
            
            # Log with context information
            self.logger.error(
                f"Error in {component} during {operation}: {error}",
                exc_info=True
            )
        else:
            # Simple error logging without context
            self.logger.error(f"Error: {error}", exc_info=True)
        
        # For debugging, print more details
        if self.logger.level <= logging.DEBUG:
            self.logger.debug(f"Error details: {json.dumps(error_info, indent=2, default=str)}")
            
        return {
            "status": "error",
            "error": error_info["error_type"],
            "message": error_info["error_message"],
            "details": error_info
        }
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log a warning message with optional context."""
        if context:
            self.logger.warning(f"{message} | Context: {json.dumps(context, default=str)}")
        else:
            self.logger.warning(message)
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None):
        """Log an info message with optional context."""
        if context:
            self.logger.info(f"{message} | Context: {json.dumps(context, default=str)}")
        else:
            self.logger.info(message)

# Create a singleton instance for global use
error_handler = ErrorHandler()
