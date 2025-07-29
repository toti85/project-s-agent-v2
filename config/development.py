"""
Development Environment Configuration
===================================
"""

from .base_config import V2Config, Environment

class DevelopmentConfig(V2Config):
    """Development-specific configuration."""
    
    def __init__(self):
        super().__init__()
        self.environment = Environment.DEVELOPMENT
        self.debug = True
        self.log_level = "DEBUG"
        
        # Enable all features for development
        self.enable_langgraph = True
        self.enable_cognitive_core = True
        self.enable_state_management = True
        self.enable_tool_registry = True
        self.enable_monitoring = True
        
        # Relaxed security for development
        self.tool_security_enabled = False
        self.tool_sandbox_enabled = False
        
        # Enhanced debugging
        self.performance_monitoring = True
        self.diagnostics_enabled = True
        self.dashboard_enabled = True
