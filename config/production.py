"""
Production Environment Configuration
==================================
"""

from .base_config import V2Config, Environment

class ProductionConfig(V2Config):
    """Production-specific configuration."""
    
    def __init__(self):
        super().__init__()
        self.environment = Environment.PRODUCTION
        self.debug = False
        self.log_level = "INFO"
        
        # Enable core features
        self.enable_langgraph = True
        self.enable_cognitive_core = True
        self.enable_state_management = True
        self.enable_tool_registry = True
        self.enable_monitoring = True
        
        # Enhanced security for production
        self.tool_security_enabled = True
        self.tool_sandbox_enabled = True
        
        # Optimized settings
        self.max_session_history = 500
        self.max_event_history = 5000
        self.performance_monitoring = True
