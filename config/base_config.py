"""
Project-S v2 Architecture - Base Configuration
=============================================

This module provides the foundational configuration management for the v2 architecture.
It supports environment-specific configurations and dependency injection.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

class Environment(str, Enum):
    """Supported environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

@dataclass
class V2Config:
    """
    Base configuration class for Project-S v2 architecture.
    """
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # Paths
    root_path: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    legacy_path: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    logs_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "logs")
    memory_path: Path = field(default_factory=lambda: Path(__file__).parent.parent / "memory")
    
    # Core Components
    enable_langgraph: bool = True
    enable_cognitive_core: bool = True
    enable_state_management: bool = True
    enable_tool_registry: bool = True
    enable_monitoring: bool = True
    
    # AI Models
    default_ai_model: str = "gpt-3.5-turbo"
    openrouter_api_key: Optional[str] = None
    enable_multi_model: bool = True
    
    # LangGraph Settings
    langgraph_checkpoint_enabled: bool = True
    langgraph_sqlite_path: Optional[Path] = None
    
    # State Management
    state_persistence_enabled: bool = True
    state_backup_enabled: bool = True
    max_session_history: int = 1000
    
    # Tool Registry
    max_tools: int = 50
    tool_security_enabled: bool = True
    tool_sandbox_enabled: bool = False
    
    # Monitoring
    diagnostics_enabled: bool = True
    dashboard_enabled: bool = True
    dashboard_port: int = 7777
    performance_monitoring: bool = True
    
    # Event Bus
    event_bus_enabled: bool = True
    max_event_history: int = 10000
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Post-initialization setup."""
        self._setup_paths()
        self._setup_logging()
        self._load_environment_overrides()
        
    def _setup_paths(self):
        """Ensure all required paths exist."""
        paths_to_create = [
            self.data_path,
            self.logs_path, 
            self.memory_path,
        ]
        
        for path in paths_to_create:
            path.mkdir(parents=True, exist_ok=True)
            
        # Set LangGraph SQLite path if not specified
        if self.langgraph_sqlite_path is None:
            self.langgraph_sqlite_path = self.memory_path / "langgraph_checkpoints.db"
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format=self.log_format,
            handlers=[
                logging.FileHandler(self.logs_path / "v2_architecture.log"),
                logging.StreamHandler()
            ]
        )
        
    def _load_environment_overrides(self):
        """Load environment-specific configuration overrides."""
        env_file = self.root_path / "config" / f"{self.environment.value}.json"
        
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    env_config = json.load(f)
                    
                # Apply overrides
                for key, value in env_config.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        logger.info(f"Applied environment override: {key} = {value}")
                        
            except Exception as e:
                logger.warning(f"Failed to load environment config {env_file}: {e}")
    
    def get_legacy_component_path(self, component_path: str) -> Path:
        """
        Get the path to a legacy component.
        
        Args:
            component_path: Relative path from legacy root (e.g., "core/cognitive_core.py")
            
        Returns:
            Full path to the legacy component
        """
        return self.legacy_path / component_path
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                config_dict[field_name] = str(field_value)
            elif isinstance(field_value, Enum):
                config_dict[field_name] = field_value.value
            else:
                config_dict[field_name] = field_value
        return config_dict

# Global configuration instance
_config: Optional[V2Config] = None

def get_config() -> V2Config:
    """
    Get the global configuration instance.
    
    Returns:
        V2Config: The global configuration instance
    """
    global _config
    if _config is None:
        _config = V2Config()
    return _config

def set_config(config: V2Config) -> None:
    """
    Set the global configuration instance.
    
    Args:
        config: New configuration instance
    """
    global _config
    _config = config

def reset_config() -> None:
    """Reset the global configuration to default."""
    global _config
    _config = None
