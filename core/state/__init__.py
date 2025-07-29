"""
Project-S v2 Architecture - State Management Module
==================================================
Modern state management system providing persistent, reliable state
handling for workflows, sessions, conversations, and application context.

This module provides:
- Abstract state management interfaces
- File-based persistent state management
- Session lifecycle management
- Conversation tracking
- LangGraph checkpoint integration
- Context preservation across restarts
"""

from .base_state import (
    BaseStateManager,
    StateType,
    StateStatus,
    StateMetadata,
    StateEntry,
    StateOperation,
    SessionManager
)

from .persistent_manager import (
    PersistentStateManager,
    LANGGRAPH_AVAILABLE
)

__all__ = [
    # Base state classes
    "BaseStateManager",
    "StateType",
    "StateStatus", 
    "StateMetadata",
    "StateEntry",
    "StateOperation",
    "SessionManager",
    
    # Persistent implementation
    "PersistentStateManager",
    "LANGGRAPH_AVAILABLE"
]

# Create default persistent state manager instance
default_state_manager = PersistentStateManager()
