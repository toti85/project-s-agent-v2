"""
Project-S v2 Architecture - State Management Base
================================================
Modern state management framework providing persistent, reliable state
handling for workflows, sessions, conversations, and application context.

This module provides:
- Abstract state management interfaces
- State persistence and retrieval
- Session lifecycle management
- Transaction-safe state operations
- State event handling and notifications
"""

import asyncio
import logging
import uuid
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, AsyncContextManager
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class StateType(str, Enum):
    """Types of state that can be managed."""
    SESSION = "session"
    CONVERSATION = "conversation"
    WORKFLOW = "workflow"
    CHECKPOINT = "checkpoint"
    CONTEXT = "context"
    CACHE = "cache"

class StateStatus(str, Enum):
    """Status of state objects."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ARCHIVED = "archived"
    EXPIRED = "expired"

@dataclass
class StateMetadata:
    """Metadata for state objects."""
    id: str
    type: StateType
    status: StateStatus
    created_at: datetime
    updated_at: datetime
    version: int
    size_bytes: int
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateMetadata':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)

@dataclass
class StateEntry:
    """A complete state entry with data and metadata."""
    data: Dict[str, Any]
    metadata: StateMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "data": self.data,
            "metadata": self.metadata.to_dict()
        }
    
    @classmethod
    def from_dict(cls, entry_dict: Dict[str, Any]) -> 'StateEntry':
        """Create from dictionary."""
        return cls(
            data=entry_dict["data"],
            metadata=StateMetadata.from_dict(entry_dict["metadata"])
        )

class StateOperation:
    """Represents a state operation for transaction logging."""
    
    def __init__(self, operation: str, state_id: str, state_type: StateType, data: Any = None):
        self.id = str(uuid.uuid4())
        self.operation = operation  # create, update, delete, archive
        self.state_id = state_id
        self.state_type = state_type
        self.data = data
        self.timestamp = datetime.now()
        self.success = False
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "operation": self.operation,
            "state_id": self.state_id,
            "state_type": self.state_type.value,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "error": self.error
        }

class BaseStateManager(ABC):
    """
    Abstract base class for state management in Project-S v2.
    
    Provides common functionality for:
    - State persistence and retrieval
    - Session lifecycle management
    - Transaction safety
    - State event handling
    - Cleanup and archival
    """
    
    def __init__(self, name: str, storage_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the state manager.
        
        Args:
            name: Unique identifier for this state manager
            storage_path: Path for state storage
            config: Optional configuration dictionary
        """
        self.name = name
        self.config = config or {}
        self.storage_path = storage_path or Path.cwd() / "state"
        
        # Event handling
        self.event_handlers: Dict[str, List] = {}
        
        # Operation tracking
        self.operations_log: List[StateOperation] = []
        
        # Initialize storage
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized {self.__class__.__name__}: {name}")
    
    @abstractmethod
    async def create_state(self, 
                          state_type: StateType, 
                          data: Dict[str, Any], 
                          state_id: Optional[str] = None,
                          tags: Optional[List[str]] = None) -> str:
        """
        Create a new state entry.
        
        Args:
            state_type: Type of state to create
            data: State data
            state_id: Optional custom state ID
            tags: Optional tags for categorization
            
        Returns:
            str: The state ID
        """
        pass
    
    @abstractmethod
    async def get_state(self, state_id: str, state_type: Optional[StateType] = None) -> Optional[StateEntry]:
        """
        Retrieve a state entry by ID.
        
        Args:
            state_id: The state ID
            state_type: Optional state type for optimization
            
        Returns:
            StateEntry or None if not found
        """
        pass
    
    @abstractmethod
    async def update_state(self, state_id: str, data: Dict[str, Any], merge: bool = True) -> bool:
        """
        Update an existing state entry.
        
        Args:
            state_id: The state ID
            data: New data to update
            merge: Whether to merge with existing data or replace
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def delete_state(self, state_id: str, archive: bool = True) -> bool:
        """
        Delete or archive a state entry.
        
        Args:
            state_id: The state ID
            archive: Whether to archive instead of permanently delete
            
        Returns:
            bool: True if successful
        """
        pass
    
    @abstractmethod
    async def list_states(self, 
                         state_type: Optional[StateType] = None,
                         status: Optional[StateStatus] = None,
                         tags: Optional[List[str]] = None,
                         limit: Optional[int] = None) -> List[StateMetadata]:
        """
        List state entries with optional filtering.
        
        Args:
            state_type: Optional type filter
            status: Optional status filter
            tags: Optional tags filter
            limit: Optional limit on results
            
        Returns:
            List of state metadata
        """
        pass
    
    def subscribe_to_events(self, event_type: str, handler: callable) -> None:
        """
        Subscribe to state events.
        
        Args:
            event_type: Type of event to subscribe to
            handler: Event handler function
        """
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        
        self.event_handlers[event_type].append(handler)
        logger.debug(f"Subscribed to event: {event_type}")
    
    async def emit_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Emit a state event to all subscribers.
        
        Args:
            event_type: Type of event
            event_data: Event data
        """
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_data)
                    else:
                        handler(event_data)
                except Exception as e:
                    logger.error(f"Error in event handler for {event_type}: {e}")
    
    def log_operation(self, operation: StateOperation) -> None:
        """
        Log a state operation for auditing.
        
        Args:
            operation: The operation to log
        """
        self.operations_log.append(operation)
        
        # Keep only recent operations (configurable limit)
        max_operations = self.config.get("max_operations_log", 1000)
        if len(self.operations_log) > max_operations:
            self.operations_log = self.operations_log[-max_operations:]
    
    async def get_operation_history(self, 
                                  state_id: Optional[str] = None, 
                                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get operation history.
        
        Args:
            state_id: Optional state ID filter
            limit: Optional limit on results
            
        Returns:
            List of operation records
        """
        operations = self.operations_log
        
        if state_id:
            operations = [op for op in operations if op.state_id == state_id]
        
        if limit:
            operations = operations[-limit:]
        
        return [op.to_dict() for op in operations]
    
    @asynccontextmanager
    async def state_transaction(self, description: str = "State transaction"):
        """
        Context manager for state transactions.
        
        Args:
            description: Description of the transaction
        """
        transaction_id = str(uuid.uuid4())
        operations_before = len(self.operations_log)
        
        logger.debug(f"Starting state transaction: {transaction_id}")
        
        try:
            yield transaction_id
            logger.debug(f"State transaction completed: {transaction_id}")
        except Exception as e:
            # Rollback logic can be implemented here
            logger.error(f"State transaction failed: {transaction_id} - {e}")
            
            # For now, just log the failed operations
            failed_operations = self.operations_log[operations_before:]
            for op in failed_operations:
                op.success = False
                op.error = str(e)
            
            raise
    
    async def cleanup_expired_states(self, max_age_days: int = 30) -> int:
        """
        Clean up expired state entries.
        
        Args:
            max_age_days: Maximum age in days before cleanup
            
        Returns:
            int: Number of states cleaned up
        """
        cutoff_date = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)
        cleaned_count = 0
        
        try:
            # Get all states
            all_states = await self.list_states()
            
            for state_meta in all_states:
                if state_meta.updated_at.timestamp() < cutoff_date:
                    await self.delete_state(state_meta.id, archive=True)
                    cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} expired states")
            
        except Exception as e:
            logger.error(f"Error during state cleanup: {e}")
        
        return cleaned_count
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about this state manager."""
        return {
            "manager_name": self.name,
            "storage_path": str(self.storage_path),
            "operations_logged": len(self.operations_log),
            "event_types": list(self.event_handlers.keys()),
            "config": self.config
        }

class SessionManager:
    """
    Helper class for managing sessions within a state manager.
    """
    
    def __init__(self, state_manager: BaseStateManager):
        """
        Initialize session manager.
        
        Args:
            state_manager: The underlying state manager
        """
        self.state_manager = state_manager
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        
    async def create_session(self, initial_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new session.
        
        Args:
            initial_data: Optional initial session data
            
        Returns:
            str: Session ID
        """
        session_data = {
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "data": initial_data or {}
        }
        
        session_id = await self.state_manager.create_state(
            state_type=StateType.SESSION,
            data=session_data,
            tags=["session", "active"]
        )
        
        self.active_sessions[session_id] = session_data
        
        # Emit session created event
        await self.state_manager.emit_event("session.created", {
            "session_id": session_id,
            "data": session_data
        })
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session data or None
        """
        # Check active sessions first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        
        # Check persistent storage
        state_entry = await self.state_manager.get_state(session_id, StateType.SESSION)
        if state_entry:
            self.active_sessions[session_id] = state_entry.data
            return state_entry.data
        
        return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """
        Update session data.
        
        Args:
            session_id: Session ID
            data: Data to update
            
        Returns:
            bool: True if successful
        """
        success = await self.state_manager.update_state(session_id, data, merge=True)
        
        if success and session_id in self.active_sessions:
            self.active_sessions[session_id].update(data)
            
            # Emit session updated event
            await self.state_manager.emit_event("session.updated", {
                "session_id": session_id,
                "data": data
            })
        
        return success
    
    async def end_session(self, session_id: str) -> bool:
        """
        End a session.
        
        Args:
            session_id: Session ID
            
        Returns:
            bool: True if successful
        """
        # Update session status
        end_data = {
            "status": "ended",
            "ended_at": datetime.now().isoformat()
        }
        
        success = await self.state_manager.update_state(session_id, end_data, merge=True)
        
        if success:
            # Remove from active sessions
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
            
            # Emit session ended event
            await self.state_manager.emit_event("session.ended", {
                "session_id": session_id
            })
        
        return success
    
    @asynccontextmanager
    async def session_context(self, session_id: Optional[str] = None):
        """
        Context manager for working with sessions.
        
        Args:
            session_id: Optional existing session ID
            
        Yields:
            tuple: (session_id, session_data)
        """
        created_new = False
        
        if not session_id:
            session_id = await self.create_session()
            created_new = True
        
        session_data = await self.get_session(session_id)
        
        try:
            yield (session_id, session_data)
        finally:
            # Auto-cleanup empty sessions if we created them
            if created_new and session_data:
                if not session_data.get("data") or len(session_data["data"]) == 0:
                    await self.end_session(session_id)
