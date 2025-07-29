"""
Project-S v2 Architecture - Persistent State Manager
===================================================
File-based persistent state management implementation with support for
sessions, conversations, workflow checkpoints, and context preservation.

This module provides:
- JSON-based state persistence
- LangGraph checkpoint integration
- Conversation history management
- Session lifecycle with cleanup
- Context preservation across restarts
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import aiofiles

from .base_state import (
    BaseStateManager,
    StateType,
    StateStatus,
    StateMetadata,
    StateEntry,
    StateOperation,
    SessionManager
)

logger = logging.getLogger(__name__)

# LangGraph integration
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger.warning("LangGraph not available - checkpoint functionality will be limited")

class PersistentStateManager(BaseStateManager):
    """
    File-based persistent state manager with comprehensive state handling.
    
    Features:
    - JSON-based persistence
    - Session management
    - Conversation tracking
    - LangGraph checkpoint integration
    - Context preservation
    - Automatic cleanup and archival
    """
    
    def __init__(self, name: str = "persistent_state", storage_path: Optional[Path] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the persistent state manager.
        
        Args:
            name: Manager identifier
            storage_path: Path for state storage
            config: Configuration dictionary
        """
        # Set default storage path
        if storage_path is None:
            storage_path = Path.cwd() / "memory" / "state"
        
        super().__init__(name, storage_path, config)
        
        # Create storage directories
        self.sessions_path = self.storage_path / "sessions"
        self.conversations_path = self.storage_path / "conversations"
        self.checkpoints_path = self.storage_path / "checkpoints"
        self.context_path = self.storage_path / "context"
        self.archive_path = self.storage_path / "archive"
        
        for path in [self.sessions_path, self.conversations_path, self.checkpoints_path, self.context_path, self.archive_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        # Initialize session manager
        self.session_manager = SessionManager(self)
        
        # Initialize LangGraph checkpoint saver if available
        self.checkpoint_saver: Optional[Any] = None
        if LANGGRAPH_AVAILABLE:
            checkpoint_db = self.checkpoints_path / "langgraph_checkpoints.db"
            self.checkpoint_saver = SqliteSaver(str(checkpoint_db))
            logger.info("LangGraph checkpoint saver initialized")
        
        # Load existing state metadata
        self._metadata_cache: Dict[str, StateMetadata] = {}
        asyncio.create_task(self._load_metadata_cache())
        
        logger.info(f"Persistent state manager '{name}' initialized at {storage_path}")
    
    async def _load_metadata_cache(self) -> None:
        """Load metadata cache from disk."""
        try:
            metadata_file = self.storage_path / "metadata_cache.json"
            if metadata_file.exists():
                async with aiofiles.open(metadata_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    metadata_dict = json.loads(content)
                    
                    for state_id, meta_data in metadata_dict.items():
                        self._metadata_cache[state_id] = StateMetadata.from_dict(meta_data)
                
                logger.info(f"Loaded {len(self._metadata_cache)} metadata entries from cache")
        except Exception as e:
            logger.error(f"Error loading metadata cache: {e}")
    
    async def _save_metadata_cache(self) -> None:
        """Save metadata cache to disk."""
        try:
            metadata_file = self.storage_path / "metadata_cache.json"
            metadata_dict = {
                state_id: metadata.to_dict() 
                for state_id, metadata in self._metadata_cache.items()
            }
            
            async with aiofiles.open(metadata_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(metadata_dict, indent=2))
                
        except Exception as e:
            logger.error(f"Error saving metadata cache: {e}")
    
    def _get_storage_path(self, state_type: StateType) -> Path:
        """Get storage path for a specific state type."""
        path_map = {
            StateType.SESSION: self.sessions_path,
            StateType.CONVERSATION: self.conversations_path,
            StateType.WORKFLOW: self.checkpoints_path,
            StateType.CHECKPOINT: self.checkpoints_path,
            StateType.CONTEXT: self.context_path,
            StateType.CACHE: self.storage_path / "cache"
        }
        
        return path_map.get(state_type, self.storage_path)
    
    def _create_metadata(self, 
                        state_id: str, 
                        state_type: StateType, 
                        data_size: int,
                        tags: Optional[List[str]] = None) -> StateMetadata:
        """Create metadata for a state entry."""
        now = datetime.now()
        
        return StateMetadata(
            id=state_id,
            type=state_type,
            status=StateStatus.ACTIVE,
            created_at=now,
            updated_at=now,
            version=1,
            size_bytes=data_size,
            tags=tags or []
        )
    
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
        if state_id is None:
            state_id = str(uuid.uuid4())
        
        operation = StateOperation("create", state_id, state_type, data)
        
        try:
            # Serialize data
            data_json = json.dumps(data, indent=2)
            data_size = len(data_json.encode('utf-8'))
            
            # Create metadata
            metadata = self._create_metadata(state_id, state_type, data_size, tags)
            
            # Create state entry
            state_entry = StateEntry(data=data, metadata=metadata)
            
            # Determine storage path
            storage_path = self._get_storage_path(state_type)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            # Save to file
            file_path = storage_path / f"{state_id}.json"
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(state_entry.to_dict(), indent=2))
            
            # Update metadata cache
            self._metadata_cache[state_id] = metadata
            await self._save_metadata_cache()
            
            # Log operation
            operation.success = True
            self.log_operation(operation)
            
            # Emit event
            await self.emit_event("state.created", {
                "state_id": state_id,
                "state_type": state_type.value,
                "tags": tags
            })
            
            logger.debug(f"Created state: {state_id} (type: {state_type})")
            return state_id
            
        except Exception as e:
            operation.success = False
            operation.error = str(e)
            self.log_operation(operation)
            
            logger.error(f"Error creating state {state_id}: {e}")
            raise
    
    async def get_state(self, state_id: str, state_type: Optional[StateType] = None) -> Optional[StateEntry]:
        """
        Retrieve a state entry by ID.
        
        Args:
            state_id: The state ID
            state_type: Optional state type for optimization
            
        Returns:
            StateEntry or None if not found
        """
        try:
            # Check metadata cache first
            if state_id in self._metadata_cache:
                metadata = self._metadata_cache[state_id]
                storage_path = self._get_storage_path(metadata.type)
            elif state_type:
                storage_path = self._get_storage_path(state_type)
            else:
                # Search all possible locations
                for search_type in StateType:
                    search_path = self._get_storage_path(search_type)
                    file_path = search_path / f"{state_id}.json"
                    if file_path.exists():
                        storage_path = search_path
                        break
                else:
                    return None
            
            # Load from file
            file_path = storage_path / f"{state_id}.json"
            if not file_path.exists():
                return None
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                entry_dict = json.loads(content)
            
            state_entry = StateEntry.from_dict(entry_dict)
            
            # Update metadata cache if not present
            if state_id not in self._metadata_cache:
                self._metadata_cache[state_id] = state_entry.metadata
                await self._save_metadata_cache()
            
            logger.debug(f"Retrieved state: {state_id}")
            return state_entry
            
        except Exception as e:
            logger.error(f"Error retrieving state {state_id}: {e}")
            return None
    
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
        operation = StateOperation("update", state_id, StateType.SESSION, data)  # Default type
        
        try:
            # Get existing state
            state_entry = await self.get_state(state_id)
            if not state_entry:
                logger.warning(f"Attempted to update non-existent state: {state_id}")
                return False
            
            # Update data
            if merge:
                if isinstance(state_entry.data, dict) and isinstance(data, dict):
                    state_entry.data.update(data)
                else:
                    state_entry.data = data
            else:
                state_entry.data = data
            
            # Update metadata
            now = datetime.now()
            state_entry.metadata.updated_at = now
            state_entry.metadata.version += 1
            
            # Recalculate size
            data_json = json.dumps(state_entry.data, indent=2)
            state_entry.metadata.size_bytes = len(data_json.encode('utf-8'))
            
            # Save updated state
            storage_path = self._get_storage_path(state_entry.metadata.type)
            file_path = storage_path / f"{state_id}.json"
            
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(state_entry.to_dict(), indent=2))
            
            # Update metadata cache
            self._metadata_cache[state_id] = state_entry.metadata
            await self._save_metadata_cache()
            
            # Log operation
            operation.success = True
            self.log_operation(operation)
            
            # Emit event
            await self.emit_event("state.updated", {
                "state_id": state_id,
                "state_type": state_entry.metadata.type.value,
                "merge": merge
            })
            
            logger.debug(f"Updated state: {state_id}")
            return True
            
        except Exception as e:
            operation.success = False
            operation.error = str(e)
            self.log_operation(operation)
            
            logger.error(f"Error updating state {state_id}: {e}")
            return False
    
    async def delete_state(self, state_id: str, archive: bool = True) -> bool:
        """
        Delete or archive a state entry.
        
        Args:
            state_id: The state ID
            archive: Whether to archive instead of permanently delete
            
        Returns:
            bool: True if successful
        """
        operation = StateOperation("delete" if not archive else "archive", state_id, StateType.SESSION)
        
        try:
            # Get existing state
            state_entry = await self.get_state(state_id)
            if not state_entry:
                logger.warning(f"Attempted to delete non-existent state: {state_id}")
                return False
            
            storage_path = self._get_storage_path(state_entry.metadata.type)
            file_path = storage_path / f"{state_id}.json"
            
            if archive:
                # Move to archive
                archive_type_path = self.archive_path / state_entry.metadata.type.value
                archive_type_path.mkdir(parents=True, exist_ok=True)
                
                archive_file_path = archive_type_path / f"{state_id}.json"
                
                # Update metadata status
                state_entry.metadata.status = StateStatus.ARCHIVED
                state_entry.metadata.updated_at = datetime.now()
                
                # Save to archive
                async with aiofiles.open(archive_file_path, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(state_entry.to_dict(), indent=2))
                
                # Remove original file
                if file_path.exists():
                    os.remove(file_path)
                
                # Update metadata cache
                self._metadata_cache[state_id] = state_entry.metadata
                
            else:
                # Permanent deletion
                if file_path.exists():
                    os.remove(file_path)
                
                # Remove from metadata cache
                if state_id in self._metadata_cache:
                    del self._metadata_cache[state_id]
            
            await self._save_metadata_cache()
            
            # Log operation
            operation.success = True
            self.log_operation(operation)
            
            # Emit event
            await self.emit_event("state.deleted" if not archive else "state.archived", {
                "state_id": state_id,
                "state_type": state_entry.metadata.type.value
            })
            
            logger.debug(f"{'Archived' if archive else 'Deleted'} state: {state_id}")
            return True
            
        except Exception as e:
            operation.success = False
            operation.error = str(e)
            self.log_operation(operation)
            
            logger.error(f"Error {'archiving' if archive else 'deleting'} state {state_id}: {e}")
            return False
    
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
        try:
            results = []
            
            for state_id, metadata in self._metadata_cache.items():
                # Apply filters
                if state_type and metadata.type != state_type:
                    continue
                
                if status and metadata.status != status:
                    continue
                
                if tags:
                    if not any(tag in metadata.tags for tag in tags):
                        continue
                
                results.append(metadata)
            
            # Sort by updated_at (most recent first)
            results.sort(key=lambda x: x.updated_at, reverse=True)
            
            # Apply limit
            if limit and limit > 0:
                results = results[:limit]
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing states: {e}")
            return []
    
    # Specialized methods for specific state types
    
    async def add_conversation_entry(self, 
                                   session_id: str, 
                                   role: str, 
                                   content: str, 
                                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a conversation entry to a session.
        
        Args:
            session_id: Session ID
            role: Message role (user, assistant, system)
            content: Message content
            metadata: Optional message metadata
            
        Returns:
            str: Entry ID
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        conversation_entry = {
            "id": entry_id,
            "timestamp": timestamp,
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        # Get or create conversation history
        conversation_state = await self.get_state(f"conv_{session_id}", StateType.CONVERSATION)
        
        if conversation_state:
            conversation_data = conversation_state.data
        else:
            conversation_data = {"entries": []}
        
        # Add new entry
        conversation_data["entries"].append(conversation_entry)
        
        # Save conversation
        if conversation_state:
            await self.update_state(f"conv_{session_id}", conversation_data)
        else:
            await self.create_state(
                StateType.CONVERSATION, 
                conversation_data, 
                f"conv_{session_id}",
                tags=["conversation", session_id]
            )
        
        logger.debug(f"Added conversation entry to session {session_id}")
        return entry_id
    
    async def get_conversation_history(self, 
                                     session_id: str, 
                                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session ID
            limit: Optional limit on entries
            
        Returns:
            List of conversation entries
        """
        conversation_state = await self.get_state(f"conv_{session_id}", StateType.CONVERSATION)
        
        if not conversation_state:
            return []
        
        entries = conversation_state.data.get("entries", [])
        
        if limit and limit > 0:
            entries = entries[-limit:]
        
        return entries
    
    async def save_langgraph_checkpoint(self, 
                                      thread_id: str, 
                                      config_id: str, 
                                      checkpoint_data: Dict[str, Any]) -> bool:
        """
        Save a LangGraph workflow checkpoint.
        
        Args:
            thread_id: Thread/session ID
            config_id: Configuration ID
            checkpoint_data: Checkpoint data
            
        Returns:
            bool: True if successful
        """
        try:
            if LANGGRAPH_AVAILABLE and self.checkpoint_saver:
                # Use LangGraph's native checkpoint saver
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.checkpoint_saver.put({
                        "thread_id": thread_id,
                        "config_id": config_id,
                        "state": checkpoint_data
                    })
                )
                logger.debug(f"Saved LangGraph checkpoint: {thread_id}/{config_id}")
            else:
                # Fallback to our state system
                checkpoint_id = f"checkpoint_{thread_id}_{config_id}"
                await self.create_state(
                    StateType.CHECKPOINT,
                    {
                        "thread_id": thread_id,
                        "config_id": config_id,
                        "data": checkpoint_data
                    },
                    checkpoint_id,
                    tags=["checkpoint", "langgraph", thread_id]
                )
                logger.debug(f"Saved fallback checkpoint: {checkpoint_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving LangGraph checkpoint: {e}")
            return False
    
    async def load_langgraph_checkpoint(self, 
                                      thread_id: str, 
                                      config_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a LangGraph workflow checkpoint.
        
        Args:
            thread_id: Thread/session ID
            config_id: Configuration ID
            
        Returns:
            Checkpoint data or None
        """
        try:
            if LANGGRAPH_AVAILABLE and self.checkpoint_saver:
                # Use LangGraph's native checkpoint loader
                checkpoint = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.checkpoint_saver.get({
                        "thread_id": thread_id,
                        "config_id": config_id
                    })
                )
                
                if checkpoint:
                    return checkpoint.get("state")
            else:
                # Fallback to our state system
                checkpoint_id = f"checkpoint_{thread_id}_{config_id}"
                checkpoint_state = await self.get_state(checkpoint_id, StateType.CHECKPOINT)
                
                if checkpoint_state:
                    return checkpoint_state.data.get("data")
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading LangGraph checkpoint: {e}")
            return None
    
    async def save_context(self, context_id: str, context_data: Dict[str, Any]) -> bool:
        """
        Save context data.
        
        Args:
            context_id: Context identifier
            context_data: Context data
            
        Returns:
            bool: True if successful
        """
        try:
            await self.create_state(
                StateType.CONTEXT,
                context_data,
                context_id,
                tags=["context"]
            )
            return True
        except Exception as e:
            logger.error(f"Error saving context {context_id}: {e}")
            return False
    
    async def load_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Load context data.
        
        Args:
            context_id: Context identifier
            
        Returns:
            Context data or None
        """
        context_state = await self.get_state(context_id, StateType.CONTEXT)
        return context_state.data if context_state else None
    
    def get_persistent_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about the persistent state manager."""
        base_stats = self.get_statistics()
        
        # Add persistent-specific statistics
        type_counts = {}
        status_counts = {}
        
        for metadata in self._metadata_cache.values():
            type_counts[metadata.type.value] = type_counts.get(metadata.type.value, 0) + 1
            status_counts[metadata.status.value] = status_counts.get(metadata.status.value, 0) + 1
        
        base_stats.update({
            "total_states": len(self._metadata_cache),
            "states_by_type": type_counts,
            "states_by_status": status_counts,
            "langgraph_available": LANGGRAPH_AVAILABLE,
            "storage_directories": {
                "sessions": str(self.sessions_path),
                "conversations": str(self.conversations_path),
                "checkpoints": str(self.checkpoints_path),
                "context": str(self.context_path),
                "archive": str(self.archive_path)
            }
        })
        
        return base_stats
