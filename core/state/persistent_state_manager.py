"""
Project-S Persistent State Manager (Golden Age Component)
--------------------------------------------------------
This module implements persistent state management for the Project-S system.
It allows storing and retrieving conversation history, LangGraph checkpoints,
and preserving context across system restarts.

MIGRATED FROM: C:\project_s_agent\integrations\persistent_state_manager.py
MIGRATION DATE: 2025-01-02
COMPONENT TYPE: Golden Age - Critical state management
V2 INTEGRATION STATUS: Migrated with compatibility layer
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import pickle
import uuid
import aiofiles
from contextlib import asynccontextmanager

# Import LangGraph-specific elements if available
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.checkpoint import BaseCheckpointSaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

logger = logging.getLogger(__name__)

class PersistentStateManager:
    """
    Manages persistent state for the Project-S system, including:
    - Conversation history storage and retrieval
    - LangGraph checkpoint management
    - Session persistence
    - Context preservation across restarts
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize the persistent state manager.
        
        Args:
            storage_path: Optional custom path for storing state data.
                          If not provided, defaults to PROJECT_ROOT/memory/state
        """
        if storage_path:
            self.storage_path = storage_path
        else:
            # V2 architecture compatible path
            self.storage_path = Path(__file__).parent.parent.parent / "memory" / "state"
            
        # Ensure storage directories exist
        self.conversations_path = self.storage_path / "conversations"
        self.checkpoints_path = self.storage_path / "checkpoints"
        self.sessions_path = self.storage_path / "sessions"
        self.context_path = self.storage_path / "context"
        
        os.makedirs(self.storage_path, exist_ok=True)
        os.makedirs(self.conversations_path, exist_ok=True)
        os.makedirs(self.checkpoints_path, exist_ok=True)
        os.makedirs(self.sessions_path, exist_ok=True)
        os.makedirs(self.context_path, exist_ok=True)
        
        # Initialize session tracking
        self.active_sessions = {}
        self._load_active_sessions()
        
        # Initialize LangGraph checkpoint saver if available
        self.checkpoint_saver = None
        if LANGGRAPH_AVAILABLE:
            self.checkpoint_saver = SqliteSaver(str(self.checkpoints_path / "langgraph_checkpoints.db"))
            logger.info("LangGraph checkpoint saver initialized")
        else:
            logger.warning("LangGraph not available, checkpoint functionality will be limited")
            
        logger.info(f"Persistent state manager initialized with storage at {self.storage_path}")
        
    def _load_active_sessions(self) -> None:
        """Load active sessions from disk."""
        try:
            sessions_file = self.sessions_path / "active_sessions.json"
            if sessions_file.exists():
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    self.active_sessions = json.load(f)
                    logger.info(f"Loaded {len(self.active_sessions)} active sessions")
        except Exception as e:
            logger.error(f"Error loading active sessions: {e}")
            self.active_sessions = {}
            
    async def _save_active_sessions(self) -> None:
        """Save active sessions to disk."""
        try:
            sessions_file = self.sessions_path / "active_sessions.json"
            async with aiofiles.open(sessions_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(self.active_sessions, indent=2))
        except Exception as e:
            logger.error(f"Error saving active sessions: {e}")
            
    async def create_session(self, session_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new session with optional initial data.
        
        Args:
            session_data: Optional initial data for the session
            
        Returns:
            str: The session ID
        """
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        session_info = {
            "id": session_id,
            "created_at": timestamp,
            "last_updated": timestamp,
            "data": session_data or {}
        }
        
        self.active_sessions[session_id] = session_info
        
        # Store the session persistently
        session_file = self.sessions_path / f"{session_id}.json"
        async with aiofiles.open(session_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(session_info, indent=2))
        
        # Update the active sessions list
        await self._save_active_sessions()
        
        logger.info(f"Created new session: {session_id}")
        return session_id
        
    async def update_session(self, session_id: str, session_data: Dict[str, Any]) -> bool:
        """
        Update an existing session with new data.
        
        Args:
            session_id: The session ID
            session_data: New data to update in the session
            
        Returns:
            bool: True if successful, False otherwise
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Attempted to update non-existent session: {session_id}")
            return False
            
        timestamp = datetime.now().isoformat()
        
        # Update in memory
        self.active_sessions[session_id]["last_updated"] = timestamp
        self.active_sessions[session_id]["data"].update(session_data)
        
        # Update on disk
        session_file = self.sessions_path / f"{session_id}.json"
        async with aiofiles.open(session_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(self.active_sessions[session_id], indent=2))
            
        logger.debug(f"Updated session: {session_id}")
        return True
        
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data by ID.
        
        Args:
            session_id: The session ID
            
        Returns:
            Optional[Dict[str, Any]]: Session data if found, None otherwise
        """
        # Try to get from memory first
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
            
        # If not in memory, try to load from disk
        session_file = self.sessions_path / f"{session_id}.json"
        if session_file.exists():
            try:
                async with aiofiles.open(session_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    session_data = json.loads(content)
                    
                    # Add to active sessions
                    self.active_sessions[session_id] = session_data
                    return session_data
            except Exception as e:
                logger.error(f"Error loading session {session_id}: {e}")
                
        return None
        
    async def end_session(self, session_id: str) -> bool:
        """
        End a session by moving it from active to archived status.
        
        Args:
            session_id: The session ID
            
        Returns:
            bool: True if successful, False otherwise
        """
        if session_id not in self.active_sessions:
            logger.warning(f"Attempted to end non-existent session: {session_id}")
            return False
            
        # Archive the session
        archived_dir = self.sessions_path / "archived"
        os.makedirs(archived_dir, exist_ok=True)
        
        session_data = self.active_sessions[session_id]
        session_data["ended_at"] = datetime.now().isoformat()
        
        # Write to archive
        archive_file = archived_dir / f"{session_id}.json"
        async with aiofiles.open(archive_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(session_data, indent=2))
            
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        # Delete active session file
        session_file = self.sessions_path / f"{session_id}.json"
        if session_file.exists():
            os.remove(session_file)
            
        # Update active sessions file
        await self._save_active_sessions()
        
        logger.info(f"Ended session: {session_id}")
        return True
    
    async def add_conversation_entry(self, 
                                   session_id: str, 
                                   role: str, 
                                   content: str, 
                                   metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Add a new entry to a conversation history.
        
        Args:
            session_id: The session ID
            role: Role of the message sender (e.g., "user", "assistant", "system")
            content: The message content
            metadata: Optional metadata for the message
            
        Returns:
            Dict[str, Any]: The created conversation entry
        """
        # Ensure session exists
        session = await self.get_session(session_id)
        if not session:
            logger.warning(f"Creating new session for conversation: {session_id}")
            await self.create_session({"conversation": []})
            
        # Create conversation entry
        timestamp = datetime.now().isoformat()
        entry_id = str(uuid.uuid4())
        
        entry = {
            "id": entry_id,
            "timestamp": timestamp,
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        # Load conversation history
        conv_file = self.conversations_path / f"{session_id}.json"
        conversation = []
        
        if conv_file.exists():
            try:
                async with aiofiles.open(conv_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    conversation = json.loads(content)
            except Exception as e:
                logger.error(f"Error loading conversation for session {session_id}: {e}")
                
        # Add new entry
        conversation.append(entry)
        
        # Save updated conversation
        async with aiofiles.open(conv_file, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(conversation, indent=2))
            
        # Update session
        if session:
            session_data = {"last_conversation_entry": timestamp}
            await self.update_session(session_id, session_data)
            
        logger.debug(f"Added conversation entry to session {session_id}")
        return entry
        
    async def get_conversation_history(self, 
                                     session_id: str, 
                                     limit: Optional[int] = None, 
                                     before_timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: The session ID
            limit: Optional maximum number of messages to return
            before_timestamp: Optional timestamp to get only messages before this time
            
        Returns:
            List[Dict[str, Any]]: Conversation history
        """
        conv_file = self.conversations_path / f"{session_id}.json"
        if not conv_file.exists():
            logger.warning(f"No conversation history found for session: {session_id}")
            return []
            
        try:
            async with aiofiles.open(conv_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                conversation = json.loads(content)
                
            # Apply filters
            if before_timestamp:
                conversation = [msg for msg in conversation if msg["timestamp"] < before_timestamp]
                
            if limit and limit > 0:
                conversation = conversation[-limit:]
                
            return conversation
        except Exception as e:
            logger.error(f"Error retrieving conversation history for session {session_id}: {e}")
            return []
            
    async def save_langgraph_checkpoint(self, 
                                      thread_id: str, 
                                      config_id: str, 
                                      checkpoint_data: Dict[str, Any]) -> bool:
        """
        Save a LangGraph workflow checkpoint.
        
        Args:
            thread_id: The thread/session ID
            config_id: The workflow configuration ID
            checkpoint_data: The checkpoint data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not LANGGRAPH_AVAILABLE or not self.checkpoint_saver:
            logger.warning("LangGraph not available, cannot save checkpoint")
            
            # Fallback to simple JSON storage
            checkpoint_dir = self.checkpoints_path / thread_id
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_file = checkpoint_dir / f"{config_id}.json"
            try:
                async with aiofiles.open(checkpoint_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(checkpoint_data, indent=2))
                logger.info(f"Saved fallback checkpoint for thread {thread_id}, config {config_id}")
                return True
            except Exception as e:
                logger.error(f"Error saving fallback checkpoint: {e}")
                return False
        
        # Use LangGraph's built-in checkpoint functionality
        try:
            # LangGraph's checkpoint saver is synchronous, so we run it in a thread
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.checkpoint_saver.persist(
                    thread_id=thread_id,
                    config_id=config_id,
                    state=checkpoint_data
                )
            )
            logger.info(f"Saved LangGraph checkpoint for thread {thread_id}, config {config_id}")
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
            thread_id: The thread/session ID
            config_id: The workflow configuration ID
            
        Returns:
            Optional[Dict[str, Any]]: The checkpoint data if found, None otherwise
        """
        if not LANGGRAPH_AVAILABLE or not self.checkpoint_saver:
            logger.warning("LangGraph not available, trying fallback checkpoint loading")
            
            # Try fallback JSON storage
            checkpoint_file = self.checkpoints_path / thread_id / f"{config_id}.json"
            if checkpoint_file.exists():
                try:
                    async with aiofiles.open(checkpoint_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        return json.loads(content)
                except Exception as e:
                    logger.error(f"Error loading fallback checkpoint: {e}")
                    return None
                    
            return None
            
        # Use LangGraph's built-in checkpoint functionality
        try:
            # LangGraph's checkpoint loader is synchronous, so we run it in a thread
            checkpoint = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.checkpoint_saver.get(
                    thread_id=thread_id,
                    config_id=config_id
                )
            )
            
            if checkpoint:
                logger.info(f"Loaded LangGraph checkpoint for thread {thread_id}, config {config_id}")
                return checkpoint
            else:
                logger.warning(f"No checkpoint found for thread {thread_id}, config {config_id}")
                return None
        except Exception as e:
            logger.error(f"Error loading LangGraph checkpoint: {e}")
            return None
            
    async def save_context(self, context_id: str, context_data: Dict[str, Any]) -> bool:
        """
        Save context data that should persist across system restarts.
        
        Args:
            context_id: A unique identifier for the context
            context_data: The context data to save
            
        Returns:
            bool: True if successful, False otherwise
        """
        context_file = self.context_path / f"{context_id}.json"
        try:
            async with aiofiles.open(context_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(context_data, indent=2))
            logger.debug(f"Saved context: {context_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving context {context_id}: {e}")
            return False
            
    async def load_context(self, context_id: str) -> Optional[Dict[str, Any]]:
        """
        Load persisted context data.
        
        Args:
            context_id: The context identifier
            
        Returns:
            Optional[Dict[str, Any]]: The context data if found, None otherwise
        """
        context_file = self.context_path / f"{context_id}.json"
        if not context_file.exists():
            return None
            
        try:
            async with aiofiles.open(context_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error loading context {context_id}: {e}")
            return None
            
    @asynccontextmanager
    async def session_context(self, session_id: Optional[str] = None):
        """
        Context manager for working with sessions.
        Creates a new session if session_id is None.
        
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
            # If we created a new session and it's not being used (no conversation entries),
            # we can clean it up
            if created_new:
                history = await self.get_conversation_history(session_id)
                if not history:
                    await self.end_session(session_id)

# Create singleton instance
persistent_state_manager = PersistentStateManager()
