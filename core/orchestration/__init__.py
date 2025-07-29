"""
Core Orchestration Module - Project-S V2 
=========================================
Workflow engine and orchestration components.
"""

try:
    from .workflow_engine import *
    WORKFLOW_ENGINE_AVAILABLE = True
except ImportError as e:
    WORKFLOW_ENGINE_AVAILABLE = False
    import logging
    logging.getLogger(__name__).warning(f"Workflow engine not available: {e}")

__all__ = ["WORKFLOW_ENGINE_AVAILABLE"]