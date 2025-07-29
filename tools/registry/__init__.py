"""
Tools Registry - True Golden Age Smart Orchestrator
---------------------------------------------------
"""

try:
    from .smart_orchestrator import SmartToolOrchestrator
    SMART_ORCHESTRATOR_AVAILABLE = True
except ImportError:
    SMART_ORCHESTRATOR_AVAILABLE = False

__all__ = ["SmartToolOrchestrator"] if SMART_ORCHESTRATOR_AVAILABLE else []
