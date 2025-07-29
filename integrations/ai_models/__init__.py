"""
AI Models Integration - True Golden Age Multi-Model Client
----------------------------------------------------------
"""

try:
    from .multi_model_ai_client import *
    MULTI_MODEL_CLIENT_AVAILABLE = True
except ImportError:
    MULTI_MODEL_CLIENT_AVAILABLE = False

__all__ = ["MULTI_MODEL_CLIENT_AVAILABLE"]
