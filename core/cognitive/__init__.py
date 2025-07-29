"""
Cognitive Package - True Golden Age Cognitive Core
--------------------------------------------------
"""

try:
    from .cognitive_core import CognitiveCore
    __all__ = ["CognitiveCore"]
except ImportError as e:
    print(f"Warning: CognitiveCore import failed: {e}")
    __all__ = []
