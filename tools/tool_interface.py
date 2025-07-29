"""
Project-S Tool Interface
-----------------------
Ez a modul definiálja az egységes eszköz (tool) interfészt a Project-S rendszerhez.
Minden eszköznek meg kell valósítania ezt az interfészt.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio
import json
import inspect

logger = logging.getLogger(__name__)

class BaseTool(ABC):
    """
    Alap eszköz interfész, amit minden Project-S eszköznek implementálnia kell.
    """
    
    def __init__(self):
        """Inicializálja az eszközt és beállítja az alapvető tulajdonságokat."""
        self.name = self.__class__.__name__
        self.description = self.__doc__ or "Nincs leírás"
        self.category = "general"
        self.version = "1.0.0"
        self.requires_permissions = False
        self.is_safe = True
        
        # Eszköz metaadatok kinyerése az osztály dokumentációjából
        self._parse_metadata()
        
        logger.debug(f"Tool inicializálva: {self.name}")
        
    def _parse_metadata(self):
        """Kinyeri a metaadatokat az osztály dokumentációjából."""
        if self.__doc__:
            doc_lines = self.__doc__.split("\n")
            
            # Kategória keresése
            for line in doc_lines:
                if "Category:" in line:
                    self.category = line.split("Category:")[1].strip()
                elif "Version:" in line:
                    self.version = line.split("Version:")[1].strip()
                elif "Requires permissions:" in line.lower():
                    self.requires_permissions = "yes" in line.lower() or "true" in line.lower()
                elif "Safe:" in line:
                    self.is_safe = not ("no" in line.lower() or "false" in line.lower())
    
    def get_info(self) -> Dict[str, Any]:
        """
        Visszaadja az eszköz információit strukturált formában.
        
        Returns:
            Dict[str, Any]: Az eszköz metaadatai
        """
        # Bemeneti paraméterek információinak kinyerése
        parameters = {}
        sig = inspect.signature(self.execute)
        for name, param in sig.parameters.items():
            if name == "self":
                continue
                
            param_info = {
                "required": param.default == inspect.Parameter.empty,
                "type": str(param.annotation) if param.annotation != inspect.Parameter.empty else "any",
            }
            
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default
                
            parameters[name] = param_info
            
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "requires_permissions": self.requires_permissions,
            "is_safe": self.is_safe,
            "parameters": parameters
        }
        
    @abstractmethod
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Végrehajtja az eszköz funkcióját.
        
        Args:
            **kwargs: Az eszköz-specifikus paraméterek
            
        Returns:
            Dict[str, Any]: Az eredmény szótár formájában
        """
        pass
