"""
Project-S v2 Architecture - Dependency Injection Container
========================================================

This module provides a lightweight dependency injection system for the v2 architecture.
It manages component lifecycle and dependencies cleanly.
"""

import logging
import asyncio
from typing import Dict, Any, Type, TypeVar, Callable, Optional, Set
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ComponentLifecycle(ABC):
    """Base class for components with lifecycle management."""
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the component."""
        pass

@dataclass
class ComponentRegistration:
    """Registration information for a component."""
    component_type: Type
    factory: Callable
    singleton: bool = True
    dependencies: Set[str] = None
    initialized: bool = False
    instance: Any = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = set()

class DependencyContainer:
    """
    Lightweight dependency injection container for v2 architecture.
    """
    
    def __init__(self):
        self._components: Dict[str, ComponentRegistration] = {}
        self._initialization_order: list = []
        self._shutdown_order: list = []
        
    def register(self, 
                name: str, 
                component_type: Type[T], 
                factory: Callable[[], T] = None,
                singleton: bool = True,
                dependencies: Set[str] = None) -> 'DependencyContainer':
        """
        Register a component with the container.
        
        Args:
            name: Component name
            component_type: Component class/type
            factory: Factory function to create the component
            singleton: Whether to create a single instance
            dependencies: Set of dependency names
            
        Returns:
            Self for method chaining
        """
        if factory is None:
            factory = component_type
            
        registration = ComponentRegistration(
            component_type=component_type,
            factory=factory,
            singleton=singleton,
            dependencies=dependencies or set()
        )
        
        self._components[name] = registration
        logger.debug(f"Registered component: {name}")
        return self
    
    def get(self, name: str) -> Any:
        """
        Get a component instance.
        
        Args:
            name: Component name
            
        Returns:
            Component instance
            
        Raises:
            KeyError: If component is not registered
        """
        if name not in self._components:
            raise KeyError(f"Component '{name}' is not registered")
        
        registration = self._components[name]
        
        # Return existing instance for singletons
        if registration.singleton and registration.instance is not None:
            return registration.instance
        
        # Create new instance
        try:
            # Resolve dependencies first
            kwargs = {}
            for dep_name in registration.dependencies:
                kwargs[dep_name] = self.get(dep_name)
            
            # Create instance
            instance = registration.factory(**kwargs)
            
            # Store for singletons
            if registration.singleton:
                registration.instance = instance
                
            logger.debug(f"Created instance for component: {name}")
            return instance
            
        except Exception as e:
            logger.error(f"Failed to create instance for component '{name}': {e}")
            raise
    
    def is_registered(self, name: str) -> bool:
        """Check if a component is registered."""
        return name in self._components
    
    def get_dependencies(self, name: str) -> Set[str]:
        """Get the dependencies of a component."""
        if name not in self._components:
            return set()
        return self._components[name].dependencies.copy()
    
    async def initialize_all(self) -> None:
        """
        Initialize all registered components in dependency order.
        """
        logger.info("Initializing all components...")
        
        # Calculate initialization order
        self._initialization_order = self._calculate_initialization_order()
        
        # Initialize components
        for component_name in self._initialization_order:
            await self._initialize_component(component_name)
            
        logger.info(f"Initialized {len(self._initialization_order)} components")
    
    async def shutdown_all(self) -> None:
        """
        Shutdown all initialized components in reverse order.
        """
        logger.info("Shutting down all components...")
        
        # Shutdown in reverse order
        self._shutdown_order = list(reversed(self._initialization_order))
        
        for component_name in self._shutdown_order:
            await self._shutdown_component(component_name)
            
        logger.info("All components shut down")
    
    async def _initialize_component(self, name: str) -> None:
        """Initialize a single component."""
        registration = self._components[name]
        
        if registration.initialized:
            return
            
        try:
            # Get or create instance
            instance = self.get(name)
            
            # Initialize if it has lifecycle
            if isinstance(instance, ComponentLifecycle):
                await instance.initialize()
                
            registration.initialized = True
            logger.debug(f"Initialized component: {name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize component '{name}': {e}")
            raise
    
    async def _shutdown_component(self, name: str) -> None:
        """Shutdown a single component."""
        registration = self._components[name]
        
        if not registration.initialized or registration.instance is None:
            return
            
        try:
            # Shutdown if it has lifecycle
            if isinstance(registration.instance, ComponentLifecycle):
                await registration.instance.shutdown()
                
            registration.initialized = False
            logger.debug(f"Shut down component: {name}")
            
        except Exception as e:
            logger.error(f"Failed to shutdown component '{name}': {e}")
    
    def _calculate_initialization_order(self) -> list:
        """Calculate the order to initialize components based on dependencies."""
        # Topological sort for dependency resolution
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(name: str):
            if name in temp_visited:
                raise ValueError(f"Circular dependency detected involving '{name}'")
            if name in visited:
                return
                
            temp_visited.add(name)
            
            # Visit dependencies first
            if name in self._components:
                for dep in self._components[name].dependencies:
                    if dep not in self._components:
                        raise ValueError(f"Dependency '{dep}' not registered for component '{name}'")
                    visit(dep)
            
            temp_visited.remove(name)
            visited.add(name)
            order.append(name)
        
        # Visit all components
        for component_name in self._components:
            visit(component_name)
            
        return order

# Global container instance
_container: Optional[DependencyContainer] = None

def get_container() -> DependencyContainer:
    """Get the global dependency container."""
    global _container
    if _container is None:
        _container = DependencyContainer()
    return _container

def set_container(container: DependencyContainer) -> None:
    """Set the global dependency container."""
    global _container
    _container = container

def reset_container() -> None:
    """Reset the global container."""
    global _container
    _container = None
