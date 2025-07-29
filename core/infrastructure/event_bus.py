import asyncio
import logging
from typing import Dict, List, Callable, Any, Awaitable

logger = logging.getLogger(__name__)

class EventBus:
    """
    Event bus for the Project-S system implementing the publish/subscribe pattern.
    Allows components to subscribe to events and publish events without direct coupling.
    """
    
    def __init__(self):
        """Initialize an empty event bus with no subscribers."""
        self._subscribers: Dict[str, List[Callable[[Any], Awaitable[None]]]] = {}
        logger.info("EventBus initialized")
        
    async def default_event_handler(self, event_data: Any) -> None:
        """
        Default event handler for logging events
        
        Args:
            event_data: The data associated with the event
        """
        logger.debug(f"Default handler received event data: {event_data}")
        
    def register_default_handlers(self) -> None:
        """Register default event handlers for common events"""
        self.subscribe("command.received", self.default_event_handler)
        self.subscribe("command.processing", self.default_event_handler)
        self.subscribe("command.processed", self.default_event_handler)
        self.subscribe("response.generating", self.default_event_handler)
        self.subscribe("response.ready", self.default_event_handler)
        logger.info("Default event handlers registered")
    
    def subscribe(self, event_type: str, callback: Callable[[Any], Awaitable[None]]) -> None:
        """
        Subscribe to an event type with a callback function.
        
        Args:
            event_type (str): The type of event to subscribe to
            callback (Callable): An async function that will be called when the event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        self._subscribers[event_type].append(callback)
        logger.info(f"Subscriber added for event type: {event_type}")
    
    def unsubscribe(self, event_type: str, callback: Callable[[Any], Awaitable[None]]) -> bool:
        """
        Unsubscribe from an event type.
        
        Args:
            event_type (str): The type of event to unsubscribe from
            callback (Callable): The callback function to remove
            
        Returns:
            bool: True if the callback was found and removed, False otherwise
        """
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            logger.info(f"Subscriber removed from event type: {event_type}")
            return True
        return False
    
    async def publish(self, event_type: str, event_data: Any = None) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type (str): The type of event being published
            event_data (Any, optional): Data associated with the event
        """
        logger.info(f"Publishing event: {event_type}")
        if event_type in self._subscribers:
            tasks = []
            for callback in self._subscribers[event_type]:
                tasks.append(asyncio.create_task(callback(event_data)))
            
            if tasks:
                # Wait for all subscribers to process the event
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            logger.info(f"No subscribers for event type: {event_type}")

# Create a singleton instance
event_bus = EventBus()
