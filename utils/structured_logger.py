import logging
import json
import time
import os

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "component": getattr(record, "component", "system"),
            "event": getattr(record, "event", ""),
            "message": record.getMessage(),
            "context": getattr(record, "context", {}),
        }
        return json.dumps(log_record)

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

handler = logging.FileHandler("logs/system.log")
handler.setFormatter(JsonFormatter())
logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def log_command_event(event, command_id, command_type, status, context=None):
    logger.info(
        "",
        extra={
            "component": "command_executor",
            "event": event,
            "context": {
                "command_id": command_id,
                "command_type": command_type,
                "status": status,
                **(context or {})
            }
        }
    )
