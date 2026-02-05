"""
Structured logging configuration using structlog
Provides JSON-formatted logs with context enrichment
"""

import logging
import structlog
import sys
from datetime import datetime
from typing import Any, Dict

def add_timestamp(logger, method_name, event_dict):
    """Add ISO timestamp to every log entry"""
    event_dict["timestamp"] = datetime.utcnow().isoformat() + "Z"
    return event_dict

def add_app_context(logger, method_name, event_dict):
    """Add application-level context"""
    event_dict["app"] = "document-intelligence"
    event_dict["version"] = "0.3.0"
    return event_dict

def configure_logging(log_level: str = "INFO", json_output: bool = True):
    """
    Configure structured logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        json_output: Whether to output JSON formatted logs
    """
    
    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        add_timestamp,
        add_app_context,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if json_output:
        # JSON output for production
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console output for development
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging to work with structlog
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
    )

def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Structured logger with bound context
    """
    return structlog.get_logger(name)

# Helper function to add request context
def bind_request_context(request_id: str = None, **kwargs):
    """
    Bind request-specific context to the current logger
    
    Usage:
        bind_request_context(request_id="abc123", user_id="user1")
        logger.info("processing request")  # Will include request_id and user_id
    """
    context = {}
    if request_id:
        context["request_id"] = request_id
    context.update(kwargs)
    structlog.contextvars.bind_contextvars(**context)

def clear_context():
    """Clear all bound context variables"""
    structlog.contextvars.clear_contextvars()
