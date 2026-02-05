"""
Rate limiter configuration using SlowAPI (based on limits and Redis)
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
import os
from app.core.logging_config import get_logger

logger = get_logger(__name__)

def get_limiter_storage_uri():
    """Get Redis storage URI for rate limiting"""
    # Use the same Redis URL as the cache/celery
    # slowapi expects 'redis://host:port/db'
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    
    # If using rediss (SSL), ensure compatibility
    if not redis_url.startswith("redis://") and not redis_url.startswith("rediss://"):
        return "memory://"
        
    return redis_url

# Initialize Limiter
# Use remote address (IP) as the key identifier
limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=get_limiter_storage_uri(),
    default_limits=["200/minute"],  # High default limit
    application_limits=["5000/day"],
    headers_enabled=True  # Return X-RateLimit-* headers
)

def rate_limit_exceeded_handler(request, exc):
    """
    Custom handler for rate limit exceptions
    Logs the violation and returns standardized error response
    """
    client_ip = get_remote_address(request)
    logger.warning("rate_limit_exceeded", ip=client_ip, path=request.url.path, limit=str(exc))
    
    return _rate_limit_exceeded_handler(request, exc)
