"""
Global Rate Limiter for LLM API Calls

P0-2 FIX: Prevents Error 429 "并发数过高" by enforcing global concurrency limits
across all LLM instances and pipeline stages.

Author: P0 Fix
Date: 2026-01-15
Priority: P0 (Critical infrastructure fix)
"""

import threading
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class GlobalRateLimiter:
    """
    Global rate limiter to prevent API rate limiting (Error 429).

    Features:
    - Semaphore-based concurrency limit (default: 2 concurrent requests)
    - Tracks active requests across all LLM instances
    - Provides wait time estimation
    - Thread-safe implementation
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, max_concurrent: int = 2):
        """
        Singleton pattern to ensure only one global limiter exists.

        Args:
            max_concurrent: Maximum number of concurrent API calls (default: 2)
        """
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_concurrent: int = 2):
        """
        Initialize the global rate limiter.

        Args:
            max_concurrent: Maximum number of concurrent API calls (default: 2)
                           Recommended: 1-2 for most APIs to avoid 429 errors
        """
        if self._initialized:
            return

        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.active_count = 0
        self.total_requests = 0
        self._lock = threading.Lock()

        logger.info(f"[P0-2] Global Rate Limiter initialized: max_concurrent={max_concurrent}")
        self._initialized = True

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        Acquire a slot for API call.

        Args:
            timeout: Maximum time to wait for slot (None = wait forever)

        Returns:
            True if slot acquired, False if timeout
        """
        acquired = self.semaphore.acquire(blocking=True, timeout=timeout)

        if acquired:
            with self._lock:
                self.active_count += 1
                self.total_requests += 1
                logger.debug(f"[P0-2] Acquired slot: active={self.active_count}/{self.max_concurrent}, total={self.total_requests}")
        else:
            logger.warning(f"[P0-2] Failed to acquire slot (timeout={timeout}s)")

        return acquired

    def release(self):
        """
        Release a slot after API call completes.
        """
        with self._lock:
            self.active_count -= 1
            logger.debug(f"[P0-2] Released slot: active={self.active_count}/{self.max_concurrent}")

        self.semaphore.release()

    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with current stats
        """
        with self._lock:
            return {
                'max_concurrent': self.max_concurrent,
                'active_count': self.active_count,
                'total_requests': self.total_requests,
                'available_slots': self.semaphore._value
            }

    def __enter__(self):
        """Context manager entry."""
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()


# Global instance
_global_limiter = None


def get_global_limiter(max_concurrent: int = 2) -> GlobalRateLimiter:
    """
    Get the global rate limiter instance.

    Args:
        max_concurrent: Maximum concurrent requests (only used on first call)

    Returns:
        GlobalRateLimiter singleton instance
    """
    global _global_limiter

    if _global_limiter is None:
        _global_limiter = GlobalRateLimiter(max_concurrent)

    return _global_limiter
