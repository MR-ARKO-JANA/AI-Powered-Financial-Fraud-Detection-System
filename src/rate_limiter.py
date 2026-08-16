"""
rate_limiter.py — In-Memory Sliding Window Rate Limiter for FinGuard AI.

Provides FastAPI middleware for per-IP request rate limiting with
configurable limits per endpoint pattern.
"""

import time
import logging
from collections import defaultdict
from typing import Dict, Tuple, List
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
import threading

from src.logging_config import get_logger

logger = get_logger(__name__)


class SlidingWindowCounter:
    """Thread-safe sliding window rate counter."""

    def __init__(self):
        self._lock = threading.Lock()
        # Key: (client_ip, path_pattern) -> List of timestamps
        self._requests: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str, path_pattern: str, max_requests: int, window_seconds: int) -> bool:
        """Check if request is allowed under rate limit."""
        key = (client_ip, path_pattern)
        now = time.monotonic()
        cutoff = now - window_seconds

        with self._lock:
            # Prune old entries
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]

            if len(self._requests[key]) >= max_requests:
                return False

            self._requests[key].append(now)
            return True

    def cleanup(self, max_age_seconds: int = 300):
        """Remove stale entries older than max_age_seconds."""
        now = time.monotonic()
        cutoff = now - max_age_seconds

        with self._lock:
            stale_keys = [
                k for k, timestamps in self._requests.items()
                if not timestamps or timestamps[-1] < cutoff
            ]
            for k in stale_keys:
                del self._requests[k]


# Default rate limits: (max_requests, window_seconds)
DEFAULT_RATE_LIMITS: Dict[str, Tuple[int, int]] = {
    "/analyze": (60, 60),           # 60 requests per minute
    "/analyze_batch": (10, 60),     # 10 batch requests per minute
    "/model/refine": (3, 300),      # 3 retrain requests per 5 minutes
    "/logs/clear": (5, 60),         # 5 clear requests per minute
    "/alerts/resolve": (30, 60),    # 30 feedback submissions per minute
    "/alerts": (60, 60),            # 60 reads per minute
    "/analytics": (30, 60),         # 30 analytics reads per minute
    "_default": (120, 60),          # 120 requests per minute for unlisted endpoints
}

# Paths exempt from rate limiting
RATE_LIMIT_EXEMPT: set = {"/", "/health", "/docs", "/openapi.json", "/redoc", "/showcase"}


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Sliding window rate limiter middleware.

    Applies per-IP rate limits based on endpoint patterns.
    Returns 429 Too Many Requests when limit is exceeded.
    """

    def __init__(self, app, rate_limits: Dict[str, Tuple[int, int]] | None = None):
        super().__init__(app)
        self.counter = SlidingWindowCounter()
        self.rate_limits = rate_limits or DEFAULT_RATE_LIMITS
        logger.info("Rate limiter initialized with %d endpoint rules.", len(self.rate_limits))

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP, respecting X-Forwarded-For behind a proxy."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _match_rate_limit(self, path: str) -> Tuple[str, int, int]:
        """Find the matching rate limit rule for a path."""
        # Strip /v1/ prefix for matching
        clean_path = path.replace("/v1", "").rstrip("/") or "/"

        for pattern, (max_req, window) in self.rate_limits.items():
            if pattern == "_default":
                continue
            if clean_path == pattern or clean_path.startswith(pattern + "/"):
                return pattern, max_req, window

        default = self.rate_limits.get("_default", (120, 60))
        return "_default", default[0], default[1]

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path.rstrip("/") or "/"

        # Exempt paths
        if path in RATE_LIMIT_EXEMPT or path.startswith("/static"):
            return await call_next(request)

        client_ip = self._get_client_ip(request)
        pattern, max_requests, window_seconds = self._match_rate_limit(path)

        if not self.counter.is_allowed(client_ip, pattern, max_requests, window_seconds):
            logger.warning(
                "Rate limit exceeded: %s from %s on %s (limit: %d/%ds)",
                path, client_ip, pattern, max_requests, window_seconds,
            )
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down.",
                    "limit": max_requests,
                    "window_seconds": window_seconds,
                },
                headers={"Retry-After": str(window_seconds)},
            )

        return await call_next(request)
