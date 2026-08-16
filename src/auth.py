"""
auth.py — API Key Authentication & Role-Based Access Control for FinGuard AI.

Provides FastAPI middleware for API key validation and RBAC enforcement.
Keys and roles are loaded from environment variables.
"""

import os
import hashlib
import hmac
import logging
from typing import Dict, Set, Optional
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.logging_config import get_logger

logger = get_logger(__name__)

# Public endpoints that don't require authentication
PUBLIC_PATHS: Set[str] = {
    "/",
    "/health",
    "/showcase",
    "/docs",
    "/openapi.json",
    "/redoc",
}

# Endpoints that require admin role
ADMIN_PATHS: Set[str] = {
    "/v1/logs/clear",
    "/v1/model/refine",
    "/v1/alerts/resolve",
    "/logs/clear",
    "/model/refine",
    "/alerts/resolve",
}


def _load_api_keys() -> Dict[str, str]:
    """
    Load API keys from environment variables.

    Format:
        FINGUARD_API_KEYS="key1:admin,key2:analyst,key3:viewer"

    Returns dict mapping key -> role.
    """
    keys_str = os.environ.get("FINGUARD_API_KEYS", "").strip()
    if not keys_str:
        return {}

    keys: Dict[str, str] = {}
    for entry in keys_str.split(","):
        entry = entry.strip()
        if ":" in entry:
            key, role = entry.rsplit(":", 1)
            keys[key.strip()] = role.strip().lower()
        elif entry:
            keys[entry.strip()] = "analyst"

    return keys


def _constant_time_compare(a: str, b: str) -> bool:
    """Constant-time string comparison to prevent timing attacks."""
    return hmac.compare_digest(a.encode(), b.encode())


class AuthMiddleware(BaseHTTPMiddleware):
    """
    API Key authentication middleware.

    - Public paths bypass authentication.
    - Static file paths bypass authentication.
    - All other paths require a valid X-API-Key header.
    - Admin paths additionally require admin role.
    - If no API keys are configured (dev mode), all requests pass through.
    """

    def __init__(self, app, api_keys: Optional[Dict[str, str]] = None):
        super().__init__(app)
        self.api_keys = api_keys if api_keys is not None else _load_api_keys()
        self.auth_enabled = bool(self.api_keys)

        if not self.auth_enabled:
            logger.warning(
                "No API keys configured (FINGUARD_API_KEYS not set). "
                "Authentication is DISABLED. Set FINGUARD_API_KEYS for production."
            )
        else:
            logger.info("Authentication enabled with %d API key(s).", len(self.api_keys))

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path.rstrip("/") or "/"

        # Allow public paths
        if path in PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)

        # If no keys configured (dev mode), allow everything
        if not self.auth_enabled:
            return await call_next(request)

        # Extract API key from header
        api_key = request.headers.get("X-API-Key", "").strip()
        if not api_key:
            logger.warning("Authentication failed: missing X-API-Key header for %s", path)
            return JSONResponse(
                status_code=401,
                content={"detail": "Missing X-API-Key header. Provide a valid API key."},
            )

        # Validate API key
        matched_role: Optional[str] = None
        for stored_key, role in self.api_keys.items():
            if _constant_time_compare(api_key, stored_key):
                matched_role = role
                break

        if matched_role is None:
            logger.warning("Authentication failed: invalid API key for %s", path)
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid API key."},
            )

        # Check admin role for restricted endpoints
        if path in ADMIN_PATHS and matched_role != "admin":
            logger.warning(
                "Authorization failed: role '%s' attempted admin endpoint %s", matched_role, path
            )
            return JSONResponse(
                status_code=403,
                content={"detail": f"Insufficient permissions. Admin role required for {path}."},
            )

        # Attach role to request state for downstream use
        request.state.api_role = matched_role
        return await call_next(request)
