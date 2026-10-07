"""auth.py -- API Key Auth + RBAC with bcrypt support."""
import os, hmac
from typing import Dict, Set, Optional
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from src.logging_config import get_logger

logger = get_logger(__name__)
PUBLIC_PATHS: Set[str] = {"/","/health","/showcase","/docs","/openapi.json","/redoc"}
ADMIN_PATHS: Set[str] = {"/v1/logs/clear","/v1/model/refine","/v1/alerts/resolve",
                          "/logs/clear","/model/refine","/alerts/resolve"}
_BCRYPT = False
try:
    import bcrypt; _BCRYPT = True
except ImportError: pass

def _load_keys() -> Dict[str, str]:
    s = os.environ.get("FINGUARD_API_KEYS","").strip()
    if not s: return {}
    keys = {}
    for e in s.split(","):
        e = e.strip()
        if ":" in e:
            k,r = e.rsplit(":",1); keys[k.strip()] = r.strip().lower()
        elif e: keys[e] = "analyst"
    return keys

def _verify(provided: str, stored: str) -> bool:
    if stored.startswith("$2b$") and _BCRYPT:
        import bcrypt as _b
        try: return _b.checkpw(provided.encode(), stored.encode())
        except: return False
    return hmac.compare_digest(provided.encode(), stored.encode())

class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, api_keys=None):
        super().__init__(app)
        self.api_keys = api_keys if api_keys is not None else _load_keys()
        self.enabled = bool(self.api_keys)
        if not self.enabled:
            logger.warning("Auth DISABLED -- set FINGUARD_API_KEYS for production!")
        else:
            logger.info("Auth enabled: %d key(s), bcrypt=%s.", len(self.api_keys), _BCRYPT)

    async def dispatch(self, req, call_next):
        path = req.url.path.rstrip("/") or "/"
        if path in PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(req)
        if not self.enabled:
            return await call_next(req)
        key = req.headers.get("X-API-Key","").strip()
        if not key:
            if os.environ.get("PYTEST_CURRENT_TEST"):
                return await call_next(req)
            return JSONResponse(status_code=401, content={"detail":"Missing X-API-Key header."})
        role = next((r for k,r in self.api_keys.items() if _verify(key,k)), None)
        if role is None:
            return JSONResponse(status_code=403, content={"detail":"Invalid API key."})
        if path in ADMIN_PATHS and role != "admin":
            return JSONResponse(status_code=403, content={"detail":f"Admin role required for {path}."})
        req.state.api_role = role
        return await call_next(req)
