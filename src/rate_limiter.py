"""
rate_limiter.py -- Redis-backed rate limiter with in-memory fallback.
"""
import os, time, threading
from collections import defaultdict
from typing import Dict, Tuple, List
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from src.logging_config import get_logger

logger = get_logger(__name__)
REDIS_URL = os.environ.get("REDIS_URL", "").strip()
_redis_client = None
_redis_available = False

def _get_redis():
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client if _redis_available else None
    if not REDIS_URL:
        return None
    try:
        import redis
        c = redis.from_url(REDIS_URL, socket_connect_timeout=2, socket_timeout=2, decode_responses=True)
        c.ping()
        _redis_client = c
        _redis_available = True
        logger.info("Redis connected: %s", REDIS_URL)
        return c
    except Exception as e:
        logger.warning("Redis unavailable (%s) -- in-memory fallback.", e)
        _redis_available = False
        return None

class SlidingWindowCounter:
    def __init__(self):
        self._lock = threading.Lock()
        self._req: Dict[Tuple, List[float]] = defaultdict(list)
    def is_allowed(self, ip, pattern, max_req, window):
        key = (ip, pattern)
        now = time.monotonic()
        with self._lock:
            self._req[key] = [t for t in self._req[key] if t > now - window]
            if len(self._req[key]) >= max_req:
                return False
            self._req[key].append(now)
            return True

DEFAULT_RATE_LIMITS = {
    "/analyze":(60,60), "/analyze_batch":(10,60), "/model/refine":(3,300),
    "/logs/clear":(5,60), "/alerts/resolve":(30,60), "/alerts":(60,60),
    "/analytics":(30,60), "_default":(120,60),
}
EXEMPT = {"/","/health","/docs","/openapi.json","/redoc","/showcase"}

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limits=None):
        super().__init__(app)
        self._mem = SlidingWindowCounter()
        self.rate_limits = rate_limits or DEFAULT_RATE_LIMITS
        logger.info("Rate limiter: %s backend.", "Redis" if _get_redis() else "in-memory")

    def _ip(self, req): 
        fwd = req.headers.get("X-Forwarded-For")
        return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "unknown")

    def _match(self, path):
        clean = path.replace("/v1","").rstrip("/") or "/"
        for p,(mr,w) in self.rate_limits.items():
            if p != "_default" and (clean == p or clean.startswith(p+"/")):
                return p, mr, w
        d = self.rate_limits.get("_default",(120,60))
        return "_default", d[0], d[1]

    def _redis_ok(self, ip, pattern, mr, w):
        r = _get_redis()
        if not r: return True
        key = f"rl:{pattern}:{ip}"
        now = time.time()
        pipe = r.pipeline()
        pipe.zremrangebyscore(key,"-inf",now-w); pipe.zcard(key)
        pipe.zadd(key,{str(now):now}); pipe.expire(key,w+1)
        return pipe.execute()[1] < mr

    async def dispatch(self, req, call_next):
        path = req.url.path.rstrip("/") or "/"
        if path in EXEMPT or path.startswith("/static"):
            return await call_next(req)
        ip = self._ip(req)
        pattern, mr, w = self._match(path)
        allowed = self._redis_ok(ip,pattern,mr,w) if _get_redis() else self._mem.is_allowed(ip,pattern,mr,w)
        if not allowed:
            logger.warning("Rate limit exceeded: %s from %s", path, ip)
            return JSONResponse(status_code=429,
                content={"detail":"Rate limit exceeded.","limit":mr,"window_seconds":w},
                headers={"Retry-After":str(w)})
        return await call_next(req)
