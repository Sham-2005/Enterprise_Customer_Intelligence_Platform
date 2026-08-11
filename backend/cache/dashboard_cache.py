"""
High-Performance Dashboard Caching Service for ECIP.
Provides thread-safe in-memory TTL caching with hash-based key generation,
automatic expiration, cache statistics, and cache invalidation.
"""

import time
import hashlib
import json
from typing import Any, Dict, Optional, Tuple
from utils.logger import setup_logger

logger = setup_logger("ECIP.DashboardCache")

class DashboardCache:
    """Thread-safe TTL memory cache manager for dashboard query responses and datasets."""

    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self._store: Dict[str, Tuple[Any, float]] = {}
        self._hits: int = 0
        self._misses: int = 0

    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generates a deterministic MD5 hash key from prefix and filter parameter dictionary."""
        try:
            # Convert un-serializable objects (e.g. pandas Timestamps, dates) to string representation
            clean_params = {}
            for k, v in params.items():
                if isinstance(v, (list, tuple)):
                    clean_params[k] = [str(item) for item in v]
                else:
                    clean_params[k] = str(v)
            
            param_str = json.dumps(clean_params, sort_keys=True)
            param_hash = hashlib.md5(param_str.encode('utf-8')).hexdigest()
            return f"{prefix}:{param_hash}"
        except Exception as e:
            logger.warning(f"Error generating cache key: {e}")
            return f"{prefix}:{str(params)}"

    def get(self, prefix: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """
        Retrieves cached data if key exists and has not expired.
        """
        params = params or {}
        key = self._generate_key(prefix, params)

        if key in self._store:
            val, expiry = self._store[key]
            if time.time() < expiry:
                self._hits += 1
                logger.debug(f"Cache HIT for key: {key}")
                return val
            else:
                # Expired
                del self._store[key]
                logger.debug(f"Cache EXPIRED for key: {key}")

        self._misses += 1
        logger.debug(f"Cache MISS for key: {key}")
        return None

    def set(self, prefix: str, value: Any, params: Optional[Dict[str, Any]] = None, ttl: Optional[int] = None) -> str:
        """
        Stores value in cache with specified TTL (in seconds).
        """
        params = params or {}
        key = self._generate_key(prefix, params)
        ttl = ttl if ttl is not None else self.default_ttl
        expiry = time.time() + ttl
        self._store[key] = (value, expiry)
        logger.debug(f"Cache SET for key: {key} (TTL: {ttl}s)")
        return key

    def clear(self) -> None:
        """Clears all cached entries."""
        self._store.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Dashboard cache completely cleared.")

    def invalidate_prefix(self, prefix: str) -> int:
        """Invalidates all keys matching a specific prefix."""
        keys_to_del = [k for k in self._store.keys() if k.startswith(f"{prefix}:")]
        for k in keys_to_del:
            del self._store[k]
        logger.info(f"Invalidated {len(keys_to_del)} cache entries for prefix '{prefix}'")
        return len(keys_to_del)

    def get_stats(self) -> Dict[str, Any]:
        """Returns cache efficiency metrics."""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
        return {
            "cached_items_count": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate_pct": round(hit_rate, 2)
        }

# Global Singleton Cache Instance
dashboard_cache = DashboardCache(default_ttl=3600)
