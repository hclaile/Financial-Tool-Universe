"""Cache manager — wraps in-memory / Redis / file caching."""

from __future__ import annotations

import time
import yaml
from typing import Any

from .errors import CacheError


class CacheManager:
    def __init__(self, config: dict) -> None:
        self._ttl_map: dict[str, int] = {}
        for pattern, seconds in config.get("ttl", {}).items():
            value = str(seconds).strip().lower()
            unit = value[-1] if value and value[-1] in "smhd" else "s"
            amount = value[:-1] if value[-1:] in "smhd" else value
            self._ttl_map[pattern] = int(amount) * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit]
        self._max_entries = config.get("max_entries", 10000)
        self._backend = config.get("backend", "memory")
        self._store: dict[str, tuple[float, Any]] = {}

    @classmethod
    def from_config(cls) -> CacheManager:
        from finance_tool_universe.config import load_config
        cfg = load_config().get("cache") or {}
        return cls(cfg.get("cache", cfg))

    def _resolve_ttl(self, key: str) -> int:
        for pattern, ttl in self._ttl_map.items():
            if key.startswith(pattern):
                return ttl
        return 3600

    def get(self, key: str) -> Any | None:
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.time() - ts > self._resolve_ttl(key):
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        if len(self._store) >= self._max_entries:
            self._evict()
        self._store[key] = (time.time(), value)

    def _evict(self) -> None:
        oldest = min(self._store, key=lambda k: self._store[k][0])
        del self._store[oldest]

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    @classmethod
    def attach(cls, mcp) -> None:
        cache = cls.from_config()
        mcp._cache = cache  # type: ignore[attr-defined]
