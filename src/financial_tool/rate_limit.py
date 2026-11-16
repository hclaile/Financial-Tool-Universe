"""Token-bucket rate limiter."""

from __future__ import annotations

import time
import yaml

from .errors import RateLimitExceededError


class RateLimiter:
    def __init__(self, config: dict) -> None:
        self._default = config.get("default", {})
        self._per_tool: dict[str, dict] = config.get("per_tool", {})
        self._per_source: dict[str, dict] = config.get("per_source", {})
        self._buckets: dict[str, tuple[float, float]] = {}

    @classmethod
    def from_config(cls) -> RateLimiter:
        from finance_tool_universe.config import load_config
        cfg = load_config().get("rate_limits") or {}
        return cls(cfg.get("rate_limits", cfg))

    def _bucket_key(self, scope: str, name: str) -> str:
        return f"{scope}:{name}"

    def _get_limits(self, scope: str, name: str) -> tuple[int, int]:
        mapping = getattr(self, f"_{scope}", self._default)
        legacy_name = name.replace("_get_", ".", 1) if "_get_" in name else name.replace("_", ".", 1)
        cfg = mapping.get(name, mapping.get(legacy_name, self._default))
        return cfg.get("requests", 60), cfg.get("burst", 10)

    def check(self, scope: str, name: str) -> None:
        key = self._bucket_key(scope, name)
        rate, burst = self._get_limits(scope, name)
        now = time.time()
        tokens, last = self._buckets.get(key, (float(burst), now))
        refill = (now - last) * (rate / 60.0)
        tokens = min(burst, tokens + refill)
        if tokens < 1:
            raise RateLimitExceededError(name, f"{rate}/min")
        self._buckets[key] = (tokens - 1, now)

    @classmethod
    def attach(cls, mcp) -> None:
        limiter = cls.from_config()
        mcp._rate_limiter = limiter  # type: ignore[attr-defined]
