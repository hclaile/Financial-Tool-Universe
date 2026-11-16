"""Authentication and permission enforcement."""

from __future__ import annotations

import yaml
from functools import wraps

from .errors import PermissionDeniedError


class AuthMiddleware:
    def __init__(self, permissions: dict) -> None:
        self._permissions = permissions
        self._role_cache: dict[str, set[str]] = {}
        self._build_role_cache()

    @classmethod
    def from_config(cls) -> AuthMiddleware:
        from finance_tool_universe.config import load_config
        perms = load_config().get("permissions") or {"roles": {"admin": {"tools": ["*"]}}}
        return cls(perms)

    @classmethod
    def from_allowed_tools(
        cls,
        allowed_tools_by_role: dict[str, set[str] | list[str] | tuple[str, ...]],
    ) -> AuthMiddleware:
        """Build the same matcher from an in-process tool policy.

        Local execution and MCP serving now share ``allowed``/``require``;
        callers may still provide the historical mapping at the boundary.
        """

        return cls(
            {
                "roles": {
                    str(role): {"tools": [str(tool) for tool in tools]}
                    for role, tools in allowed_tools_by_role.items()
                }
            }
        )

    def _build_role_cache(self) -> None:
        roles = self._permissions.get("roles", {})
        for role, cfg in roles.items():
            patterns = set(cfg.get("tools", []))
            inherited = cfg.get("inherits")
            if inherited and inherited in self._role_cache:
                patterns |= self._role_cache[inherited]
            self._role_cache[role] = patterns

    def allowed(self, role: str, tool_name: str) -> bool:
        patterns = self._role_cache.get(role, set())
        if "*" in patterns:
            return True
        canonical_tool = str(tool_name).replace(".", "_")
        for pat in patterns:
            canonical_pattern = str(pat).replace(".", "_")
            if canonical_pattern == canonical_tool:
                return True
            if canonical_pattern.endswith("_*") and canonical_tool.startswith(
                canonical_pattern[:-2] + "_"
            ):
                return True
        return False

    def require(self, role: str, tool_name: str) -> None:
        if not self.allowed(role, tool_name):
            raise PermissionDeniedError(role, tool_name)

    @classmethod
    def attach(cls, mcp) -> None:
        auth = cls.from_config()
        mcp._auth = auth  # type: ignore[attr-defined]
