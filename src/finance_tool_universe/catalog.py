"""One validated catalog for public tool metadata and groups."""

from __future__ import annotations

import json
from typing import Any

from .config import catalog_dir

_TOOL_KEYS = {"name", "description", "group", "input_schema", "output_schema", "example"}
_TOP_KEYS = {"tools", "groups", "replacements"}


def load_catalog() -> dict[str, Any]:
    path = catalog_dir() / "tool_catalog.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != _TOP_KEYS:
        raise ValueError("tool_catalog.json must contain tools, groups and replacements")
    tools = data["tools"]
    groups = data["groups"]
    replacements = data["replacements"]
    if not isinstance(tools, list) or not isinstance(groups, dict) or not isinstance(replacements, dict):
        raise ValueError("Invalid tool_catalog.json section types")
    if any(not isinstance(tool, dict) or set(tool) != _TOOL_KEYS for tool in tools):
        raise ValueError("Every catalog tool must use the same six fields")
    names = [tool["name"] for tool in tools]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate public tool name in tool_catalog.json")
    if any(not isinstance(tool["group"], str) or tool["group"] not in groups for tool in tools):
        raise ValueError("Every tool must reference a defined group")
    if any(not isinstance(group, dict) or set(group) != {"description", "default_source"} for group in groups.values()):
        raise ValueError("Every group must define description and default_source")
    if any(not any(tool["group"] == group for tool in tools) for group in groups):
        raise ValueError("Empty group in tool_catalog.json")
    public = set(names)
    if any(old in public or not isinstance(targets, list) or not targets or not set(targets) <= public for old, targets in replacements.items()):
        raise ValueError("Retired tool mappings must point to public tools")
    return data


def tools() -> list[dict[str, Any]]:
    return load_catalog()["tools"]


def groups() -> list[dict[str, Any]]:
    data = load_catalog()
    return [
        {
            "name": name,
            "description": metadata["description"],
            "tools": [tool["name"] for tool in data["tools"] if tool["group"] == name],
            "default_source": metadata["default_source"],
        }
        for name, metadata in data["groups"].items()
    ]
