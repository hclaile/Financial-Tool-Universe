"""Audit logging — records tool invocations for compliance."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("financial_tool.audit")


class AuditLogger:
    def __init__(self, audit_dir: Path | None = None) -> None:
        from finance_tool_universe.config import configured_path

        self._dir = audit_dir or configured_path(None, "logs")
        self._dir.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        tool: str,
        user: str,
        inputs: dict[str, Any],
        source: str,
        status: str,
        latency_ms: float,
    ) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool,
            "user": user,
            "inputs": inputs,
            "source": source,
            "status": status,
            "latency_ms": round(latency_ms, 2),
        }
        logger.info(json.dumps(entry))
