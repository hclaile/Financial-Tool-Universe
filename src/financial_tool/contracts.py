"""Static validation for source-priority routes and connector actions."""

from __future__ import annotations

import yaml

from .connectors.base import declared_actions_for
from finance_tool_universe.config import catalog_dir

CATALOG_DIR = catalog_dir()


def validate_source_priority_actions() -> list[str]:
    """Return route/action mismatches without performing provider I/O."""
    priority = yaml.safe_load((CATALOG_DIR / "source_priority.yaml").read_text())
    contract = yaml.safe_load((CATALOG_DIR / "source_action_contract.yaml").read_text())
    actions_by_type = contract["actions"]
    errors: list[str] = []
    for rule in priority["priority_rules"]:
        data_type = rule["data_type"]
        if data_type == "*":
            continue
        actions = actions_by_type.get(data_type)
        region = rule.get("region", "GLOBAL")
        if actions is None:
            errors.append(f"{data_type}/{region}: missing action contract")
            continue
        for source in rule.get("sources") or []:
            declared = declared_actions_for(source)
            for action in actions:
                if action not in declared:
                    errors.append(f"{data_type}/{region}: {source}.{action} is not declared")
    return sorted(errors)
