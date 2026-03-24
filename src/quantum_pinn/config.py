from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def resolve_framework_config(config: dict[str, Any], framework: str) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    resolved["training"]["framework"] = framework
    framework_overrides = resolved.get("framework_overrides", {})
    if framework in framework_overrides:
        deep_update(resolved, framework_overrides[framework])
    return resolved
