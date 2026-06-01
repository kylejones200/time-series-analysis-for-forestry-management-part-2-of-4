from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from forestry_ts.paths import DEFAULT_CONFIG_PATH, PROJECT_ROOT

LOCAL_CONFIG_PATH = PROJECT_ROOT / "config.local.yaml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        return {}
    with config_path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    if path is None and LOCAL_CONFIG_PATH.is_file():
        with LOCAL_CONFIG_PATH.open(encoding="utf-8") as handle:
            local = yaml.safe_load(handle) or {}
        if local:
            cfg = _deep_merge(cfg, local)

    return cfg


def configure_logging(cfg: dict[str, Any]) -> None:
    level_name = (cfg.get("logging") or {}).get("level", "INFO")
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
