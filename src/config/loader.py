from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from ..utils import deep_merge, substitute_env_vars
from .schema import ConfigSchema


class ConfigLoader:
    """Loads, merges, and validates configuration overlays.

    Supports recursive `include:` directives and environment variable
    substitution for ${VAR} patterns.
    """

    def __init__(self, overlay_path: Path | str):
        self.overlay_path = Path(overlay_path)
        if not self.overlay_path.exists():
            raise FileNotFoundError(f"Config overlay not found: {self.overlay_path}")

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"YAML root must be a mapping: {path}")
        return data

    def _collect_includes(self, root_path: Path, data: Dict[str, Any]) -> Tuple[List[Path], Dict[str, Any]]:
        includes: List[Path] = []
        include_val = data.get("include")
        if include_val is None:
            return includes, data
        if isinstance(include_val, str):
            includes = [root_path.parent / include_val]
        elif isinstance(include_val, list):
            includes = [root_path.parent / str(p) for p in include_val]
        else:
            raise ValueError("`include` must be a string or list of strings")
        # Remove include key from the returned data
        data = {k: v for k, v in data.items() if k != "include"}
        return includes, data

    def _load_with_includes(self, path: Path) -> Dict[str, Any]:
        cur = self._load_yaml(path)
        includes, cur_wo_inc = self._collect_includes(path, cur)
        merged: Dict[str, Any] = {}
        for inc in includes:
            inc_data = self._load_with_includes(inc)
            merged = deep_merge(merged, inc_data)
        merged = deep_merge(merged, cur_wo_inc)
        return merged

    def load(self) -> ConfigSchema:
        raw = self._load_with_includes(self.overlay_path)
        raw = substitute_env_vars(raw)
        # Validate and freeze
        cfg = ConfigSchema.model_validate(raw)
        return cfg

    @staticmethod
    def snapshot_json(cfg: ConfigSchema, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(cfg.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

