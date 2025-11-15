from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Union


def deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge two dict-like mappings. Values in overlay win.

    - Dict vs dict: merge recursively
    - List vs list: overlay replaces base entirely (simple and predictable)
    - Other types: overlay replaces base
    """
    result: Dict[str, Any] = dict(base)
    for k, v in overlay.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _deep_update(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    """Compatibility alias for recursive dict merge where override wins.

    Mirrors deep_merge() behavior to align with V1 terminology.
    """
    return deep_merge(base, override)


_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} in strings using environment variables.

    If an environment variable is missing, leave the pattern unchanged
    to allow upstream validation to catch it.
    """
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            name = match.group(1)
            return os.environ.get(name, match.group(0))

        return _ENV_PATTERN.sub(repl, value)
    if isinstance(value, list):
        return [substitute_env_vars(v) for v in value]
    if isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}
    return value


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def strip_markdown_fences(text: str) -> str:
    """Remove common markdown code block fences from a string."""
    if not isinstance(text, str):
        return text
    # Remove triple-backtick fenced code blocks optionally with language
    text = re.sub(r"^\s*```[a-zA-Z0-9]*\s*\n|\n\s*```\s*$", "", text, flags=re.MULTILINE)
    # Remove JSON block hints
    text = re.sub(r"^\s*```json\s*\n|\n\s*```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_json_from_text(text: str, *, try_direct_parse: bool = True) -> Dict[str, Any]:
    """Best-effort JSON extraction from an LLM response string.

    - Optionally attempts direct json.loads
    - Otherwise searches for first {...} or [...] block and parses
    Returns an empty dict on failure.
    """
    if not text:
        return {}
    cleaned = strip_markdown_fences(text)

    if try_direct_parse:
        try:
            return json.loads(cleaned)
        except Exception:
            pass

    # Find first JSON-like object/array
    match = re.search(r"(\{.*\}|\[.*\])", cleaned, flags=re.DOTALL)
    if match:
        candidate = match.group(1)
        try:
            return json.loads(candidate)
        except Exception:
            return {}
    return {}


def json_pointer_get(doc: Any, pointer: str) -> Any:
    """Minimal JSON Pointer resolver (RFC 6901 subset).

    Supports paths like '/a/b/0'. If pointer is empty or '/', returns doc.
    """
    if pointer in ("", "/"):
        return doc
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with '/'")
    parts = [p.replace("~1", "/").replace("~0", "~") for p in pointer.split("/")[1:]]
    cur = doc
    for part in parts:
        if isinstance(cur, list):
            try:
                idx = int(part)
            except ValueError:
                raise KeyError(f"Expected list index at '{part}'")
            cur = cur[idx]
        elif isinstance(cur, dict):
            if part not in cur:
                raise KeyError(part)
            cur = cur[part]
        else:
            raise KeyError(part)
    return cur


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value
