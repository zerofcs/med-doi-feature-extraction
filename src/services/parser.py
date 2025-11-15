from __future__ import annotations

from typing import Any, Dict, Tuple

from ..utils import parse_json_from_text, json_pointer_get, strip_markdown_fences


class Parser:
    """Parses and normalizes LLM output into a flat field dict."""

    def __init__(
        self,
        *,
        pointer_map: Dict[str, str] | None = None,
        normalization: Dict[str, Dict[str, Any]] | None = None,
    ) -> None:
        self.pointer_map = pointer_map or {}
        self.normalization = normalization or {}

    def parse_and_normalize(self, text: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        raw_obj = parse_json_from_text(text, try_direct_parse=True)
        extracted: Dict[str, Any] = {}
        if not raw_obj:
            return extracted, {}

        if not self.pointer_map:
            # assume direct mapping
            extracted = {k: v for k, v in raw_obj.items() if not isinstance(v, (dict, list))}
        else:
            for field, pointer in self.pointer_map.items():
                try:
                    extracted[field] = json_pointer_get(raw_obj, pointer)
                except Exception:
                    pass

        normalized: Dict[str, Any] = dict(extracted)

        # Apply simple normalization choices with synonyms mapping
        for field, rules in self.normalization.items():
            if field not in normalized:
                continue
            value = normalized[field]
            if not isinstance(value, str):
                continue
            target = value.strip()
            # case folding
            lower = target.lower()
            if "choices" in rules:
                # canonical list
                choices = rules.get("choices", [])
                # direct match first
                for c in choices:
                    if c.lower() == lower:
                        target = c
                        break
                else:
                    # try synonyms
                    syn_map = rules.get("synonyms", {})
                    for canon, synonyms in syn_map.items():
                        if isinstance(synonyms, list) and any(s.lower() == lower for s in synonyms):
                            target = canon
                            break
            normalized[field] = target

        return extracted, normalized

