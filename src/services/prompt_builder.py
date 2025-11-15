from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from jinja2 import Environment, BaseLoader


class PromptBuilder:
    """Renders system and user prompts using Jinja2 templates."""

    def __init__(
        self,
        *,
        system_template: Optional[str] = None,
        user_template: Optional[str] = None,
        variables: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.env = Environment(loader=BaseLoader(), autoescape=False, trim_blocks=True, lstrip_blocks=True)
        self.system_template = system_template or ""
        self.user_template = user_template or ""
        self.variables = variables or {}

        self._compiled_system = self.env.from_string(self.system_template)
        self._compiled_user = self.env.from_string(self.user_template)

    def build(self, record_data: Dict[str, Any]) -> Tuple[str, str]:
        ctx = {**self.variables, **record_data}
        system = self._compiled_system.render(**ctx) if self.system_template else ""
        user = self._compiled_user.render(**ctx) if self.user_template else ""
        # Support legacy `{var}` placeholders used in existing prompt files
        try:
            if system:
                system = system.format(**ctx)
            if user:
                user = user.format(**ctx)
        except Exception:
            # Silently ignore formatting errors; Jinja is primary renderer
            pass
        return system.strip(), user.strip()
