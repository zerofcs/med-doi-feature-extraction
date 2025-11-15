from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

from ..models import SessionSummary
from ..utils import ensure_dir


class AuditService:
    """Session-scoped audit logging service writing JSONL events and snapshots."""

    def __init__(self, base_dir: Path | str, session_id: Optional[str] = None) -> None:
        self.session_id = session_id or uuid.uuid4().hex[:12]
        self.base_dir = Path(base_dir)
        ensure_dir(self.base_dir)

        self.events_path = self.base_dir / f"events_{self.session_id}.jsonl"
        self.llm_path = self.base_dir / f"llm_{self.session_id}.jsonl"
        self.failures_path = self.base_dir / f"failures_{self.session_id}.jsonl"
        self.summary_path = self.base_dir / f"summary_{self.session_id}.json"
        self.config_snapshot_path = self.base_dir / f"config_{self.session_id}.json"

        self.summary = SessionSummary(
            session_id=self.session_id,
            started_at=datetime.utcnow(),
        )

    def _append_jsonl(self, path: Path, obj: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def log_event(self, type_: str, **data: Any) -> None:
        evt = {
            "ts": datetime.utcnow().isoformat(),
            "session": self.session_id,
            "type": type_,
            **data,
        }
        self._append_jsonl(self.events_path, evt)

    def log_llm_interaction(self, **data: Any) -> None:
        self._append_jsonl(self.llm_path, {"ts": datetime.utcnow().isoformat(), **data})

    def log_failure(self, **data: Any) -> None:
        self.summary.failed += 1
        self._append_jsonl(self.failures_path, {"ts": datetime.utcnow().isoformat(), **data})

    def increment(self, *, total: int = 0, succeeded: int = 0, skipped: int = 0, cost: float = 0.0) -> None:
        self.summary.total += total
        self.summary.succeeded += succeeded
        self.summary.skipped += skipped
        self.summary.cost_total += cost

    def finalize_session(self) -> None:
        self.summary.completed_at = datetime.utcnow()
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(self.summary.model_dump(mode="json"), f, indent=2)

    def snapshot_config(self, cfg_json: Dict[str, Any]) -> None:
        self.config_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_snapshot_path.open("w", encoding="utf-8") as f:
            json.dump(cfg_json, f, indent=2, ensure_ascii=False)

