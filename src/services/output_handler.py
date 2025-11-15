from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
import pandas as pd

from ..models import ExtractionResult
from ..utils import ensure_dir


class OutputHandler:
    def __init__(
        self,
        base_dir: Path | str,
        *,
        session_id: Optional[str] = None,
        pipeline_name: Optional[str] = None,
        aggregate_csv: Optional[Path | str] = None
    ) -> None:
        self.base_dir = Path(base_dir)
        self.session_id = session_id or "default"
        self.pipeline_name = pipeline_name or "extraction"

        # Create session-based directory structure: base_dir/sessions/{pipeline}/{session_id}/
        self.session_dir = self.base_dir / "sessions" / self.pipeline_name / self.session_id
        self.records_dir = self.session_dir / "records"
        ensure_dir(self.records_dir)

        # Set aggregate CSV to session-specific file if not provided
        if aggregate_csv:
            self.aggregate_csv = Path(aggregate_csv)
        else:
            self.aggregate_csv = self.session_dir / "results.csv"

    def write_record_output(self, result: ExtractionResult) -> Path:
        # Sanitize filename by replacing slashes with underscores to avoid directory issues
        safe_filename = str(result.key).replace("/", "_").replace("\\", "_")
        out_path = self.records_dir / f"{safe_filename}.yaml"
        payload = {
            "key": result.key,
            "input": result.input,
            "extracted": result.extracted,
            "normalized": result.normalized,
            "confidence": result.confidence.model_dump(),
            "valid": result.valid,
            "errors": result.errors,
            "transparency": result.transparency.model_dump(),
            "created_at": result.created_at.isoformat(),
        }
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
        return out_path

    def update_csv_aggregate(self, result: ExtractionResult) -> Optional[Path]:
        if not self.aggregate_csv:
            return None
        row: Dict[str, Any] = {"key": result.key, **result.normalized, "confidence": result.confidence.overall}
        path = self.aggregate_csv
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            df = pd.read_csv(path)
            # Upsert by key
            df = df[df["key"] != result.key]
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(path, index=False)
        return path

