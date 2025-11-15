from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Optional

import pandas as pd

from ..models import Record


class InputHandler:
    """Reads tabular input (CSV/Excel) and yields normalized records."""

    def __init__(
        self,
        path: Path | str,
        *,
        id_column: str = "id",
        column_map: Optional[Dict[str, str]] = None,
        skip: int = 0,
        limit: Optional[int] = None,
    ) -> None:
        self.path = Path(path)
        self.id_column = id_column
        self.column_map = column_map or {}
        self.skip = max(0, skip)
        self.limit = limit

        if not self.path.exists():
            raise FileNotFoundError(f"Input file not found: {self.path}")

    def _read(self) -> pd.DataFrame:
        if self.path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(self.path)
        return pd.read_csv(self.path)

    def iter_records(self) -> Generator[Record, None, None]:
        df = self._read()
        if self.skip:
            df = df.iloc[self.skip :]
        if self.limit is not None:
            df = df.iloc[: self.limit]

        # Apply simple column mapping
        df = df.rename(columns=self.column_map)

        for _, row in df.iterrows():
            data: Dict[str, Any] = row.to_dict()
            key = str(data.get(self.id_column) or data.get("DOI") or data.get("id") or _)
            yield Record(key=key, data=data)

