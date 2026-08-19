"""이벤트 저널 — 3개월 뒤 "무슨 일이 있었나"를 답할 수 있게 전부 남긴다.

JSONL(한 줄에 JSON 하나) 형식이라 중간에 프로세스가 죽어도 앞부분은 온전하고,
나중에 pandas 로 바로 읽어 분석할 수 있다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class Journal:
    """추가 전용(append-only) 이벤트 로그."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event": event,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue  # 크래시로 잘린 마지막 줄은 건너뛴다

    def to_frame(self):
        import pandas as pd

        rows = list(self.read())
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["ts", "event"])
