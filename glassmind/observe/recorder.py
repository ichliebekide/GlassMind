from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import TextIO

from glassmind.observe.events import ObservationEvent


class JSONLRecorder:
    """Gepufferter JSONL-Empfänger; jede Zeile ist unabhängig replaybar."""

    def __init__(self, path: str | Path, *, flush_every: int = 32) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle: TextIO = self.path.open("w", encoding="utf-8", buffering=64 * 1024)
        self._flush_every = flush_every
        self._pending = 0
        self._lock = Lock()

    def __call__(self, event: ObservationEvent) -> None:
        line = json.dumps(event.to_dict(), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        with self._lock:
            self._handle.write(line + "\n")
            self._pending += 1
            if self._pending >= self._flush_every:
                self._handle.flush()
                self._pending = 0

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()

    def __enter__(self) -> "JSONLRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

