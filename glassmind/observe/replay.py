from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from glassmind.observe.events import ObservationEvent


def iter_events(path: str | Path) -> Iterator[ObservationEvent]:
    trace = Path(path)
    with trace.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield ObservationEvent.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError) as exc:
                raise ValueError(f"Ungültiges Trace-Ereignis in {trace}, Zeile {line_number}") from exc


def load_network_frames(path: str | Path) -> list[ObservationEvent]:
    frames = [event for event in iter_events(path) if event.event == "network_step"]
    if not frames:
        raise ValueError(f"Trace {path} enthält keine network_step-Ereignisse")
    return frames
