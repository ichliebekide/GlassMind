from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Any


@dataclass(frozen=True)
class ObservationEvent:
    event: str
    step: int
    token_index: int | None = None
    token_id: int | None = None
    layer_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 2
    wall_time: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObservationEvent":
        return cls(**data)
