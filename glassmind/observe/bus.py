from __future__ import annotations

from enum import IntEnum
from typing import Callable, Protocol

from glassmind.observe.events import ObservationEvent
from glassmind.observe.activity import ActivityTracker


class ObservationMode(IntEnum):
    OFF = 0
    SUMMARY = 1
    TRACE = 2
    FULL = 3

    @classmethod
    def parse(cls, value: str | "ObservationMode") -> "ObservationMode":
        if isinstance(value, cls):
            return value
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            allowed = ", ".join(mode.name.lower() for mode in cls)
            raise ValueError(f"Unbekannter Beobachtungsmodus {value!r}; erlaubt: {allowed}") from exc


class EventSink(Protocol):
    def __call__(self, event: ObservationEvent) -> None: ...


class ObservationBus:
    """Synchroner, bewusst kleiner Verteiler für kompakte Telemetrieereignisse."""

    def __init__(self, mode: str | ObservationMode = ObservationMode.OFF) -> None:
        self.mode = ObservationMode.parse(mode)
        self._sinks: list[EventSink] = []
        self.events_emitted = 0
        self.activity_tracker = ActivityTracker()

    @property
    def enabled(self) -> bool:
        return self.mode is not ObservationMode.OFF

    @property
    def traces_tokens(self) -> bool:
        return self.mode >= ObservationMode.TRACE

    def subscribe(self, sink: EventSink) -> Callable[[], None]:
        self._sinks.append(sink)

        def unsubscribe() -> None:
            if sink in self._sinks:
                self._sinks.remove(sink)

        return unsubscribe

    def emit(self, event: ObservationEvent) -> None:
        if not self.enabled:
            return
        self.events_emitted += 1
        for sink in tuple(self._sinks):
            sink(event)

    def annotate_cluster_activity(
        self,
        nodes: list[dict[str, object]],
        *,
        token_id: int,
        update_threshold: float,
    ) -> None:
        if self.traces_tokens:
            self.activity_tracker.annotate(
                nodes,
                token_id=token_id,
                update_threshold=update_threshold,
            )

    def close(self) -> None:
        for sink in tuple(self._sinks):
            close = getattr(sink, "close", None)
            if close is not None:
                close()


OFF_BUS = ObservationBus(ObservationMode.OFF)
