"""Milestone 4.5: Live-Telemetrie für den Visual Inspector.

Die harte Anforderung lautet: Die Visualisierung darf den Modellpfad nicht
blockieren. Deshalb schreibt das Modell nur in einen Ringpuffer – ein
Listenanhang unter einem Lock, sonst nichts – und die Oberfläche holt sich die
Ereignisse in ihrem eigenen Takt ab.

Was der Puffer bewusst *nicht* tut: warten, rendern, umrechnen oder Ereignisse
zusammenfassen. Jede dieser Tätigkeiten läge im Modellthread und würde die
Inferenz ausbremsen.

Läuft der Puffer voll, werden die *ältesten* Ereignisse verworfen und gezählt.
Ein Anzeigefenster, das nicht hinterherkommt, darf die Inferenz nicht
verlangsamen; dass Daten fehlen, muss aber sichtbar sein statt verschwiegen.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Iterable, Sequence

import torch

from glassmind.observe.bus import ObservationBus, ObservationMode
from glassmind.observe.events import ObservationEvent
from glassmind.visualize.graph import ReplayTimeline, _memory_frame
from glassmind.visualize.graph import NetworkFrame


class TelemetryBuffer:
    """Nichtblockierender Ringpuffer zwischen Modell- und Anzeigethread."""

    def __init__(self, capacity: int = 20_000) -> None:
        self._events: deque[ObservationEvent] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self.received = 0
        self.dropped = 0
        self.capacity = capacity

    def __call__(self, event: ObservationEvent) -> None:
        """Sink für den ObservationBus. Läuft im Modellthread – hält sich kurz."""
        with self._lock:
            if len(self._events) == self._events.maxlen:
                self.dropped += 1
            self._events.append(event)
            self.received += 1

    def drain(self, limit: int | None = None) -> list[ObservationEvent]:
        """Holt die angesammelten Ereignisse ab. Läuft im Anzeigethread."""
        with self._lock:
            if limit is None or limit >= len(self._events):
                events = list(self._events)
                self._events.clear()
                return events
            events = [self._events.popleft() for _ in range(limit)]
        return events

    def __len__(self) -> int:
        with self._lock:
            return len(self._events)

    def close(self) -> None:  # ObservationBus ruft close auf seinen Sinks auf
        return None

    def stats(self) -> dict[str, int]:
        return {"received": self.received, "dropped": self.dropped,
                "pending": len(self), "capacity": self.capacity}


class FrameAssembler:
    """Baut aus einzelnen Ereignissen fertige Frames.

    Ein Token ist erst vollständig, wenn alle Layer ihre Telemetrie geliefert
    haben. Weil die Ereignisse in Reihenfolge kommen, gilt ein Frame als
    fertig, sobald ein Ereignis mit höherem Tokenindex eintrifft.
    """

    def __init__(self) -> None:
        self._pending: dict[int, dict[str, Any]] = {}
        self.frames: list[NetworkFrame] = []

    def _bucket(self, token_index: int, token_id: int | None) -> dict[str, Any]:
        bucket = self._pending.setdefault(token_index, {
            "token_id": token_id, "nodes": {}, "edges": {},
            "entropy": None, "memory": None, "full": {}, "summary": {},
        })
        if token_id is not None:
            bucket["token_id"] = token_id
        return bucket

    def feed(self, events: Iterable[ObservationEvent]) -> list[NetworkFrame]:
        """Übernimmt Ereignisse und gibt die dadurch fertigen Frames zurück."""
        highest = max(self._pending, default=-1)
        for event in events:
            if event.token_index is None:
                continue
            bucket = self._bucket(event.token_index, event.token_id)
            highest = max(highest, event.token_index)
            if event.event == "network_step":
                for node in event.payload.get("nodes", []):
                    bucket["nodes"][node["id"]] = node
                for edge in event.payload.get("edges", []):
                    bucket["edges"][edge["id"]] = edge
                if "full" in event.payload and event.layer_id:
                    bucket["full"][event.layer_id] = event.payload["full"]
            elif event.event == "state_summary" and event.layer_id:
                bucket["summary"][event.layer_id] = event.payload
            elif event.event == "prediction":
                bucket["entropy"] = event.payload.get("entropy")
            elif event.event == "memory_step":
                bucket["memory"] = _memory_frame(event.payload)
        completed = []
        for index in sorted(self._pending):
            if index >= highest:
                break
            data = self._pending.pop(index)
            if not (data["nodes"] or data["memory"] or data["summary"]):
                continue
            completed.append(NetworkFrame(
                token_index=index, token_id=data["token_id"],
                nodes=tuple(data["nodes"].values()),
                edges=tuple(data["edges"].values()),
                entropy=data["entropy"], memory=data["memory"],
                full=data["full"] or None, summary=data["summary"] or None,
            ))
        self.frames.extend(completed)
        return completed

    def flush(self) -> list[NetworkFrame]:
        """Schließt auch den letzten, noch offenen Frame ab."""
        remaining = sorted(self._pending)
        completed = []
        for index in remaining:
            data = self._pending.pop(index)
            if not (data["nodes"] or data["memory"] or data["summary"]):
                continue
            completed.append(NetworkFrame(
                token_index=index, token_id=data["token_id"],
                nodes=tuple(data["nodes"].values()),
                edges=tuple(data["edges"].values()),
                entropy=data["entropy"], memory=data["memory"],
                full=data["full"] or None, summary=data["summary"] or None,
            ))
        self.frames.extend(completed)
        return completed


class GrowingTimeline(ReplayTimeline):
    """Eine Zeitachse, an die im Livebetrieb hinten angehängt wird."""

    def __init__(self, frames: list[NetworkFrame] | None = None) -> None:
        self.frames = list(frames or [])

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> NetworkFrame:
        if not self.frames:
            # Vor dem ersten Token gibt es nichts zu zeigen – und es wird
            # auch nichts erfunden.
            return NetworkFrame(token_index=0, token_id=None, nodes=(), edges=())
        return self.frames[max(0, min(index, len(self.frames) - 1))]

    def append(self, frames: Sequence[NetworkFrame]) -> None:
        self.frames.extend(frames)


@dataclass
class LiveStats:
    tokens: int = 0
    seconds: float = 0.0
    events: int = 0
    dropped: int = 0

    @property
    def tokens_per_second(self) -> float:
        return self.tokens / self.seconds if self.seconds > 0 else 0.0


class LiveSession:
    """Führt Inferenz in einem Arbeitsthread aus und füllt den Puffer.

    Die Oberfläche ruft ``poll()`` in ihrem Timer auf. Alles Teure – Frames
    bauen, zeichnen – passiert dort, nicht im Modellthread.
    """

    def __init__(
        self,
        model: Any,
        prompt_ids: Sequence[int],
        *,
        max_new_tokens: int = 256,
        mode: ObservationMode | str = ObservationMode.SUMMARY,
        temperature: float = 0.8,
        device: torch.device | str = "cpu",
        capacity: int = 20_000,
        interventions: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.prompt_ids = list(prompt_ids)
        self.max_new_tokens = max_new_tokens
        self.mode = ObservationMode.parse(mode)
        self.temperature = temperature
        self.device = torch.device(device)
        self.buffer = TelemetryBuffer(capacity)
        self.assembler = FrameAssembler()
        self.timeline = GrowingTimeline()
        self.generated: list[int] = []
        self.stats = LiveStats()
        self.interventions = dict(interventions or {})
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Die Live-Sitzung läuft bereits")
        self._thread = threading.Thread(target=self._run, name="glassmind-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        try:
            bus = ObservationBus(self.mode)
            bus.subscribe(self.buffer)
            tokens = torch.tensor([self.prompt_ids], dtype=torch.long, device=self.device)
            started = time.perf_counter()
            self.model.eval()
            with torch.inference_mode():
                logits, state = self.model(tokens, observer=bus, **self.interventions)
                next_logits = logits[:, -1]
                produced = 0
                while produced < self.max_new_tokens and not self._stop.is_set():
                    if self.temperature <= 0:
                        token = next_logits.argmax(dim=-1)
                    else:
                        probabilities = torch.softmax(next_logits / self.temperature, dim=-1)
                        token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
                    self.generated.append(int(token))
                    next_logits, state = self.model.step(
                        token, state, observer=bus, **self.interventions
                    )
                    produced += 1
            self.stats.tokens = len(self.prompt_ids) + produced
            self.stats.seconds = time.perf_counter() - started
            bus.close()
        except BaseException as exc:  # im Thread darf nichts verlorengehen
            self.error = exc

    def poll(self, *, limit: int | None = None) -> list[NetworkFrame]:
        """Holt neue Ereignisse ab und hängt fertige Frames an die Zeitachse."""
        events = self.buffer.drain(limit)
        frames = self.assembler.feed(events)
        if not self.running and not self.buffer.stats()["pending"]:
            frames = frames + self.assembler.flush()
        self.timeline.append(frames)
        self.stats.events = self.buffer.received
        self.stats.dropped = self.buffer.dropped
        return frames

    def status(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "tokens": len(self.timeline),
            "generated": len(self.generated),
            "tokens_per_second": self.stats.tokens_per_second,
            "error": None if self.error is None else repr(self.error),
            **self.buffer.stats(),
        }
