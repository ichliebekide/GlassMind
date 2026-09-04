from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from glassmind.observe.replay import iter_events


@dataclass(frozen=True)
class MemoryFrame:
    """Ein Speicherzugriff, vollständig aus aufgezeichneter Telemetrie."""

    slots: int
    read_slots: tuple[int, ...]
    read_scores: tuple[float, ...]
    read_weights: tuple[float, ...]
    read_gate: float
    read_output_rms: float
    write_slots: tuple[int, ...]
    write_strength: tuple[float, ...]
    write_gate: float
    replacement_events: tuple[int, ...]
    strength: tuple[float, ...]
    age: tuple[float, ...]
    usage: tuple[float, ...]
    read_count: tuple[float, ...]
    write_count: tuple[float, ...]
    occupied: tuple[float, ...]
    source_state: str = ""
    destination_state: str = ""

    def slot_detail(self, slot: int, token_index: int) -> dict[str, Any]:
        """Werte eines einzelnen Slots – ohne jede Deutung.

        Es wird ausdrücklich keine menschliche Bedeutung behauptet; die Zahlen
        sind gemessene Zugriffsstatistiken.
        """
        if not 0 <= slot < self.slots:
            raise ValueError(f"Slot {slot} liegt außerhalb von 0..{self.slots - 1}")
        return {
            "slot": slot,
            "strength": self.strength[slot],
            "age": self.age[slot],
            "reads": self.read_count[slot],
            "writes": self.write_count[slot],
            "usage": self.usage[slot],
            "occupied": self.occupied[slot] > 0.5,
            "active_read": slot in self.read_slots,
            "active_write": slot in self.write_slots,
            "replaced": slot in self.replacement_events,
            "current_score": (
                self.read_scores[self.read_slots.index(slot)] if slot in self.read_slots else None
            ),
            "token_index": token_index,
        }


@dataclass(frozen=True)
class NetworkFrame:
    token_index: int
    token_id: int | None
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]
    entropy: float | None = None
    memory: MemoryFrame | None = None
    #: Vollständige Zustandsvektoren je Layer und State. Nur im Modus ``full``
    #: vorhanden; sonst ``None``. Der Visual Inspector zeigt die Unit-Stufe
    #: ausschließlich dann, wenn dieses Feld belegt ist.
    full: dict[str, dict[str, list[float]]] | None = None
    #: Kennzahlen aus ``state_summary``. Sie kommen bereits im Modus
    #: ``summary`` an und tragen die Layer-/State-Stufe, wenn gar keine
    #: Clusterdaten aufgezeichnet wurden.
    summary: dict[str, dict[str, Any]] | None = None


class ReplayTimeline:
    def __init__(self, frames: list[NetworkFrame]) -> None:
        if not frames:
            raise ValueError("Ein Replay benötigt mindestens einen Netzwerk-Frame")
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> NetworkFrame:
        return self.frames[index]

    @classmethod
    def from_trace(cls, path: str | Path) -> "ReplayTimeline":
        grouped: dict[int, dict[str, Any]] = {}
        for event in iter_events(path):
            if event.token_index is None:
                continue
            frame = grouped.setdefault(
                event.token_index,
                {
                    "token_id": event.token_id,
                    "nodes": {},
                    "edges": {},
                    "entropy": None,
                    "memory": None,
                    "full": {},
                    "summary": {},
                },
            )
            if event.token_id is not None:
                frame["token_id"] = event.token_id
            if event.event == "network_step":
                for node in event.payload.get("nodes", []):
                    frame["nodes"][node["id"]] = node
                for edge in event.payload.get("edges", []):
                    frame["edges"][edge["id"]] = edge
                if "full" in event.payload and event.layer_id:
                    frame["full"][event.layer_id] = event.payload["full"]
            elif event.event == "state_summary" and event.layer_id:
                frame["summary"][event.layer_id] = event.payload
            elif event.event == "prediction":
                frame["entropy"] = event.payload.get("entropy")
            elif event.event == "memory_step":
                frame["memory"] = _memory_frame(event.payload)
        frames = [
            NetworkFrame(
                token_index=index,
                token_id=data["token_id"],
                nodes=tuple(data["nodes"].values()),
                edges=tuple(data["edges"].values()),
                entropy=data["entropy"],
                memory=data["memory"],
                full=data["full"] or None,
                summary=data["summary"] or None,
            )
            for index, data in sorted(grouped.items())
            if data["nodes"] or data["memory"] or data["summary"]
        ]
        return cls(frames)


def _memory_frame(payload: dict[str, Any]) -> MemoryFrame:
    number = lambda key: tuple(float(value) for value in payload.get(key, ()))
    integer = lambda key: tuple(int(value) for value in payload.get(key, ()))
    return MemoryFrame(
        slots=int(payload.get("slots", 0)),
        read_slots=integer("selected_read_slots"),
        read_scores=number("read_scores"),
        read_weights=number("read_weights"),
        read_gate=float(payload.get("read_gate", 0.0)),
        read_output_rms=float(payload.get("read_output_rms", 0.0)),
        write_slots=integer("selected_write_slots"),
        write_strength=number("write_strength"),
        write_gate=float(payload.get("write_gate", 0.0)),
        replacement_events=integer("replacement_events"),
        strength=number("slot_strength"),
        age=number("slot_age"),
        usage=number("slot_usage"),
        read_count=number("slot_read_count"),
        write_count=number("slot_write_count"),
        occupied=number("slot_occupied"),
        source_state=str(payload.get("source_state", "")),
        destination_state=str(payload.get("destination_state", "")),
    )


def memory_arrays(frame: MemoryFrame | None, *, columns: int = 16) -> dict[str, Any]:
    """Messwerte der Speicherbank je Zelle – ohne Positionen.

    Die räumliche Anordnung liegt allein in ``glassmind.visualize.layout``.
    Zwei getrennte Rasterberechnungen hatten dazu geführt, dass Kamera und
    Zeichnung die Bank an verschiedenen Stellen erwarteten.
    """
    if frame is None or frame.slots <= 0:
        return {"ids": [], "slots": 0}
    read_set, write_set = set(frame.read_slots), set(frame.write_slots)
    replaced = set(frame.replacement_events)
    max_age = max(frame.age) if frame.age else 1.0
    return {
        "slots": frame.slots,
        "ids": [f"memory.slot.{index}" for index in range(frame.slots)],
        "strength": list(frame.strength),
        "age": list(frame.age),
        # Normiertes Alter nur für die Darstellung; die Rohwerte stehen daneben.
        "age_normalised": [value / max(max_age, 1e-9) for value in frame.age],
        "occupied": [value > 0.5 for value in frame.occupied],
        "read_active": [index in read_set for index in range(frame.slots)],
        "write_active": [index in write_set for index in range(frame.slots)],
        "replaced": [index in replaced for index in range(frame.slots)],
        "read_count": list(frame.read_count),
        "write_count": list(frame.write_count),
        "read_weight_by_slot": {
            slot: weight for slot, weight in zip(frame.read_slots, frame.read_weights)
        },
        "read_gate": frame.read_gate,
        "write_gate": frame.write_gate,
        "read_output_rms": frame.read_output_rms,
        "source_state": frame.source_state,
        "destination_state": frame.destination_state,
    }


def stable_position(node_id: str, *, cluster_count: int = 8) -> tuple[float, float]:
    parts = node_id.split(".")
    try:
        layer = int(parts[1])
    except (IndexError, ValueError):
        layer = 0
    kind = parts[2] if len(parts) > 2 else "other"
    x_by_kind = {"input": 0.0, "fast": 1.4, "context": 2.8, "semantic": 4.2, "output": 5.6}
    x = x_by_kind.get(kind, 2.8) + layer * 6.8
    if "cluster" in parts:
        cluster = int(parts[-1])
        y = (cluster - (cluster_count - 1) / 2) * 0.72
    else:
        y = 0.0
    return x, y


def graph_arrays(frame: NetworkFrame, history: dict[str, float] | None = None) -> dict[str, Any]:
    history = history or {}
    node_by_id = {node["id"]: node for node in frame.nodes}
    ids = sorted(node_by_id)
    cluster_count = max(
        (int(node_id.rsplit(".", 1)[-1]) + 1 for node_id in ids if ".cluster." in node_id),
        default=1,
    )
    positions = [stable_position(node_id, cluster_count=cluster_count) for node_id in ids]
    activities = [float(node_by_id[node_id].get("activity", 0.0)) for node_id in ids]
    kinds = [str(node_by_id[node_id].get("kind", "other")) for node_id in ids]
    deltas = [
        float(node_by_id[node_id].get("components", {}).get("delta_rms", 0.0))
        for node_id in ids
    ]
    persistence = [int(node_by_id[node_id].get("persistence", 0)) for node_id in ids]
    reactivations = [bool(node_by_id[node_id].get("reactivation", False)) for node_id in ids]
    segments: list[tuple[tuple[float, float], tuple[float, float], float]] = []
    position_by_id = dict(zip(ids, positions, strict=True))
    for edge in frame.edges:
        if edge["source"] in position_by_id and edge["target"] in position_by_id:
            segments.append((position_by_id[edge["source"]], position_by_id[edge["target"]], float(edge.get("flow", 0.0))))
    return {
        "ids": ids,
        "positions": positions,
        "activities": activities,
        "kinds": kinds,
        "deltas": deltas,
        "persistence": persistence,
        "reactivations": reactivations,
        "nodes": [node_by_id[node_id] for node_id in ids],
        "segments": segments,
        "history": history,
    }
