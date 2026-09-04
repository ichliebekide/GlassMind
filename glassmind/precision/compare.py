"""Vergleicht die Telemetrie zweier Precision-Varianten über den Observation Bus.

Der Bus liefert für jeden sichtbaren Knoten reale Messwerte. Werden zwei
Varianten desselben Modells beobachtet, lässt sich exakt beziffern, wie stark
sich die *sichtbare* Aktivität durch eine Präzisionsänderung verschiebt – ohne
irgendetwas zu erfinden. Fehlende Knoten werden als fehlend gemeldet, nicht
stillschweigend übergangen.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import Tensor

from glassmind.observe.bus import ObservationBus, ObservationMode
from glassmind.observe.events import ObservationEvent

if TYPE_CHECKING:  # Nur für Typprüfung – zur Laufzeit gäbe es einen Ringschluss.
    from glassmind.model.lm import GlassMindLM

#: Knotenwerte, deren Verschiebung gemessen wird.
NODE_FIELDS = ("activity", "delta_rms", "information_flow", "persistence_duration", "reactivation")


@dataclass
class NodeDeviation:
    node_id: str
    kind: str
    samples: int
    activity: float
    delta: float
    flow: float
    persistence: float
    reactivation: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "samples": self.samples,
            "activity_deviation": self.activity,
            "delta_deviation": self.delta,
            "flow_deviation": self.flow,
            "persistence_deviation": self.persistence,
            "reactivation_deviation": self.reactivation,
        }


class _TraceCollector:
    """Sammelt Knoten- und Kantenwerte je Token und Knoten-ID."""

    def __init__(self) -> None:
        self.nodes: dict[tuple[int, str], dict[str, float]] = {}
        self.edges: dict[tuple[int, str], float] = {}

    def __call__(self, event: ObservationEvent) -> None:
        if event.event != "network_step":
            return
        token = int(event.token_index or 0)
        for node in event.payload.get("nodes", []):
            components = node.get("components", {})
            self.nodes[(token, str(node["id"]))] = {
                "kind": str(node.get("kind", "")),
                "activity": float(node.get("activity", 0.0)),
                "delta_rms": float(components.get("delta_rms", 0.0)),
                "information_flow": float(components.get("information_flow", 0.0)),
                "persistence_duration": float(components.get("persistence_duration", 0.0)),
                "reactivation": float(bool(components.get("reactivation", False))),
            }
        for edge in event.payload.get("edges", []):
            self.edges[(token, str(edge["id"]))] = float(edge.get("flow", 0.0))


@torch.inference_mode()
def collect_trace(model: GlassMindLM, tokens: Tensor) -> _TraceCollector:
    collector = _TraceCollector()
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(collector)
    model.eval()
    model(tokens, observer=bus)
    bus.close()
    return collector


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compare_telemetry(
    reference_model: GlassMindLM, test_model: GlassMindLM, tokens: Tensor
) -> dict[str, Any]:
    """Misst die Abweichung sichtbarer Aktivität zwischen zwei Varianten.

    Alle Werte sind mittlere absolute Abweichungen über alle Token. Sie sind
    echte Messwerte aus dem Observation Bus, keine Schätzungen.
    """
    reference = collect_trace(reference_model, tokens)
    test = collect_trace(test_model, tokens)

    shared = reference.nodes.keys() & test.nodes.keys()
    missing = (reference.nodes.keys() | test.nodes.keys()) - shared
    per_node: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    kinds: dict[str, str] = {}
    for key in shared:
        _, node_id = key
        a, b = reference.nodes[key], test.nodes[key]
        kinds[node_id] = str(a["kind"])
        for field in NODE_FIELDS:
            per_node[node_id][field].append(abs(float(b[field]) - float(a[field])))

    deviations = [
        NodeDeviation(
            node_id=node_id,
            kind=kinds.get(node_id, ""),
            samples=len(values["activity"]),
            activity=_mean(values["activity"]),
            delta=_mean(values["delta_rms"]),
            flow=_mean(values["information_flow"]),
            persistence=_mean(values["persistence_duration"]),
            reactivation=_mean(values["reactivation"]),
        )
        for node_id, values in sorted(per_node.items())
    ]

    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({item.kind for item in deviations if item.kind}):
        group = [item for item in deviations if item.kind == kind]
        by_kind[kind] = {
            "activity_deviation": _mean([item.activity for item in group]),
            "delta_deviation": _mean([item.delta for item in group]),
            "flow_deviation": _mean([item.flow for item in group]),
            "persistence_deviation": _mean([item.persistence for item in group]),
            "reactivation_deviation": _mean([item.reactivation for item in group]),
            "nodes": len(group),
        }

    shared_edges = reference.edges.keys() & test.edges.keys()
    edge_deviation = _mean(
        [abs(test.edges[key] - reference.edges[key]) for key in shared_edges]
    )
    return {
        "compared_node_samples": len(shared),
        "missing_node_samples": len(missing),
        "edge_flow_deviation": edge_deviation,
        "by_state": by_kind,
        "worst_nodes": [
            item.to_dict()
            for item in sorted(deviations, key=lambda item: item.activity, reverse=True)[:8]
        ],
    }


def format_telemetry_comparison(result: dict[str, Any]) -> str:
    lines = [
        f"{'Zustand':<10s} {'Aktivität':>12s} {'Delta':>12s} {'Fluss':>12s} "
        f"{'Persistenz':>12s} {'Reaktiv.':>10s}"
    ]
    lines.append("-" * len(lines[0]))
    for kind, values in result.get("by_state", {}).items():
        lines.append(
            f"{kind:<10s} {values['activity_deviation']:12.3e} {values['delta_deviation']:12.3e} "
            f"{values['flow_deviation']:12.3e} {values['persistence_deviation']:12.3e} "
            f"{values['reactivation_deviation']:10.3e}"
        )
    lines.append(f"Kantenfluss-Abweichung: {result.get('edge_flow_deviation', 0.0):.3e}")
    if result.get("missing_node_samples"):
        lines.append(
            f"Nicht vergleichbare Knotenmessungen: {result['missing_node_samples']} "
            "(in nur einer der beiden Varianten vorhanden)"
        )
    return "\n".join(lines)
