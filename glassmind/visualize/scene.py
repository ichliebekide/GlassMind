"""Milestone 4.5: Szenenmodell des Visual Inspector.

Dieses Modul enthält die gesamte Darstellungs*logik* – ohne VisPy, ohne
Fenster, ohne Grafikkontext. Damit ist sie testbar, und das Rendering bleibt
eine dünne Schicht darüber.

Der zentrale Gedanke ist die Detailstufe (Level of Detail). Ein 90M-Modell hat
zu viele Einheiten, um sie einzeln zu zeichnen; also wird aggregiert. Wichtig
ist dabei die Ehrlichkeitsregel: Aggregiert werden darf nur, was tatsächlich
gemessen wurde. Welche Stufe überhaupt zur Verfügung steht, hängt vom
Observation-Modus ab, mit dem der Trace entstanden ist:

``summary``  ``state_summary`` – Kennzahlen je Layer und State, keine Cluster
``trace``    ``network_step``  – Clusterknoten und echte Flusskanten
``full``     zusätzlich vollständige Zustandsvektoren – einzelne Units

Eine Stufe, für die keine Daten vorliegen, wird nicht geschätzt, sondern
gemeldet. Die GUI zeigt dann die tiefste Stufe, die belegt ist.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
import math
from typing import Any, Iterable, Sequence


class DetailLevel(IntEnum):
    """Detailstufen von grob nach fein."""

    MODEL = 0
    LAYER = 1
    STATE = 2
    CLUSTER = 3
    UNIT = 4

    @property
    def label(self) -> str:
        return {
            DetailLevel.MODEL: "Modell",
            DetailLevel.LAYER: "Layer",
            DetailLevel.STATE: "State-Region",
            DetailLevel.CLUSTER: "Cluster",
            DetailLevel.UNIT: "Units",
        }[self]

    @classmethod
    def parse(cls, value: "str | DetailLevel") -> "DetailLevel":
        if isinstance(value, cls):
            return value
        try:
            return cls[str(value).strip().upper()]
        except KeyError as exc:
            allowed = ", ".join(level.name.lower() for level in cls)
            raise ValueError(f"Unbekannte Detailstufe {value!r}; erlaubt: {allowed}") from exc


#: Die drei Zeitskalen plus die beiden Randknoten und der Speicher. Die
#: Reihenfolge bestimmt auch die Anordnung von links nach rechts.
STATE_KINDS = ("input", "fast", "context", "semantic", "output")
ALL_KINDS = (*STATE_KINDS, "memory")


@dataclass(frozen=True)
class SceneNode:
    """Ein zeichenbarer Knoten – jeder Wert stammt aus gemessener Telemetrie."""

    id: str
    kind: str
    level: DetailLevel
    layer: int
    #: Cluster- bzw. Unit-Index, sofern die Stufe einen kennt.
    index: int | None = None
    #: Wie viele gemessene Elemente in diesem Knoten zusammengefasst sind.
    members: int = 1
    activity: float = 0.0
    delta: float = 0.0
    state_norm: float = 0.0
    incoming_flow: float = 0.0
    outgoing_flow: float = 0.0
    persistence: int = 0
    reactivation: bool = False
    gate: float = 0.0
    components: dict[str, Any] = field(default_factory=dict)

    def detail(self) -> dict[str, Any]:
        """Die Detailtafel für die Auswahl – ausschließlich Messwerte.

        Es wird bewusst keine Deutung erzeugt. Was hier steht, ist gemessen;
        was es bedeutet, sagt der Inspector nicht.
        """
        return {
            "id": self.id,
            "Ebene": self.level.label,
            "Layer": self.layer,
            "State": self.kind,
            "Index": "–" if self.index is None else self.index,
            "zusammengefasst": self.members,
            "Activation": round(self.activity, 4),
            "Delta": round(self.delta, 4),
            "State-Norm": round(self.state_norm, 4),
            "Persistence": self.persistence,
            "Reactivation": self.reactivation,
            "Incoming Flow": round(self.incoming_flow, 4),
            "Outgoing Flow": round(self.outgoing_flow, 4),
            "Gate": round(self.gate, 4),
        }


@dataclass(frozen=True)
class SceneEdge:
    """Eine Kante, die einen gemessenen Informationsfluss darstellt."""

    id: str
    source: str
    target: str
    flow: float
    kind: str = "state_state"


@dataclass(frozen=True)
class SceneGraph:
    """Knoten und Kanten einer Detailstufe für genau einen Token."""

    level: DetailLevel
    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]
    token_index: int
    token_id: int | None = None
    #: Welche Stufen die zugrundeliegende Telemetrie überhaupt hergibt.
    available_levels: tuple[DetailLevel, ...] = ()
    #: Wenn die gewünschte Stufe nicht verfügbar war, steht hier, warum.
    downgraded_from: DetailLevel | None = None

    def by_id(self) -> dict[str, SceneNode]:
        return {node.id: node for node in self.nodes}


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


#: Knoten-IDs sind über alle Token stabil. Sie einmal zu zerlegen statt in
#: jedem Bild spart bei zehntausenden Knoten den Löwenanteil der Stringarbeit.
_ID_CACHE: dict[str, tuple[int, str, int | None]] = {}


def _parse_node_id(node_id: str) -> tuple[int, str, int | None]:
    """``core.3.context.cluster.17`` -> (3, "context", 17)."""
    cached = _ID_CACHE.get(node_id)
    if cached is not None:
        return cached
    parts = node_id.split(".")
    try:
        layer = int(parts[1])
    except (IndexError, ValueError):
        layer = 0
    kind = parts[2] if len(parts) > 2 else "other"
    index = None
    if "cluster" in parts:
        try:
            index = int(parts[-1])
        except ValueError:
            index = None
    result = (layer, kind, index)
    if len(_ID_CACHE) < 500_000:
        _ID_CACHE[node_id] = result
    return result


def cluster_nodes(
    frame: Any,
    *,
    outgoing: dict[str, float] | None = None,
    incoming: dict[str, float] | None = None,
) -> list[SceneNode]:
    """Clusterknoten aus einem ``network_step``-Frame."""
    outgoing = outgoing or {}
    incoming = incoming or {}
    nodes: list[SceneNode] = []
    for raw in frame.nodes:
        node_id = str(raw["id"])
        layer, kind, index = _parse_node_id(node_id)
        components = dict(raw.get("components", {}))
        nodes.append(SceneNode(
            id=node_id,
            kind=kind,
            level=DetailLevel.CLUSTER if index is not None else DetailLevel.STATE,
            layer=layer,
            index=index,
            members=1,
            activity=float(raw.get("activity", 0.0)),
            delta=float(components.get("delta_rms", 0.0)),
            state_norm=float(components.get("state_norm", 0.0)),
            persistence=int(raw.get("persistence", components.get("persistence_duration", 0)) or 0),
            reactivation=bool(raw.get("reactivation", components.get("reactivation", False))),
            gate=float(components.get("gate_mean", 0.0)),
            incoming_flow=incoming.get(
                node_id, float(components.get("incoming_flow_rms", 0.0))
            ),
            outgoing_flow=outgoing.get(node_id, 0.0),
            components=components,
        ))
    return nodes


def unit_nodes(frame: Any, *, max_units: int) -> list[SceneNode] | None:
    """Einzelne Units – nur möglich, wenn der Trace im Modus ``full`` entstand.

    ``max_units`` begrenzt, wie viele Einheiten je State gezeichnet werden.
    Überschüssige werden nicht erfunden und nicht zusammengefasst, sondern
    weggelassen; die Zahl der tatsächlich dargestellten steht im Ergebnis.
    """
    full = getattr(frame, "full", None)
    if not full:
        return None
    nodes: list[SceneNode] = []
    for layer_id, states in full.items():
        try:
            layer = int(str(layer_id).split(".")[-1])
        except ValueError:
            layer = 0
        for kind, values in states.items():
            for unit, value in enumerate(values[:max_units]):
                magnitude = abs(float(value))
                nodes.append(SceneNode(
                    id=f"core.{layer}.{kind}.unit.{unit}",
                    kind=kind,
                    level=DetailLevel.UNIT,
                    layer=layer,
                    index=unit,
                    members=1,
                    activity=magnitude,
                    state_norm=magnitude,
                ))
    return nodes or None


def aggregate_raw(frame: Any, level: DetailLevel) -> list[SceneNode]:
    """Aggregiert direkt aus den Rohknoten des Frames.

    Der Umweg über einzelne ``SceneNode``-Objekte entfällt. Genau das macht
    die groben Detailstufen billig – und damit das Level of Detail überhaupt
    wirksam. Ohne diesen Weg kostete jede Stufe so viel wie die feinste.
    """
    buckets: dict[tuple[int, str], list[float]] = {}
    counts: dict[tuple[int, str], int] = {}
    persistence: dict[tuple[int, str], int] = {}
    reactivation: dict[tuple[int, str], bool] = {}
    for raw in frame.nodes:
        layer, kind, _ = _parse_node_id(str(raw["id"]))
        if level == DetailLevel.MODEL:
            key = (0, "model")
        elif level == DetailLevel.LAYER:
            key = (layer, "layer")
        else:
            key = (layer, kind)
        components = raw.get("components", {})
        row = buckets.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        row[0] += float(raw.get("activity", 0.0))
        row[1] += float(components.get("delta_rms", 0.0))
        row[2] += float(components.get("state_norm", 0.0))
        row[3] += float(components.get("gate_mean", 0.0))
        counts[key] = counts.get(key, 0) + 1
        value = int(raw.get("persistence", components.get("persistence_duration", 0)) or 0)
        persistence[key] = max(persistence.get(key, 0), value)
        reactivation[key] = reactivation.get(key, False) or bool(
            raw.get("reactivation", components.get("reactivation", False))
        )
    result: list[SceneNode] = []
    for (layer, kind), row in sorted(buckets.items()):
        count = counts[(layer, kind)]
        identifier = {
            DetailLevel.MODEL: "model",
            DetailLevel.LAYER: f"core.{layer}",
            DetailLevel.STATE: f"core.{layer}.{kind}",
        }[level]
        result.append(SceneNode(
            id=identifier,
            kind=kind if level == DetailLevel.STATE else "other",
            level=level, layer=layer, index=None, members=count,
            activity=row[0] / count, delta=row[1] / count,
            state_norm=row[2] / count, gate=row[3] / count,
            persistence=persistence[(layer, kind)],
            reactivation=reactivation[(layer, kind)],
        ))
    return result


def aggregate(nodes: Iterable[SceneNode], level: DetailLevel) -> list[SceneNode]:
    """Fasst Knoten auf eine gröbere Stufe zusammen.

    Aktivität, Delta und Fluss werden gemittelt, Persistenz als Maximum
    genommen und Reaktivierung als Oder-Verknüpfung. Die Zahl der
    zusammengefassten Elemente bleibt in ``members`` sichtbar, damit an der
    Oberfläche erkennbar ist, wie stark verdichtet wurde.
    """
    if level >= DetailLevel.CLUSTER:
        return list(nodes)
    buckets: dict[tuple[int, str], list[SceneNode]] = {}
    for node in nodes:
        key = (node.layer, node.kind) if level == DetailLevel.STATE else (node.layer, "layer")
        if level == DetailLevel.MODEL:
            key = (0, "model")
        buckets.setdefault(key, []).append(node)
    result: list[SceneNode] = []
    for (layer, kind), group in sorted(buckets.items()):
        identifier = {
            DetailLevel.MODEL: "model",
            DetailLevel.LAYER: f"core.{layer}",
            DetailLevel.STATE: f"core.{layer}.{kind}",
        }[level]
        result.append(SceneNode(
            id=identifier,
            kind=kind if level == DetailLevel.STATE else "other",
            level=level,
            layer=layer,
            index=None,
            members=sum(node.members for node in group),
            activity=_mean([node.activity for node in group]),
            delta=_mean([node.delta for node in group]),
            state_norm=_mean([node.state_norm for node in group]),
            incoming_flow=_mean([node.incoming_flow for node in group]),
            outgoing_flow=_mean([node.outgoing_flow for node in group]),
            persistence=max((node.persistence for node in group), default=0),
            reactivation=any(node.reactivation for node in group),
            gate=_mean([node.gate for node in group]),
        ))
    return result


def aggregate_edges(
    edges: Iterable[SceneEdge], nodes: Sequence[SceneNode], level: DetailLevel,
    mapping: dict[str, str],
) -> list[SceneEdge]:
    """Bündelt Kanten entsprechend der Knotenaggregation.

    Parallele Kanten zwischen denselben aggregierten Endpunkten werden addiert,
    weil der Gesamtfluss zwischen zwei Bereichen genau die Summe der
    Einzelflüsse ist – das ist keine Schätzung, sondern eine Rechnung.
    """
    if level >= DetailLevel.CLUSTER:
        return list(edges)
    known = {node.id for node in nodes}
    combined: dict[tuple[str, str, str], float] = {}
    for edge in edges:
        source = mapping.get(edge.source, edge.source)
        target = mapping.get(edge.target, edge.target)
        if source == target or source not in known or target not in known:
            continue
        combined[(source, target, edge.kind)] = (
            combined.get((source, target, edge.kind), 0.0) + edge.flow
        )
    return [
        SceneEdge(id=f"{source}->{target}", source=source, target=target,
                  flow=flow, kind=kind)
        for (source, target, kind), flow in sorted(combined.items())
    ]


def _aggregate_mapping(nodes: Sequence[SceneNode], level: DetailLevel) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for node in nodes:
        if level == DetailLevel.MODEL:
            mapping[node.id] = "model"
        elif level == DetailLevel.LAYER:
            mapping[node.id] = f"core.{node.layer}"
        elif level == DetailLevel.STATE:
            mapping[node.id] = f"core.{node.layer}.{node.kind}"
        else:
            mapping[node.id] = node.id
    return mapping


def edges_from_frame(frame: Any) -> list[SceneEdge]:
    result = []
    for raw in frame.edges:
        identifier = str(raw["id"])
        kind = identifier.split(".")[2] if identifier.count(".") >= 3 else "state_state"
        result.append(SceneEdge(
            id=identifier,
            source=str(raw["source"]),
            target=str(raw["target"]),
            flow=float(raw.get("flow", 0.0)),
            kind=kind,
        ))
    return result


def available_levels(frame: Any) -> tuple[DetailLevel, ...]:
    """Welche Detailstufen dieser Frame tatsächlich belegen kann."""
    levels = [DetailLevel.MODEL, DetailLevel.LAYER, DetailLevel.STATE]
    if getattr(frame, "nodes", ()):
        levels.append(DetailLevel.CLUSTER)
    if getattr(frame, "full", None):
        levels.append(DetailLevel.UNIT)
    return tuple(levels)


def build_scene(
    frame: Any, level: DetailLevel, *, max_units: int = 256,
    allowed: Sequence[DetailLevel] | None = None,
) -> SceneGraph:
    """Baut die Szene einer Detailstufe aus einem Replay-Frame.

    Ist die gewünschte Stufe nicht belegt, wird auf die tiefste verfügbare
    zurückgefallen und das im Ergebnis vermerkt – nicht stillschweigend.
    """
    levels = tuple(allowed) if allowed else available_levels(frame)
    requested = level
    if level not in levels:
        level = max(item for item in levels if item <= level) if any(
            item <= level for item in levels
        ) else levels[0]

    # Flüsse zuerst, und zwar direkt aus den Roh-Wörterbüchern. Auf groben
    # Stufen entstehen dadurch überhaupt keine ``SceneEdge``-Objekte: Bei
    # zehntausenden Kanten war deren Bau gemessen der zweitteuerste Posten,
    # obwohl die Kanten dort anschließend ohnehin gebündelt werden.
    mapping_for = {
        DetailLevel.MODEL: lambda node_id: "model",
        DetailLevel.LAYER: lambda node_id: f"core.{_parse_node_id(node_id)[0]}",
        DetailLevel.STATE: lambda node_id: "core.{}.{}".format(
            *_parse_node_id(node_id)[:2]
        ),
    }.get(level, lambda node_id: node_id)
    outgoing: dict[str, float] = {}
    incoming: dict[str, float] = {}
    for raw in frame.edges:
        source = mapping_for(str(raw["source"]))
        target = mapping_for(str(raw["target"]))
        flow = float(raw.get("flow", 0.0))
        outgoing[source] = outgoing.get(source, 0.0) + flow
        incoming[target] = incoming.get(target, 0.0) + flow
    raw_edges = edges_from_frame(frame) if level >= DetailLevel.CLUSTER else ()

    if level == DetailLevel.UNIT:
        nodes = unit_nodes(frame, max_units=max_units)
        if nodes is None:
            level = DetailLevel.CLUSTER
            nodes = cluster_nodes(frame, outgoing=outgoing, incoming=incoming)
            edges = raw_edges
        else:
            edges = []
    elif level == DetailLevel.CLUSTER:
        nodes = cluster_nodes(frame, outgoing=outgoing, incoming=incoming)
        edges = raw_edges
    else:
        nodes = aggregate_raw(frame, level)
        known = {node.id for node in nodes}
        combined: dict[tuple[str, str, str], float] = {}
        for raw in frame.edges:
            source = mapping_for(str(raw["source"]))
            target = mapping_for(str(raw["target"]))
            if source == target or source not in known or target not in known:
                continue
            identifier = str(raw["id"])
            kind = identifier.split(".")[2] if identifier.count(".") >= 3 else "state_state"
            key = (source, target, kind)
            combined[key] = combined.get(key, 0.0) + float(raw.get("flow", 0.0))
        edges = [
            SceneEdge(id=f"{source}->{target}", source=source, target=target,
                      flow=flow, kind=kind)
            for (source, target, kind), flow in sorted(combined.items())
        ]
        nodes = [
            replace(node, outgoing_flow=outgoing.get(node.id, 0.0),
                    incoming_flow=incoming.get(node.id, 0.0))
            for node in nodes
        ]

    return SceneGraph(
        level=level,
        nodes=tuple(nodes),
        edges=tuple(edges),
        token_index=frame.token_index,
        token_id=frame.token_id,
        available_levels=levels,
        downgraded_from=requested if requested != level else None,
    )


def level_for_zoom(zoom: float, available: Sequence[DetailLevel]) -> DetailLevel:
    """Detailstufe aus dem Zoomfaktor.

    ``zoom`` ist das Verhältnis von Bildschirm- zu Weltkoordinaten: größer
    heißt näher dran. Die Schwellen sind so gewählt, dass beim Hineinzoomen
    jeweils eine Stufe feiner wird, sobald die vorige Stufe den Bildschirm
    füllt. Nicht belegte Stufen werden übersprungen.
    """
    if not available:
        return DetailLevel.MODEL
    thresholds = (
        (DetailLevel.UNIT, 24.0),
        (DetailLevel.CLUSTER, 6.0),
        (DetailLevel.STATE, 1.6),
        (DetailLevel.LAYER, 0.5),
        (DetailLevel.MODEL, 0.0),
    )
    for level, minimum in thresholds:
        if zoom >= minimum and level in available:
            return level
    return min(available)
