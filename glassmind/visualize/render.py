"""Milestone 4.5: die VisPy-Zeichenschicht des Visual Inspector.

Hier passiert ausschließlich das Zeichnen. Was gezeichnet wird, entscheidet der
``Inspector``; dieses Modul übersetzt seine Szene in GPU-Batches.

Rendering-Grundsatz: **ein Visual je Elementklasse, nicht je Element.** Alle
Knoten liegen in einem einzigen ``Markers``-Aufruf, alle Kanten in einem
``Line``-Aufruf. Damit hängt die Bildrate an der Zahl der Zeichenaufrufe (fünf)
statt an der Zahl der Knoten – das ist die Voraussetzung dafür, dass auch
zehntausende Knoten flüssig bleiben.

Jede sichtbare Eigenschaft stammt aus einem Messwert:

======================  ==================================================
Knotengröße             Activation und Delta
Helligkeit              Activation, abklingende Aktivitätshistorie
Randfarbe               Reaktivierung (rot) sonst Persistenz als Helligkeit
Randstärke              Auswahl
Kantenstärke/Deckkraft  gemessener Flusswert
Zellenfarbe (Memory)    Lese-/Schreibzugriff, Belegung
Zellengröße (Memory)    Slot-Stärke und Lesegewicht
Zellenrand (Memory)     Ersetzungsereignis, sonst Alter
======================  ==================================================

Dekorative Effekte gibt es nicht. Was sich bewegt, hat sich im Modell bewegt.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from glassmind.visualize.inspector import Inspector, VisibleScene
from glassmind.visualize.scene import DetailLevel, SceneNode

#: Dunkles Thema. Die Farbtöne sind entsättigt gewählt, damit Helligkeit als
#: Aktivitätsmaß über längere Betrachtung lesbar bleibt und nicht von
#: Signalfarben überstrahlt wird.
DARK_BACKGROUND = "#0a1017"
KIND_COLORS: dict[str, tuple[float, float, float]] = {
    "input":    (0.38, 0.62, 0.92),
    "fast":     (0.24, 0.84, 0.80),
    "context":  (0.93, 0.68, 0.29),
    "semantic": (0.78, 0.42, 0.90),
    "output":   (0.44, 0.90, 0.52),
    "memory":   (0.62, 0.66, 0.78),
    "other":    (0.70, 0.72, 0.76),
}
FLOW_COLOR = (0.42, 0.74, 1.0)
REACTIVATION_COLOR = (1.0, 0.34, 0.18, 1.0)
SELECTION_COLOR = (1.0, 1.0, 1.0, 1.0)
MEMORY_READ_COLOR = (0.35, 1.0, 0.55)
MEMORY_WRITE_COLOR = (1.0, 0.48, 0.18)
MEMORY_OCCUPIED_COLOR = (0.34, 0.52, 0.82)
MEMORY_EMPTY_COLOR = (0.20, 0.22, 0.26)


def node_colors(
    nodes: Sequence[SceneNode], history: dict[str, list[float]], selected: str | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Farbe, Randfarbe und Größe je Knoten – alles aus Messwerten."""
    if not nodes:
        empty = np.zeros((0, 4), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.float32)
    faces = np.empty((len(nodes), 4), dtype=np.float32)
    edges = np.empty((len(nodes), 4), dtype=np.float32)
    sizes = np.empty(len(nodes), dtype=np.float32)
    for index, node in enumerate(nodes):
        base = KIND_COLORS.get(node.kind, KIND_COLORS["other"])
        recent = history.get(node.id, ())
        # Abklingende Historie: kürzlich aktive Knoten bleiben kurz sichtbar.
        trail = max(recent[-8:], default=0.0) * 0.35 if recent else 0.0
        intensity = min(1.0, 0.18 + node.activity * 2.5 + trail)
        faces[index] = (*(channel * intensity for channel in base), 1.0)
        if node.id == selected:
            edges[index] = SELECTION_COLOR
        elif node.reactivation:
            edges[index] = REACTIVATION_COLOR
        else:
            persistence = min(1.0, 0.2 + node.persistence / 24.0)
            edges[index] = (0.72, 0.86, 1.0, persistence)
        # Größe wächst mit Aktivität und Zustandsänderung; aggregierte Knoten
        # bekommen einen Zuschlag nach der Zahl zusammengefasster Elemente,
        # damit grobe Stufen nicht winzig wirken.
        aggregate_bonus = 6.0 * np.log1p(node.members) if node.members > 1 else 0.0
        sizes[index] = (
            11.0
            + min(26.0, node.activity * 42.0)
            + min(16.0, node.delta * 55.0)
            + aggregate_bonus
            + (9.0 if node.id == selected else 0.0)
        )
    return faces, edges, sizes


def edge_arrays(scene: VisibleScene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Liniensegmente und Farben aus echten Flusswerten."""
    if not scene.edges:
        return (np.zeros((0, 2), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
                np.zeros(0, dtype=np.float32))
    positions = np.empty((len(scene.edges) * 2, 2), dtype=np.float32)
    colors = np.empty((len(scene.edges) * 2, 4), dtype=np.float32)
    widths = np.empty(len(scene.edges), dtype=np.float32)
    for index, edge in enumerate(scene.edges):
        source = scene.layout.of(edge.source)
        target = scene.layout.of(edge.target)
        positions[2 * index] = source
        positions[2 * index + 1] = target
        alpha = float(min(0.95, 0.08 + abs(edge.flow) * 3.0))
        colors[2 * index] = colors[2 * index + 1] = (*FLOW_COLOR, alpha)
        widths[index] = 0.8 + min(4.0, abs(edge.flow) * 12.0)
    return positions, colors, widths


def memory_arrays_for_render(
    scene: VisibleScene, selected_slot: int | None
) -> dict[str, np.ndarray] | None:
    """Zeichendaten der Speicherbank – oder ``None``, wenn es keine gibt.

    Ohne Speicher im Modell wird kein leerer Bereich gezeichnet.
    """
    memory = scene.memory
    if not memory.get("slots"):
        return None
    slots = memory["slots"]
    # Die Positionen kommen aus dem Layout – derselben Quelle, aus der auch die
    # Kameragrenzen stammen. Zwei getrennte Rasterberechnungen hatten die Bank
    # aus dem sichtbaren Ausschnitt geschoben.
    positions = np.asarray(
        [scene.layout.of(f"memory.slot.{index}") for index in range(slots)],
        dtype=np.float32,
    )
    faces = np.empty((slots, 4), dtype=np.float32)
    edges = np.empty((slots, 4), dtype=np.float32)
    sizes = np.empty(slots, dtype=np.float32)
    for index in range(slots):
        strength = float(memory["strength"][index])
        if memory["write_active"][index]:
            base = MEMORY_WRITE_COLOR
        elif memory["read_active"][index]:
            base = MEMORY_READ_COLOR
        elif memory["occupied"][index]:
            base = MEMORY_OCCUPIED_COLOR
        else:
            base = MEMORY_EMPTY_COLOR
        intensity = 0.35 + min(0.65, strength)
        faces[index] = (*(channel * intensity for channel in base), 1.0)
        if memory["replaced"][index]:
            edges[index] = (1.0, 0.18, 0.18, 1.0)
        elif index == selected_slot:
            edges[index] = SELECTION_COLOR
        else:
            age = 1.0 - min(1.0, float(memory["age_normalised"][index]))
            edges[index] = (0.85, 0.9, 1.0, 0.15 + 0.7 * age)
        weight = float(memory["read_weight_by_slot"].get(index, 0.0))
        sizes[index] = (
            9.0 + 16.0 * min(1.0, strength) + 14.0 * weight
            + (7.0 if index == selected_slot else 0.0)
        )
    return {"positions": positions, "face_color": faces, "edge_color": edges,
            "size": sizes}


def memory_edges(scene: VisibleScene, frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """Lese- und Schreibkanten zwischen Netz und Speicherbank."""
    memory = scene.memory
    if not memory.get("slots") or frame.memory is None:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    if scene.nodes:
        points = np.asarray(
            [scene.layout.of(node.id) for node in scene.nodes], dtype=np.float32
        )
        anchor = (float(points[:, 0].mean()), float(points[:, 1].min()) - 1.6)
    else:
        anchor = (0.0, 0.0)
    positions: list[tuple[float, float]] = []
    colors: list[tuple[float, float, float, float]] = []
    for slot in frame.memory.write_slots:
        if 0 <= slot < memory["slots"]:
            alpha = min(0.95, 0.12 + float(frame.memory.write_gate) * 1.6)
            positions.extend((anchor, scene.layout.of(f"memory.slot.{slot}")))
            colors.extend(((*MEMORY_WRITE_COLOR, alpha),) * 2)
    for slot, weight in zip(frame.memory.read_slots, frame.memory.read_weights):
        if 0 <= slot < memory["slots"]:
            alpha = min(0.95, 0.12 + float(weight) * 1.4)
            positions.extend((scene.layout.of(f"memory.slot.{slot}"), anchor))
            colors.extend(((*MEMORY_READ_COLOR, alpha),) * 2)
    if not positions:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 4), dtype=np.float32)
    return (np.asarray(positions, dtype=np.float32),
            np.asarray(colors, dtype=np.float32))


def region_labels(scene: VisibleScene) -> list[tuple[str, tuple[float, float]]]:
    """Beschriftungen der State-Regionen – nur auf groben Stufen.

    Die Namen sind die technischen State-Namen. Es wird keine Bedeutung
    behauptet.
    """
    if scene.graph.level > DetailLevel.STATE:
        seen: dict[tuple[int, str], list[tuple[float, float]]] = {}
        for node in scene.nodes:
            if node.kind in KIND_COLORS and node.kind != "other":
                seen.setdefault((node.layer, node.kind), []).append(
                    scene.layout.of(node.id)
                )
        return [
            (kind, (float(np.mean([p[0] for p in points])),
                    float(max(p[1] for p in points)) + 0.7))
            for (layer, kind), points in sorted(seen.items())
        ]
    return [(node.kind if node.kind != "other" else node.id, scene.layout.of(node.id))
            for node in scene.nodes]


class NetworkRenderer:
    """Hält die VisPy-Visuals und aktualisiert sie aus einer Szene."""

    def __init__(self, parent_view: Any) -> None:
        from vispy import scene as vispy_scene

        self.view = parent_view
        self.nodes = vispy_scene.visuals.Markers(parent=parent_view.scene)
        self.edges = vispy_scene.visuals.Line(
            parent=parent_view.scene, connect="segments", method="gl", width=1.5
        )
        self.memory_nodes = vispy_scene.visuals.Markers(parent=parent_view.scene)
        self.memory_edges = vispy_scene.visuals.Line(
            parent=parent_view.scene, connect="segments", method="gl", width=2.0
        )
        self.labels = vispy_scene.visuals.Text(
            parent=parent_view.scene, color=(0.65, 0.72, 0.82, 0.85),
            font_size=7.0, anchor_x="center", anchor_y="bottom",
        )
        self.edges.order = 1
        self.nodes.order = 0
        self.draw_calls = 5

    def update(self, inspector: Inspector, scene: VisibleScene) -> dict[str, Any]:
        positions = (
            np.asarray([scene.layout.of(node.id) for node in scene.nodes],
                       dtype=np.float32)
            if scene.nodes else np.zeros((0, 2), dtype=np.float32)
        )
        faces, edge_colors, sizes = node_colors(
            scene.nodes, inspector.history, inspector.selected_node
        )
        if len(positions):
            self.nodes.set_data(positions, face_color=faces, edge_color=edge_colors,
                                edge_width=2.0, size=sizes)
        else:
            self.nodes.set_data(np.zeros((0, 2), dtype=np.float32))

        line_positions, line_colors, _ = edge_arrays(scene)
        if len(line_positions):
            self.edges.set_data(pos=line_positions, color=line_colors, connect="segments")
        else:
            self.edges.set_data(pos=np.zeros((0, 2), dtype=np.float32))

        memory = memory_arrays_for_render(scene, inspector.selected_slot)
        if memory is not None:
            self.memory_nodes.set_data(
                memory["positions"], face_color=memory["face_color"],
                edge_color=memory["edge_color"], edge_width=2.0, size=memory["size"],
            )
            positions_memory, colors_memory = memory_edges(scene, inspector.frame)
            if len(positions_memory):
                self.memory_edges.set_data(pos=positions_memory, color=colors_memory,
                                           connect="segments")
            else:
                self.memory_edges.set_data(pos=np.zeros((0, 2), dtype=np.float32))
        else:
            self.memory_nodes.set_data(np.zeros((0, 2), dtype=np.float32))
            self.memory_edges.set_data(pos=np.zeros((0, 2), dtype=np.float32))

        labels = region_labels(scene) if len(scene.nodes) <= 400 else []
        if labels:
            self.labels.text = [text for text, _ in labels]
            self.labels.pos = np.asarray([point for _, point in labels], dtype=np.float32)
            self.labels.visible = True
        else:
            self.labels.visible = False
        return {
            "drawn_nodes": len(scene.nodes),
            "drawn_edges": len(scene.edges),
            "memory_slots": 0 if memory is None else len(memory["positions"]),
            "draw_calls": self.draw_calls,
        }
