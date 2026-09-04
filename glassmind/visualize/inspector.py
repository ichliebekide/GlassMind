"""Milestone 4.5: die Steuerlogik des Visual Inspector.

Alles, was der Benutzer tut – blättern, filtern, suchen, auswählen,
vergleichen, eingreifen – liegt hier und kommt ohne Fenster aus. Das Rendering
in ``app.py`` liest diesen Zustand nur ab.

Die Trennung hat einen praktischen Grund: GUI-Logik, die einen Grafikkontext
braucht, lässt sich nicht sinnvoll testen. So aber ist jede Interaktion in
einem gewöhnlichen Unittest prüfbar.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from glassmind.observe.bus import ObservationMode
from glassmind.visualize.graph import NetworkFrame, ReplayTimeline, memory_arrays
from glassmind.visualize.layout import LayoutResult, activity_layout, structure_layout
from glassmind.visualize.scene import (
    ALL_KINDS,
    DetailLevel,
    SceneEdge,
    SceneGraph,
    SceneNode,
    build_scene,
    level_for_zoom,
)

#: Welche Detailstufe ein Observation-Modus überhaupt tragen kann. Die GUI
#: erfindet nichts, was der Modus nicht liefert.
MODE_CEILING = {
    ObservationMode.OFF: None,
    ObservationMode.SUMMARY: DetailLevel.STATE,
    ObservationMode.TRACE: DetailLevel.CLUSTER,
    ObservationMode.FULL: DetailLevel.UNIT,
}


@dataclass
class Filters:
    """Sichtbarkeitsregeln. Jede reduziert die tatsächlich gezeichnete Menge."""

    kinds: set[str] = field(default_factory=lambda: set(ALL_KINDS))
    show_flow: bool = True
    activity_threshold: float = 0.0
    top_nodes: int | None = None
    top_flows: int | None = None
    only_changed: bool = False
    only_persistent: bool = False
    only_reactivated: bool = False
    #: Ab welchem Delta ein Knoten als "verändert" gilt.
    change_threshold: float = 0.01
    #: Ab welcher Persistenz ein Knoten als "persistent" gilt.
    persistence_threshold: int = 8

    def toggle_kind(self, kind: str) -> None:
        if kind in self.kinds:
            self.kinds.discard(kind)
        else:
            self.kinds.add(kind)

    def describe(self) -> str:
        active = []
        hidden = sorted(set(ALL_KINDS) - self.kinds)
        if hidden:
            active.append("aus: " + ",".join(hidden))
        if self.activity_threshold > 0:
            active.append(f"Aktivität≥{self.activity_threshold:.2f}")
        if self.top_nodes:
            active.append(f"Top-{self.top_nodes} Knoten")
        if self.top_flows:
            active.append(f"Top-{self.top_flows} Flüsse")
        if self.only_changed:
            active.append("nur verändert")
        if self.only_persistent:
            active.append("nur persistent")
        if self.only_reactivated:
            active.append("nur reaktiviert")
        if not self.show_flow:
            active.append("ohne Fluss")
        return " | ".join(active) if active else "keine Filter"


@dataclass(frozen=True)
class VisibleScene:
    """Was nach Aggregation und Filterung tatsächlich gezeichnet wird."""

    graph: SceneGraph
    nodes: tuple[SceneNode, ...]
    edges: tuple[SceneEdge, ...]
    layout: LayoutResult
    memory: dict[str, Any]
    #: Wie viele Knoten die Stufe vor der Filterung hatte.
    total_nodes: int
    total_edges: int

    @property
    def reduction(self) -> float:
        """Anteil der weggefilterten Knoten – belegt, dass Filter wirken."""
        if not self.total_nodes:
            return 0.0
        return 1.0 - len(self.nodes) / self.total_nodes


class Inspector:
    """Der beobachtbare Zustand der Oberfläche.

    Der Inspector besitzt die Zeitachse, die Auswahl, die Filter und die
    Detailstufe. Er kennt weder VisPy noch ein Fenster.
    """

    def __init__(
        self,
        timeline: ReplayTimeline,
        *,
        mode: ObservationMode | str = ObservationMode.TRACE,
        tokens: Sequence[int] | None = None,
        decode: Callable[[Sequence[int]], str] | None = None,
        max_units: int = 256,
    ) -> None:
        self.timeline = timeline
        self.mode = ObservationMode.parse(mode)
        self.index = 0
        self.playing = False
        self.speed = 8.0
        self.filters = Filters()
        self.selected_node: str | None = None
        self.selected_slot: int | None = None
        self.auto_level = True
        self.zoom = 6.0
        self.max_units = max_units
        self.layout_name = "structure"
        self._activity_layout: LayoutResult | None = None
        self._layout_cache: tuple[Any, LayoutResult] | None = None
        self.interventions: dict[str, Any] = {}
        self.comparison: "Inspector | None" = None
        self.history: dict[str, list[float]] = {}
        self._tokens = list(tokens or [])
        self._decode = decode
        self._requested_level = DetailLevel.CLUSTER

    # -- Zeitachse ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.timeline)

    @property
    def frame(self) -> NetworkFrame:
        return self.timeline[self.index]

    def seek(self, index: int) -> int:
        self.index = max(0, min(len(self.timeline) - 1, int(index)))
        return self.index

    def step(self, delta: int = 1) -> int:
        return self.seek(self.index + delta)

    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle_play(self) -> None:
        self.playing = not self.playing

    def advance(self) -> bool:
        """Ein Schritt im Abspielmodus. Gibt zurück, ob sich etwas änderte."""
        if not self.playing:
            return False
        if self.index + 1 >= len(self.timeline):
            self.playing = False
            return False
        self.index += 1
        return True

    def set_speed(self, fps: float) -> float:
        self.speed = max(0.5, min(120.0, float(fps)))
        return self.speed

    # -- Detailstufe --------------------------------------------------
    @property
    def ceiling(self) -> DetailLevel | None:
        """Die feinste Stufe, die der Observation-Modus überhaupt trägt."""
        return MODE_CEILING[self.mode]

    def set_level(self, level: DetailLevel | str) -> DetailLevel:
        self._requested_level = DetailLevel.parse(level)
        self.auto_level = False
        return self._requested_level

    def set_zoom(self, zoom: float) -> DetailLevel:
        self.zoom = max(0.01, float(zoom))
        return self.level()

    def requested_level(self) -> DetailLevel:
        """Die *gewünschte* Stufe, nur durch den Observation-Modus begrenzt.

        Bewusst noch nicht auf die vorhandenen Daten beschnitten: Erst
        ``build_scene`` tut das und vermerkt dabei, dass es beschnitten hat.
        Sonst verschwände die Auskunft, dass eine angeforderte Stufe gar nicht
        aufgezeichnet wurde – und die Oberfläche zeigte stillschweigend etwas
        anderes als verlangt.
        """
        if self.auto_level:
            # Beim Zoomen ist ein Rückfall selbstverständlich und wird nicht
            # als Mangel gemeldet; die Stufe folgt dem, was da ist.
            return level_for_zoom(self.zoom, self._available_levels())
        return self._requested_level

    def level(self) -> DetailLevel:
        """Die tatsächlich darstellbare Stufe."""
        available = self._available_levels()
        if not available:
            return DetailLevel.MODEL
        candidate = self.requested_level()
        allowed = [item for item in available if item <= candidate]
        return max(allowed) if allowed else min(available)

    def _available_levels(self) -> tuple[DetailLevel, ...]:
        """Stufen, die *sowohl* der Modus trägt *als auch* die Daten belegen.

        Beide Grenzen laufen hier zusammen, damit es genau einen Ort gibt, an
        dem eine Stufe wegfällt – und genau eine Meldung darüber.
        """
        from glassmind.visualize.scene import available_levels

        if self.mode is ObservationMode.OFF:
            return ()
        levels = available_levels(self.frame)
        ceiling = self.ceiling
        if ceiling is not None:
            levels = tuple(level for level in levels if level <= ceiling)
        return levels or (DetailLevel.MODEL,)

    # -- Szene --------------------------------------------------------
    def scene(self) -> VisibleScene:
        """Baut die sichtbare Szene: aggregieren, filtern, anordnen."""
        if self.mode is ObservationMode.OFF:
            empty = SceneGraph(DetailLevel.MODEL, (), (), self.frame.token_index,
                               self.frame.token_id, (), None)
            return VisibleScene(empty, (), (), structure_layout(()), {}, 0, 0)
        graph = build_scene(self.frame, self.requested_level(),
                            max_units=self.max_units,
                            allowed=self._available_levels())
        self._remember(graph)
        nodes = self._filter_nodes(graph.nodes)
        edges = self._filter_edges(graph.edges, {node.id for node in nodes})
        memory = memory_arrays(self.frame.memory) if self._memory_visible() else {}
        layout = self._layout(graph.nodes, memory.get("slots", 0))
        return VisibleScene(
            graph=graph, nodes=tuple(nodes), edges=tuple(edges), layout=layout,
            memory=memory, total_nodes=len(graph.nodes), total_edges=len(graph.edges),
        )

    def _memory_visible(self) -> bool:
        # Ohne Speicher im Modell wird auch kein leerer Bereich gezeigt.
        return self.frame.memory is not None and "memory" in self.filters.kinds

    #: Ab wie vielen Knoten die Aktivitätshistorie nur noch für den
    #: ausgewählten Knoten geführt wird. Bei sehr großen Netzen kostet die
    #: Buchhaltung sonst mehr als das Zeichnen – gemessen, nicht vermutet.
    HISTORY_LIMIT = 4096

    def _remember(self, graph: SceneGraph) -> None:
        if len(graph.nodes) > self.HISTORY_LIMIT:
            if self.selected_node is None:
                return
            selected = next(
                (node for node in graph.nodes if node.id == self.selected_node), None
            )
            nodes = (selected,) if selected is not None else ()
        else:
            nodes = graph.nodes
        for node in nodes:
            series = self.history.setdefault(node.id, [])
            series.append(node.activity)
            if len(series) > 512:
                del series[: len(series) - 512]

    def _filter_nodes(self, nodes: Sequence[SceneNode]) -> list[SceneNode]:
        filters = self.filters
        result = [
            node for node in nodes
            if (node.kind in filters.kinds or node.kind == "other")
            and node.activity >= filters.activity_threshold
            and (not filters.only_changed or node.delta >= filters.change_threshold)
            and (not filters.only_persistent
                 or node.persistence >= filters.persistence_threshold)
            and (not filters.only_reactivated or node.reactivation)
        ]
        if filters.top_nodes:
            result = sorted(result, key=lambda node: -node.activity)[: filters.top_nodes]
            result = sorted(result, key=lambda node: node.id)
        # Ein ausgewählter Knoten bleibt sichtbar, auch wenn ein Filter ihn
        # ausschließen würde – sonst verschwindet die Detailtafel unter der Hand.
        if self.selected_node and all(node.id != self.selected_node for node in result):
            keep = next((node for node in nodes if node.id == self.selected_node), None)
            if keep is not None:
                result.append(keep)
        return result

    def _filter_edges(self, edges: Sequence[SceneEdge], visible: set[str]) -> list[SceneEdge]:
        if not self.filters.show_flow:
            return []
        result = [
            edge for edge in edges
            if edge.source in visible and edge.target in visible
        ]
        if self.filters.top_flows:
            result = sorted(result, key=lambda edge: -abs(edge.flow))[: self.filters.top_flows]
        return result

    def _layout(self, nodes: Sequence[SceneNode], memory_slots: int) -> LayoutResult:
        if self.layout_name == "activity" and self._activity_layout is not None:
            return self._activity_layout
        # Die Struktur-Positionen hängen ausschließlich an den Knoten-IDs, und
        # die sind über alle Token stabil. Einmal rechnen genügt daher, solange
        # sich die Knotenmenge nicht ändert.
        key = (len(nodes), nodes[0].id if nodes else "", nodes[-1].id if nodes else "",
               memory_slots)
        if self._layout_cache is not None and self._layout_cache[0] == key:
            return self._layout_cache[1]
        result = structure_layout(nodes, memory_slots=memory_slots)
        self._layout_cache = (key, result)
        return result

    def use_structure_layout(self) -> None:
        self.layout_name = "structure"

    def use_activity_layout(self, *, iterations: int = 150) -> LayoutResult:
        """Berechnet das Analyse-Layout aus dem gesamten Trace.

        Nähe bedeutet dort ausschließlich gemeinsame Aktivität. Es wird keine
        Gruppe benannt oder gedeutet – die Knoten behalten ihre IDs.
        """
        series: dict[str, list[float]] = {}
        for index in range(len(self.timeline)):
            frame = self.timeline[index]
            for node in frame.nodes:
                series.setdefault(str(node["id"]), []).append(
                    float(node.get("activity", 0.0))
                )
        self._activity_layout = activity_layout(series, iterations=iterations)
        self.layout_name = "activity"
        return self._activity_layout

    # -- Auswahl ------------------------------------------------------
    def select(self, node_id: str | None) -> str | None:
        self.selected_node = node_id
        if node_id is not None:
            self.selected_slot = None
        return self.selected_node

    def select_slot(self, slot: int | None) -> int | None:
        self.selected_slot = slot
        if slot is not None:
            self.selected_node = None
        return self.selected_slot

    def select_nearest(self, point: tuple[float, float]) -> str | None:
        """Wählt den Knoten, dessen Position dem Klick am nächsten liegt."""
        scene = self.scene()
        best_id, best_distance = None, float("inf")
        for node in scene.nodes:
            x, y = scene.layout.of(node.id)
            distance = (x - point[0]) ** 2 + (y - point[1]) ** 2
            if distance < best_distance:
                best_id, best_distance = node.id, distance
        slot_id, slot_distance = None, float("inf")
        for index in range(scene.memory.get("slots", 0)):
            x, y = scene.layout.of(f"memory.slot.{index}")
            distance = (x - point[0]) ** 2 + (y - point[1]) ** 2
            if distance < slot_distance:
                slot_id, slot_distance = index, distance
        if slot_id is not None and slot_distance < best_distance:
            self.select_slot(slot_id)
            return None
        return self.select(best_id)

    def selection_detail(self) -> dict[str, Any]:
        """Messwerte des ausgewählten Objekts – ohne jede Deutung."""
        if self.selected_slot is not None and self.frame.memory is not None:
            return {"typ": "memory_slot",
                    **self.frame.memory.slot_detail(self.selected_slot,
                                                    self.frame.token_index)}
        if self.selected_node is None:
            return {}
        node = self.scene().graph.by_id().get(self.selected_node)
        if node is None:
            return {"id": self.selected_node, "Hinweis": "auf dieser Stufe nicht vorhanden"}
        detail = {"typ": "node", **node.detail()}
        series = self.history.get(self.selected_node, [])
        if len(series) > 1:
            detail["Verlauf"] = series[-64:]
        return detail

    # -- Suche --------------------------------------------------------
    def search(self, query: str) -> list[str]:
        """Findet Knoten und Speicherzellen nach Layer, Cluster, Unit oder ID.

        Erlaubte Formen sind unter anderem ``layer 3``, ``cluster 17``,
        ``slot 42``, ``token 128`` sowie beliebige Teilzeichenketten einer
        Knoten-ID. Ein Treffer wird ausgewählt, damit die Ansicht ihn
        zentrieren kann.
        """
        text = query.strip().lower()
        if not text:
            return []
        scene = self.scene()
        identifiers = [node.id for node in scene.nodes]

        if text.startswith("token"):
            rest = text.removeprefix("token").strip()
            if rest.isdigit():
                self.seek(int(rest))
                return [f"token:{self.index}"]
            return []
        if text.startswith("slot") and self.frame.memory is not None:
            rest = text.removeprefix("slot").strip()
            if rest.isdigit() and 0 <= int(rest) < self.frame.memory.slots:
                self.select_slot(int(rest))
                return [f"memory.slot.{rest}"]
            return []
        for prefix, matcher in (
            ("layer", lambda value, node_id: node_id.startswith(f"core.{value}.")),
            ("cluster", lambda value, node_id: node_id.endswith(f".cluster.{value}")),
            ("unit", lambda value, node_id: node_id.endswith(f".unit.{value}")),
        ):
            if text.startswith(prefix):
                rest = text.removeprefix(prefix).strip()
                if rest.isdigit():
                    hits = [node_id for node_id in identifiers if matcher(rest, node_id)]
                    if hits:
                        self.select(hits[0])
                    return hits
                return []
        hits = [node_id for node_id in identifiers if text in node_id.lower()]
        if hits:
            self.select(hits[0])
        return hits

    # -- Token-Kontext ------------------------------------------------
    def token_context(self, *, width: int = 12) -> dict[str, Any]:
        """Der aktuelle Token mit ein wenig Umgebung.

        Ohne Tokenliste bleibt der Text leer statt erfunden zu werden.
        """
        if not self._tokens:
            return {"text": "", "caret": 0, "available": False}
        start = max(0, self.index - width)
        end = min(len(self._tokens), self.index + width + 1)
        window = self._tokens[start:end]
        if self._decode is None:
            text = " ".join(str(value) for value in window)
            caret = len(" ".join(str(value) for value in self._tokens[start:self.index]))
        else:
            text = self._decode(window)
            caret = len(self._decode(self._tokens[start:self.index]))
        return {"text": text, "caret": max(0, caret), "available": True,
                "token_index": self.index, "total": len(self._tokens)}

    # -- Vergleich ----------------------------------------------------
    def attach_comparison(self, other: "Inspector") -> None:
        self.comparison = other

    def comparison_report(self) -> dict[str, Any]:
        """Unterschiede zwischen zwei Läufen – nur gemessene Größen."""
        if self.comparison is None:
            return {}
        other = self.comparison
        other.seek(self.index)
        mine = {node.id: node for node in self.scene().nodes}
        theirs = {node.id: node for node in other.scene().nodes}
        shared = sorted(set(mine) & set(theirs))
        rows = [
            {
                "id": node_id,
                "activation_difference": mine[node_id].activity - theirs[node_id].activity,
                "state_difference": mine[node_id].state_norm - theirs[node_id].state_norm,
                "flow_difference": (
                    mine[node_id].outgoing_flow - theirs[node_id].outgoing_flow
                ),
            }
            for node_id in shared
        ]
        entropy_a = self.frame.entropy
        entropy_b = other.frame.entropy
        return {
            "shared_nodes": len(shared),
            "only_here": sorted(set(mine) - set(theirs)),
            "only_there": sorted(set(theirs) - set(mine)),
            "rows": rows,
            "mean_absolute_activation_difference": (
                sum(abs(row["activation_difference"]) for row in rows) / len(rows)
                if rows else 0.0
            ),
            "prediction_difference": (
                None if entropy_a is None or entropy_b is None
                else entropy_a - entropy_b
            ),
        }

    # -- Eingriffe ----------------------------------------------------
    def set_intervention(self, name: str, value: Any) -> dict[str, Any]:
        """Merkt einen Analyse-Eingriff vor.

        Der Inspector verändert das Modell nicht selbst. Er sammelt nur die
        Optionen, die eine Live-Sitzung an ``forward`` weitergibt. Ohne
        aktiven Eingriff ist das Wörterbuch leer und die Inferenz läuft
        unverändert.
        """
        allowed = {
            "ablate_states", "ablate_memory_slots",
            "disable_memory", "disable_memory_read", "disable_memory_write",
        }
        if name not in allowed:
            raise ValueError(f"Unbekannter Eingriff {name!r}; erlaubt: {', '.join(sorted(allowed))}")
        if not value:
            self.interventions.pop(name, None)
        else:
            self.interventions[name] = value
        return dict(self.interventions)

    def clear_interventions(self) -> None:
        self.interventions.clear()

    @property
    def analysis_mode(self) -> bool:
        """Ob gerade ein Eingriff aktiv ist – die GUI muss das anzeigen."""
        return bool(self.interventions)

    def status_line(self) -> str:
        scene = self.scene()
        parts = [
            f"Token {self.index + 1}/{len(self.timeline)}",
            f"Stufe {scene.graph.level.label}",
            f"{len(scene.nodes)}/{scene.total_nodes} Knoten",
            f"{len(scene.edges)}/{scene.total_edges} Kanten",
            self.filters.describe(),
            f"Layout {self.layout_name}",
        ]
        if scene.graph.downgraded_from is not None:
            parts.append(
                f"Stufe {scene.graph.downgraded_from.label} nicht in der Telemetrie"
            )
        if self.analysis_mode:
            parts.append("ANALYSEMODUS: " + ", ".join(sorted(self.interventions)))
        return " | ".join(parts)
