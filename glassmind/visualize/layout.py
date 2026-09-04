"""Milestone 4.5: räumliche Anordnung des Netzes.

Zwei getrennte Layouts, die nie vermischt werden:

``structure``  Die feste Struktur des Modells: Layer von links nach rechts,
               State-Regionen darin. Eine Position hängt ausschließlich von
               der Knoten-ID ab, niemals von Messwerten. Deshalb springt beim
               Tokenwechsel nichts.

``activity``   Ein *Analyse*-Layout: häufig gemeinsam aktive Cluster rücken
               zusammen. Es wird aus aufgezeichneter Aktivität berechnet und
               ist ausdrücklich als Analysewerkzeug gekennzeichnet.

Die Trennung ist wichtig, weil das zweite Layout leicht als Aussage über die
Bedeutung von Clustern missverstanden wird. Nähe im Analyse-Layout heißt
"war oft gleichzeitig aktiv" – nicht mehr.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from glassmind.visualize.scene import ALL_KINDS, STATE_KINDS, DetailLevel, SceneNode

#: Waagerechter Abstand zweier Layer.
LAYER_SPACING = 9.0
#: Waagerechter Versatz der State-Regionen innerhalb eines Layers.
KIND_OFFSET = {name: index * 1.55 for index, name in enumerate(STATE_KINDS)}
#: Der Speicher bekommt einen eigenen Bereich unterhalb des Netzes.
MEMORY_ORIGIN = (-9.0, -7.5)
MEMORY_PITCH = 0.62
MEMORY_COLUMNS = 16


@dataclass(frozen=True)
class LayoutResult:
    positions: dict[str, tuple[float, float]]
    #: Grenzen als (xmin, ymin, xmax, ymax) – für "Ansicht zurücksetzen".
    bounds: tuple[float, float, float, float]
    name: str

    def of(self, node_id: str) -> tuple[float, float]:
        return self.positions.get(node_id, (0.0, 0.0))


def _column_height(count: int) -> float:
    return max(count - 1, 0) * 0.72


def structure_position(node: SceneNode, *, cluster_count: int = 8) -> tuple[float, float]:
    """Position eines Knotens allein aus seiner Identität.

    Zwei Läufe mit derselben Modellstruktur ergeben dieselben Positionen.
    Das ist die Bedingung dafür, dass ein Replay überhaupt lesbar ist.
    """
    x = node.layer * LAYER_SPACING + KIND_OFFSET.get(node.kind, 2.8)
    if node.level <= DetailLevel.LAYER:
        return (node.layer * LAYER_SPACING + 2.8, 0.0)
    if node.index is None:
        return (x, 0.0)
    if node.level == DetailLevel.UNIT:
        # Units werden innerhalb ihrer State-Region in ein schmales Raster
        # gelegt, damit auch tausende Einheiten eine feste Position behalten.
        columns = 8
        row, column = divmod(node.index, columns)
        return (x + column * 0.16 - 0.56, -row * 0.16 + _column_height(columns) * 0.5)
    centre = (cluster_count - 1) / 2
    return (x, (node.index - centre) * 0.72)


def structure_layout(
    nodes: Sequence[SceneNode], *, memory_slots: int = 0
) -> LayoutResult:
    cluster_count = max(
        (node.index + 1 for node in nodes
         if node.level == DetailLevel.CLUSTER and node.index is not None),
        default=8,
    )
    positions = {node.id: structure_position(node, cluster_count=cluster_count)
                 for node in nodes}
    if positions:
        xs = [point[0] for point in positions.values()]
        ys = [point[1] for point in positions.values()]
        # Die Speicherbank sitzt mittig *unter* dem Netz, nicht an festen
        # Koordinaten. Sonst driften Netz und Bank bei mehr Layern oder
        # Clustern auseinander und die Ansicht zeigt überwiegend leere Fläche.
        columns = max(1, min(MEMORY_COLUMNS, memory_slots or 1))
        grid_width = (columns - 1) * MEMORY_PITCH
        centre = (min(xs) + max(xs)) / 2
        origin = (centre - grid_width / 2, min(ys) - 1.8)
    else:
        origin = MEMORY_ORIGIN
    positions.update(memory_layout(memory_slots, origin=origin))
    if not positions:
        return LayoutResult({}, (-1.0, -1.0, 1.0, 1.0), "structure")
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    return LayoutResult(
        positions,
        (min(xs) - 1.0, min(ys) - 1.0, max(xs) + 1.0, max(ys) + 1.0),
        "structure",
    )


def memory_layout(
    slots: int, *, columns: int = MEMORY_COLUMNS,
    origin: tuple[float, float] = MEMORY_ORIGIN,
) -> dict[str, tuple[float, float]]:
    """Rasterposition je Speicherzelle.

    Ein Gitter statt einer Reihe: Bei 64 bis 128 Slots bleibt die Bank kompakt
    und die Zellgröße konstant, sodass die Darstellung flüssig bleibt.
    """
    if slots <= 0:
        return {}
    columns = max(1, min(columns, slots))
    return {
        f"memory.slot.{index}": (
            origin[0] + (index % columns) * MEMORY_PITCH,
            origin[1] - (index // columns) * MEMORY_PITCH,
        )
        for index in range(slots)
    }


# ----------------------------------------------------------------------
# Analyse-Layout: Aktivitätsinseln
# ----------------------------------------------------------------------

def coactivation_matrix(
    activity_series: dict[str, Sequence[float]]
) -> tuple[list[str], list[list[float]]]:
    """Korrelation der Aktivitätsverläufe zweier Cluster.

    Grundlage ist der Pearson-Korrelationskoeffizient über alle Token des
    Traces. Cluster ohne Varianz (durchgehend gleich aktiv) korrelieren mit
    nichts – sie bekommen den Wert 0 statt eines undefinierten Ergebnisses.
    """
    ids = sorted(activity_series)
    series = [list(activity_series[node_id]) for node_id in ids]
    length = min((len(values) for values in series), default=0)
    if length < 2:
        return ids, [[0.0] * len(ids) for _ in ids]
    series = [values[:length] for values in series]
    means = [sum(values) / length for values in series]
    deviations = [
        [value - mean for value in values] for values, mean in zip(series, means)
    ]
    norms = [math.sqrt(sum(value * value for value in row)) for row in deviations]
    matrix = [[0.0] * len(ids) for _ in ids]
    for i in range(len(ids)):
        for j in range(i, len(ids)):
            if norms[i] < 1e-12 or norms[j] < 1e-12:
                value = 0.0
            else:
                value = sum(
                    a * b for a, b in zip(deviations[i], deviations[j])
                ) / (norms[i] * norms[j])
            matrix[i][j] = matrix[j][i] = value
    return ids, matrix


def activity_layout(
    activity_series: dict[str, Sequence[float]],
    *,
    iterations: int = 200,
    seed: int = 17,
    spread: float = 8.0,
) -> LayoutResult:
    """Offline berechnetes Layout: gemeinsam aktive Cluster liegen näher.

    **Das ist ein Analyse-Layout, keine Aussage über Modellbedeutung.** Nähe
    bedeutet ausschließlich, dass zwei Cluster über den Trace hinweg
    zusammen aktiv waren. Namen bleiben die technischen Knoten-IDs; es wird
    keine Gruppe benannt, kategorisiert oder gedeutet.

    Verfahren: einfache kräftebasierte Anordnung. Korrelation zieht an,
    ein konstanter Term stößt ab. Der Startzustand ist über ``seed``
    reproduzierbar, damit dasselbe Trace dasselbe Bild ergibt.
    """
    import random

    ids, matrix = coactivation_matrix(activity_series)
    if not ids:
        return LayoutResult({}, (-1.0, -1.0, 1.0, 1.0), "activity")
    generator = random.Random(seed)
    count = len(ids)
    points = [
        [generator.uniform(-spread, spread), generator.uniform(-spread, spread)]
        for _ in range(count)
    ]
    for step in range(iterations):
        cooling = 1.0 - step / max(iterations, 1)
        forces = [[0.0, 0.0] for _ in range(count)]
        for i in range(count):
            for j in range(i + 1, count):
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                distance = math.hypot(dx, dy) or 1e-6
                ux, uy = dx / distance, dy / distance
                # Anziehung proportional zur Korrelation, Abstoßung konstant.
                attraction = max(0.0, matrix[i][j]) * distance * 0.05
                repulsion = spread * spread / (distance * distance * count) * 0.5
                force = attraction - repulsion
                forces[i][0] += ux * force
                forces[i][1] += uy * force
                forces[j][0] -= ux * force
                forces[j][1] -= uy * force
        for index, point in enumerate(points):
            point[0] += forces[index][0] * cooling
            point[1] += forces[index][1] * cooling
    positions = {node_id: (point[0], point[1]) for node_id, point in zip(ids, points)}
    xs = [point[0] for point in positions.values()]
    ys = [point[1] for point in positions.values()]
    return LayoutResult(
        positions,
        (min(xs) - 1.0, min(ys) - 1.0, max(xs) + 1.0, max(ys) + 1.0),
        "activity",
    )
