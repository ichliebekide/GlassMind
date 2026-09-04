"""Milestone 4.5: Tests für den Visual Inspector.

Die gesamte Bedienlogik liegt in ``Inspector`` und ``scene`` und braucht kein
Fenster. Genau deshalb ist sie hier vollständig prüfbar. Ein einziger Test
baut wirklich ein Qt-Fenster und wird übersprungen, wo kein Display existiert.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from glassmind.observe.bus import ObservationMode
from glassmind.observe.events import ObservationEvent
from glassmind.visualize.graph import MemoryFrame, NetworkFrame, ReplayTimeline
from glassmind.visualize.inspector import MODE_CEILING, Filters, Inspector
from glassmind.visualize.layout import (
    activity_layout,
    coactivation_matrix,
    structure_layout,
)
from glassmind.visualize.live import FrameAssembler, TelemetryBuffer
from glassmind.visualize.scene import (
    DetailLevel,
    SceneNode,
    available_levels,
    build_scene,
)


# ----------------------------------------------------------------------
# Testdaten: ein kleines, aber vollständiges Netz
# ----------------------------------------------------------------------

def make_frame(
    token_index: int = 0, *, layers: int = 2, clusters: int = 4,
    memory_slots: int = 0, full: bool = False,
) -> NetworkFrame:
    nodes, edges = [], []
    for layer in range(layers):
        for kind in ("fast", "context", "semantic"):
            for cluster in range(clusters):
                nodes.append({
                    "id": f"core.{layer}.{kind}.cluster.{cluster}",
                    "kind": kind,
                    "activity": 0.1 * (cluster + 1) + 0.05 * layer,
                    "persistence": cluster * 5,
                    "reactivation": cluster == 1,
                    "components": {
                        "delta_rms": 0.02 * (cluster + 1),
                        "state_norm": 1.0 + cluster,
                        "gate_mean": 0.5,
                        "incoming_flow_rms": 0.03 * (cluster + 1),
                    },
                })
        for cluster in range(clusters):
            edges.append({
                "id": f"core.{layer}.fast_context.{cluster}",
                "source": f"core.{layer}.fast.cluster.{cluster}",
                "target": f"core.{layer}.context.cluster.{cluster}",
                "flow": 0.1 * (cluster + 1),
            })
            edges.append({
                "id": f"core.{layer}.context_semantic.{cluster}",
                "source": f"core.{layer}.context.cluster.{cluster}",
                "target": f"core.{layer}.semantic.cluster.{cluster}",
                "flow": 0.05 * (cluster + 1),
            })
    memory = None
    if memory_slots:
        memory = MemoryFrame(
            slots=memory_slots,
            read_slots=(0, 1), read_scores=(0.9, 0.4), read_weights=(0.7, 0.3),
            read_gate=0.8, read_output_rms=0.2,
            write_slots=(2,), write_strength=(0.6,), write_gate=0.5,
            replacement_events=(3,),
            strength=tuple(0.1 * i for i in range(memory_slots)),
            age=tuple(float(i) for i in range(memory_slots)),
            usage=tuple(float(i) for i in range(memory_slots)),
            read_count=tuple(float(i) for i in range(memory_slots)),
            write_count=tuple(float(i) for i in range(memory_slots)),
            occupied=tuple(1.0 if i < memory_slots // 2 else 0.0
                           for i in range(memory_slots)),
        )
    payload = None
    if full:
        payload = {
            f"core.{layer}": {kind: [0.1 * i for i in range(16)]
                              for kind in ("fast", "context", "semantic")}
            for layer in range(layers)
        }
    return NetworkFrame(token_index=token_index, token_id=100 + token_index,
                        nodes=tuple(nodes), edges=tuple(edges), entropy=2.5,
                        memory=memory, full=payload)


@pytest.fixture
def timeline() -> ReplayTimeline:
    return ReplayTimeline([make_frame(index) for index in range(12)])


@pytest.fixture
def inspector(timeline: ReplayTimeline) -> Inspector:
    return Inspector(timeline, mode=ObservationMode.TRACE)


# ----------------------------------------------------------------------
# Replay laden
# ----------------------------------------------------------------------

def test_replay_loads_from_jsonl(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    frame = make_frame()
    events = [
        ObservationEvent(event="network_step", step=0, token_index=0, token_id=7,
                         layer_id="core.0",
                         payload={"nodes": list(frame.nodes), "edges": list(frame.edges)}),
        ObservationEvent(event="prediction", step=0, token_index=0, token_id=7,
                         payload={"entropy": 1.75}),
    ]
    path.write_text("\n".join(json.dumps(event.to_dict()) for event in events) + "\n")
    loaded = ReplayTimeline.from_trace(path)
    assert len(loaded) == 1
    assert loaded[0].entropy == pytest.approx(1.75)
    assert loaded[0].token_id == 7


def test_broken_trace_is_rejected_with_a_clear_error(tmp_path: Path):
    path = tmp_path / "kaputt.jsonl"
    path.write_text("{kein json\n")
    with pytest.raises(ValueError):
        ReplayTimeline.from_trace(path)


# ----------------------------------------------------------------------
# Detailstufen
# ----------------------------------------------------------------------

def test_every_level_reduces_the_node_count(inspector: Inspector):
    counts = []
    for level in (DetailLevel.MODEL, DetailLevel.LAYER, DetailLevel.STATE,
                  DetailLevel.CLUSTER):
        inspector.set_level(level)
        counts.append(len(inspector.scene().nodes))
    assert counts == sorted(counts), f"Stufen müssen aufsteigend feiner werden: {counts}"
    assert counts[0] == 1, "Die Modellstufe fasst alles zu einem Knoten zusammen"


def test_aggregation_preserves_the_total_measured_flow(inspector: Inspector):
    """Aggregieren darf Fluss weder erfinden noch verlieren."""
    totals = []
    for level in (DetailLevel.MODEL, DetailLevel.LAYER, DetailLevel.STATE,
                  DetailLevel.CLUSTER):
        inspector.set_level(level)
        totals.append(sum(node.outgoing_flow for node in inspector.scene().nodes))
    for value in totals[1:]:
        assert value == pytest.approx(totals[0], rel=1e-9)


def test_unit_level_is_unavailable_without_full_telemetry(inspector: Inspector):
    """Ohne Modus ``full`` gibt es keine Unit-Daten – und sie werden nicht erfunden."""
    inspector.set_level(DetailLevel.UNIT)
    scene = inspector.scene()
    assert scene.graph.level == DetailLevel.CLUSTER
    assert scene.graph.downgraded_from == DetailLevel.UNIT
    assert "nicht in der Telemetrie" in inspector.status_line()


def test_unit_level_appears_with_full_telemetry():
    timeline = ReplayTimeline([make_frame(index, full=True) for index in range(3)])
    inspector = Inspector(timeline, mode=ObservationMode.FULL)
    assert DetailLevel.UNIT in available_levels(timeline[0])
    inspector.set_level(DetailLevel.UNIT)
    scene = inspector.scene()
    assert scene.graph.level == DetailLevel.UNIT
    assert scene.graph.downgraded_from is None
    assert all(node.level == DetailLevel.UNIT for node in scene.nodes)


def test_zoom_selects_the_detail_level(inspector: Inspector):
    inspector.auto_level = True
    coarse = inspector.set_zoom(0.2)
    fine = inspector.set_zoom(20.0)
    assert coarse < fine


def test_observation_mode_caps_the_detail_level():
    timeline = ReplayTimeline([make_frame(0, full=True)])
    summary = Inspector(timeline, mode=ObservationMode.SUMMARY)
    summary.set_level(DetailLevel.UNIT)
    assert summary.level() <= DetailLevel.STATE
    assert MODE_CEILING[ObservationMode.SUMMARY] == DetailLevel.STATE


def test_observation_mode_off_shows_nothing():
    timeline = ReplayTimeline([make_frame(0)])
    inspector = Inspector(timeline, mode=ObservationMode.OFF)
    scene = inspector.scene()
    assert scene.nodes == () and scene.edges == ()


# ----------------------------------------------------------------------
# Navigation
# ----------------------------------------------------------------------

def test_token_navigation_stays_in_range(inspector: Inspector):
    assert inspector.seek(-5) == 0
    assert inspector.seek(10_000) == len(inspector) - 1
    inspector.seek(3)
    assert inspector.step(1) == 4
    assert inspector.step(-2) == 2


def test_play_stops_at_the_end(inspector: Inspector):
    inspector.seek(len(inspector) - 2)
    inspector.play()
    assert inspector.advance() is True
    assert inspector.advance() is False
    assert inspector.playing is False


def test_token_context_is_empty_without_token_text(inspector: Inspector):
    """Ohne Tokenliste wird kein Text erfunden."""
    context = inspector.token_context()
    assert context["available"] is False and context["text"] == ""


def test_token_context_marks_the_current_token(timeline: ReplayTimeline):
    tokens = list(range(len(timeline)))
    inspector = Inspector(timeline, tokens=tokens,
                          decode=lambda ids: "".join(chr(97 + int(i) % 26) for i in ids))
    inspector.seek(5)
    context = inspector.token_context(width=3)
    assert context["available"] is True
    assert context["text"][context["caret"]] == chr(97 + 5)


# ----------------------------------------------------------------------
# Auswahl
# ----------------------------------------------------------------------

def test_clicking_selects_the_nearest_node(inspector: Inspector):
    scene = inspector.scene()
    target = scene.nodes[7]
    point = scene.layout.of(target.id)
    assert inspector.select_nearest(point) == target.id


def test_selection_detail_contains_only_measurements(inspector: Inspector):
    inspector.select("core.0.semantic.cluster.2")
    detail = inspector.selection_detail()
    assert detail["id"] == "core.0.semantic.cluster.2"
    assert detail["State"] == "semantic"
    for key in ("Activation", "Delta", "State-Norm", "Persistence",
                "Reactivation", "Incoming Flow", "Outgoing Flow"):
        assert key in detail
    # Es darf keine automatisch erzeugte Deutung geben.
    assert not any("bedeut" in str(key).lower() for key in detail)


def test_selection_history_grows_with_playback(inspector: Inspector):
    inspector.select("core.0.fast.cluster.1")
    for _ in range(5):
        inspector.scene()
        inspector.step(1)
    detail = inspector.selection_detail()
    assert len(detail["Verlauf"]) >= 5


def test_selected_node_survives_a_filter_that_would_hide_it(inspector: Inspector):
    inspector.select("core.0.fast.cluster.0")
    inspector.filters.activity_threshold = 10.0
    ids = {node.id for node in inspector.scene().nodes}
    assert "core.0.fast.cluster.0" in ids


def test_cluster_selection_reports_measured_cluster_values(inspector: Inspector):
    scene = inspector.scene()
    cluster = next(node for node in scene.nodes if node.index is not None)
    inspector.select(cluster.id)
    detail = inspector.selection_detail()
    assert detail["Index"] == cluster.index
    assert detail["Activation"] == pytest.approx(round(cluster.activity, 4))


# ----------------------------------------------------------------------
# Filter
# ----------------------------------------------------------------------

def test_kind_filter_hides_a_whole_state_region(inspector: Inspector):
    before = len(inspector.scene().nodes)
    inspector.filters.toggle_kind("semantic")
    after = inspector.scene()
    assert len(after.nodes) < before
    assert all(node.kind != "semantic" for node in after.nodes)


def test_activity_threshold_reduces_nodes(inspector: Inspector):
    baseline = inspector.scene()
    inspector.filters.activity_threshold = 0.25
    reduced = inspector.scene()
    assert len(reduced.nodes) < len(baseline.nodes)
    assert reduced.reduction > 0.0
    assert all(node.activity >= 0.25 for node in reduced.nodes)


def test_top_n_filter_keeps_exactly_the_most_active(inspector: Inspector):
    inspector.filters.top_nodes = 5
    scene = inspector.scene()
    assert len(scene.nodes) == 5
    everything = sorted(
        build_scene(inspector.frame, DetailLevel.CLUSTER).nodes,
        key=lambda node: -node.activity,
    )[:5]
    assert {node.id for node in scene.nodes} == {node.id for node in everything}


def test_top_flow_filter_keeps_the_strongest_edges(inspector: Inspector):
    inspector.filters.top_flows = 3
    scene = inspector.scene()
    assert len(scene.edges) == 3
    assert scene.edges == tuple(sorted(scene.edges, key=lambda e: -abs(e.flow)))


def test_flow_can_be_switched_off(inspector: Inspector):
    inspector.filters.show_flow = False
    assert inspector.scene().edges == ()


def test_only_reactivated_filter_uses_real_flags(inspector: Inspector):
    inspector.filters.only_reactivated = True
    scene = inspector.scene()
    assert scene.nodes
    assert all(node.reactivation for node in scene.nodes)


def test_only_persistent_filter_uses_real_persistence(inspector: Inspector):
    inspector.filters.only_persistent = True
    inspector.filters.persistence_threshold = 10
    assert all(node.persistence >= 10 for node in inspector.scene().nodes)


def test_filter_description_lists_active_filters(inspector: Inspector):
    assert inspector.filters.describe() == "keine Filter"
    inspector.filters.top_nodes = 4
    inspector.filters.only_changed = True
    text = inspector.filters.describe()
    assert "Top-4" in text and "verändert" in text


# ----------------------------------------------------------------------
# Suche
# ----------------------------------------------------------------------

def test_search_by_layer(inspector: Inspector):
    hits = inspector.search("layer 1")
    assert hits and all(node_id.startswith("core.1.") for node_id in hits)
    assert inspector.selected_node == hits[0]


def test_search_by_cluster(inspector: Inspector):
    hits = inspector.search("cluster 2")
    assert hits and all(node_id.endswith(".cluster.2") for node_id in hits)


def test_search_by_token_jumps(inspector: Inspector):
    inspector.search("token 7")
    assert inspector.index == 7


def test_search_by_substring(inspector: Inspector):
    hits = inspector.search("semantic")
    assert hits and all("semantic" in node_id for node_id in hits)


def test_search_without_match_returns_nothing(inspector: Inspector):
    assert inspector.search("gibtesnicht") == []


def test_search_for_memory_slot():
    timeline = ReplayTimeline([make_frame(0, memory_slots=8)])
    inspector = Inspector(timeline, mode=ObservationMode.TRACE)
    assert inspector.search("slot 3") == ["memory.slot.3"]
    assert inspector.selected_slot == 3
    assert inspector.search("slot 99") == []


# ----------------------------------------------------------------------
# Memory
# ----------------------------------------------------------------------

def test_no_memory_area_without_memory(inspector: Inspector):
    """Ohne Speicher darf kein leerer Speicherbereich erscheinen."""
    assert inspector.scene().memory == {}


def test_memory_area_shows_measured_slot_state():
    timeline = ReplayTimeline([make_frame(0, memory_slots=8)])
    inspector = Inspector(timeline, mode=ObservationMode.TRACE)
    memory = inspector.scene().memory
    assert memory["slots"] == 8
    assert memory["read_active"][0] is True and memory["read_active"][5] is False
    assert memory["write_active"][2] is True
    assert memory["replaced"][3] is True


def test_memory_slot_detail_reports_measurements():
    timeline = ReplayTimeline([make_frame(0, memory_slots=8)])
    inspector = Inspector(timeline, mode=ObservationMode.TRACE)
    inspector.select_slot(1)
    detail = inspector.selection_detail()
    assert detail["typ"] == "memory_slot" and detail["slot"] == 1
    assert detail["active_read"] is True


def test_memory_filter_hides_the_bank():
    timeline = ReplayTimeline([make_frame(0, memory_slots=8)])
    inspector = Inspector(timeline, mode=ObservationMode.TRACE)
    inspector.filters.toggle_kind("memory")
    assert inspector.scene().memory == {}


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

def test_structure_layout_is_stable_across_tokens(inspector: Inspector):
    first = inspector.scene().layout.positions
    inspector.seek(7)
    assert inspector.scene().layout.positions == first


def test_structure_position_depends_only_on_identity():
    node = SceneNode(id="core.2.context.cluster.3", kind="context",
                     level=DetailLevel.CLUSTER, layer=2, index=3, activity=0.9)
    quiet = SceneNode(id="core.2.context.cluster.3", kind="context",
                      level=DetailLevel.CLUSTER, layer=2, index=3, activity=0.0)
    assert structure_layout([node]).of(node.id) == structure_layout([quiet]).of(quiet.id)


def test_activity_layout_places_coactive_clusters_closer():
    import math

    series = {"a": [1, 0, 1, 0, 1, 0], "b": [1, 0, 1, 0, 1, 0], "c": [0, 1, 0, 1, 0, 1]}
    layout = activity_layout(series, iterations=120, seed=3)
    together = math.dist(layout.of("a"), layout.of("b"))
    apart = math.dist(layout.of("a"), layout.of("c"))
    assert together < apart


def test_activity_layout_is_reproducible():
    series = {"a": [1, 0, 1], "b": [0, 1, 0], "c": [1, 1, 0]}
    assert (activity_layout(series, iterations=40, seed=9).positions
            == activity_layout(series, iterations=40, seed=9).positions)


def test_coactivation_of_a_constant_series_is_zero():
    _, matrix = coactivation_matrix({"a": [1, 0, 1, 0], "b": [0.5, 0.5, 0.5, 0.5]})
    assert matrix[0][1] == 0.0


# ----------------------------------------------------------------------
# Live-Betrieb
# ----------------------------------------------------------------------

def test_buffer_accepts_events_without_blocking():
    buffer = TelemetryBuffer(capacity=4)
    for index in range(10):
        buffer(ObservationEvent(event="network_step", step=index, token_index=index))
    stats = buffer.stats()
    assert stats["received"] == 10
    # Sechs Ereignisse mussten weichen – und das wird gezählt, nicht verschwiegen.
    assert stats["dropped"] == 6
    assert stats["pending"] == 4


def test_buffer_drain_empties_the_queue():
    buffer = TelemetryBuffer()
    for index in range(5):
        buffer(ObservationEvent(event="network_step", step=index, token_index=index))
    assert len(buffer.drain()) == 5
    assert len(buffer) == 0


def test_assembler_completes_frames_in_order():
    assembler = FrameAssembler()
    frame = make_frame()
    events = []
    for token in range(3):
        events.append(ObservationEvent(
            event="network_step", step=token, token_index=token, token_id=token,
            layer_id="core.0",
            payload={"nodes": list(frame.nodes), "edges": list(frame.edges)},
        ))
    completed = assembler.feed(events)
    # Der letzte Token ist noch offen, solange kein späterer eingetroffen ist.
    assert [item.token_index for item in completed] == [0, 1]
    assert [item.token_index for item in assembler.flush()] == [2]


def test_live_events_reach_the_inspector():
    from glassmind.visualize.live import GrowingTimeline

    timeline = GrowingTimeline()
    inspector = Inspector(timeline, mode=ObservationMode.TRACE)
    # Vor dem ersten Token gibt es nichts – und es wird nichts erfunden.
    assert inspector.scene().nodes == ()
    timeline.append([make_frame(0), make_frame(1)])
    assert len(inspector) == 2
    assert inspector.scene().nodes


# ----------------------------------------------------------------------
# Vergleich und Eingriffe
# ----------------------------------------------------------------------

def test_comparison_reports_real_differences(timeline: ReplayTimeline):
    quiet = ReplayTimeline([
        NetworkFrame(
            token_index=frame.token_index, token_id=frame.token_id,
            nodes=tuple({**node, "activity": float(node["activity"]) * 0.5}
                        for node in frame.nodes),
            edges=frame.edges, entropy=1.0,
        )
        for frame in timeline.frames
    ])
    left = Inspector(timeline, mode=ObservationMode.TRACE)
    left.attach_comparison(Inspector(quiet, mode=ObservationMode.TRACE))
    report = left.comparison_report()
    assert report["shared_nodes"] > 0
    assert report["mean_absolute_activation_difference"] > 0
    assert all(row["activation_difference"] > 0 for row in report["rows"])


def test_interventions_are_opt_in_and_reversible(inspector: Inspector):
    assert inspector.interventions == {} and inspector.analysis_mode is False
    inspector.set_intervention("ablate_states", ["fast"])
    assert inspector.analysis_mode is True
    assert "ANALYSEMODUS" in inspector.status_line()
    inspector.set_intervention("ablate_states", [])
    assert inspector.analysis_mode is False


def test_unknown_intervention_is_rejected(inspector: Inspector):
    with pytest.raises(ValueError):
        inspector.set_intervention("modell_neu_trainieren", True)


# ----------------------------------------------------------------------
# Große Netze
# ----------------------------------------------------------------------

def test_large_network_stays_workable():
    """Ein künstlich großes Netz – reiner Lasttest, keine Modelltelemetrie."""
    frame = make_frame(layers=40, clusters=64)
    assert len(frame.nodes) == 40 * 3 * 64
    timeline = ReplayTimeline([frame])
    inspector = Inspector(timeline, mode=ObservationMode.TRACE)
    inspector.set_level(DetailLevel.MODEL)
    assert len(inspector.scene().nodes) == 1
    inspector.set_level(DetailLevel.STATE)
    assert len(inspector.scene().nodes) == 40 * 3
    inspector.set_level(DetailLevel.CLUSTER)
    inspector.filters.top_nodes = 500
    assert len(inspector.scene().nodes) == 500


def test_history_is_not_kept_for_huge_networks():
    """Bei sehr großen Netzen darf die Buchhaltung nicht teurer werden als das Bild."""
    frame = make_frame(layers=40, clusters=64)
    inspector = Inspector(ReplayTimeline([frame]), mode=ObservationMode.TRACE)
    inspector.set_level(DetailLevel.CLUSTER)
    inspector.scene()
    assert inspector.history == {}
    inspector.select("core.0.fast.cluster.0")
    inspector.scene()
    assert set(inspector.history) == {"core.0.fast.cluster.0"}


# ----------------------------------------------------------------------
# Fehlende Telemetrie
# ----------------------------------------------------------------------

def test_empty_frame_does_not_crash():
    timeline = ReplayTimeline([NetworkFrame(token_index=0, token_id=None,
                                            nodes=(), edges=())])
    inspector = Inspector(timeline, mode=ObservationMode.TRACE)
    scene = inspector.scene()
    assert scene.nodes == () and scene.edges == ()
    assert inspector.selection_detail() == {}
    assert "Token 1/1" in inspector.status_line()


def test_selection_that_vanishes_on_another_level_is_reported(inspector: Inspector):
    inspector.select("core.0.fast.cluster.3")
    inspector.set_level(DetailLevel.MODEL)
    detail = inspector.selection_detail()
    assert detail["Hinweis"] == "auf dieser Stufe nicht vorhanden"


# ----------------------------------------------------------------------
# Ein echter Fenstertest, wo ein Display vorhanden ist
# ----------------------------------------------------------------------

@pytest.mark.skipif(
    not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
    reason="kein Display verfügbar",
)
def test_window_builds_and_renders(timeline: ReplayTimeline):
    pytest.importorskip("PyQt6")
    pytest.importorskip("vispy")
    from PyQt6 import QtWidgets

    from glassmind.visualize.app import InspectorWindow

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = InspectorWindow(Inspector(timeline, mode=ObservationMode.TRACE))
    window.refresh()
    assert "Token 1/12" in window.statusBar().currentMessage()
    window._step(1)
    assert window.inspector.index == 1
    window.top_nodes.setValue(3)
    window.refresh()
    assert "3/24 Knoten" in window.statusBar().currentMessage()
    window.close()
    application.processEvents()
