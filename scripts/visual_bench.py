#!/usr/bin/env python3
"""Milestone 4.5: was der Visual Inspector wirklich leistet.

Drei getrennte Messungen, weil drei verschiedene Dinge langsam sein können:

``render``   Bildrate über wachsende Knotenzahl. Die Netzstruktur ist hier
             synthetisch – ausdrücklich als Lasttest gekennzeichnet, nicht als
             Modelltelemetrie. Sie beantwortet allein die Frage, wie viele
             Knoten der Zeichenpfad trägt.
``pipeline`` Kosten der Inspector-Logik allein: aggregieren, filtern,
             anordnen. Ohne Fenster, damit die Zahl nicht vom Treiber abhängt.
``live``     Zusatzkosten der Telemetrie im Modellpfad. Genau das ist der
             Wert, der die Anforderung "die Visualisierung darf die Inferenz
             nicht blockieren" belegt oder widerlegt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from glassmind.observe.bus import ObservationBus, ObservationMode
from glassmind.visualize.graph import NetworkFrame, ReplayTimeline
from glassmind.visualize.inspector import Inspector
from glassmind.visualize.scene import DetailLevel


def synthetic_frame(nodes: int, *, seed: int = 5, clusters_per_state: int | None = None
                    ) -> NetworkFrame:
    """Ein künstlich großes Netz – nur für den Lasttest des Renderers.

    Dies ist **keine** Modelltelemetrie und wird nirgends als solche
    dargestellt. Der Zweck ist allein, den Zeichenpfad zu belasten.
    """
    generator = np.random.default_rng(seed)
    kinds = ("fast", "context", "semantic")
    per_state = clusters_per_state or max(1, nodes // (3 * 8))
    layers = max(1, nodes // max(3 * per_state, 1))
    raw_nodes: list[dict[str, Any]] = []
    for layer in range(layers):
        for kind in kinds:
            for cluster in range(per_state):
                activity = float(generator.random() * 0.6)
                raw_nodes.append({
                    "id": f"core.{layer}.{kind}.cluster.{cluster}",
                    "kind": kind,
                    "activity": activity,
                    "persistence": int(generator.integers(0, 40)),
                    "reactivation": bool(generator.random() < 0.05),
                    "components": {
                        "delta_rms": float(generator.random() * 0.3),
                        "state_norm": float(generator.random() * 4.0),
                        "gate_mean": float(generator.random()),
                        "incoming_flow_rms": float(generator.random() * 0.4),
                    },
                })
    edges = []
    for layer in range(layers):
        for index in range(per_state):
            edges.append({
                "id": f"core.{layer}.fast_context.{index}",
                "source": f"core.{layer}.fast.cluster.{index}",
                "target": f"core.{layer}.context.cluster.{index}",
                "flow": float(generator.random() * 0.5),
            })
            edges.append({
                "id": f"core.{layer}.context_semantic.{index}",
                "source": f"core.{layer}.context.cluster.{index}",
                "target": f"core.{layer}.semantic.cluster.{index}",
                "flow": float(generator.random() * 0.5),
            })
    return NetworkFrame(token_index=0, token_id=1, nodes=tuple(raw_nodes),
                        edges=tuple(edges))


class _SyntheticTimeline(ReplayTimeline):
    def __init__(self, frame: NetworkFrame, length: int = 8) -> None:
        self.frames = [frame] * length


def section_pipeline(args, console) -> dict[str, Any]:
    """Kosten der Inspector-Logik ohne jedes Fenster."""
    console(f"\n=== Inspector-Logik ohne Fenster ===")
    records = []
    for count in args.nodes:
        frame = synthetic_frame(count)
        inspector = Inspector(_SyntheticTimeline(frame), mode=ObservationMode.TRACE)
        inspector.set_level(DetailLevel.CLUSTER)
        inspector.scene()
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            scene = inspector.scene()
            samples.append(time.perf_counter() - started)
        median = statistics.median(samples)
        entry = {
            "requested_nodes": count,
            "actual_nodes": len(frame.nodes),
            "drawn_nodes": len(scene.nodes),
            "drawn_edges": len(scene.edges),
            "seconds_per_scene": median,
            "scenes_per_second": 1.0 / median if median else float("inf"),
        }
        # Was bringt eine Reduktion? Derselbe Frame mit Top-N-Filter.
        inspector.filters.top_nodes = min(2000, len(frame.nodes))
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            filtered = inspector.scene()
            samples.append(time.perf_counter() - started)
        entry["filtered_nodes"] = len(filtered.nodes)
        entry["filtered_seconds_per_scene"] = statistics.median(samples)
        records.append(entry)
        console(f"  {len(frame.nodes):>7,d} Knoten  {median*1000:8.2f} ms/Szene  "
                f"{entry['scenes_per_second']:8.1f} Szenen/s  "
                f"gefiltert auf {entry['filtered_nodes']:>6,d}: "
                f"{entry['filtered_seconds_per_scene']*1000:7.2f} ms")
    return {"levels": records}


def section_lod(args, console) -> dict[str, Any]:
    """Wie stark die Detailstufen die gezeichnete Menge reduzieren."""
    console("\n=== Level of Detail: Reduktion je Stufe ===")
    frame = synthetic_frame(max(args.nodes))
    inspector = Inspector(_SyntheticTimeline(frame), mode=ObservationMode.TRACE)
    records = []
    for level in (DetailLevel.MODEL, DetailLevel.LAYER, DetailLevel.STATE,
                  DetailLevel.CLUSTER):
        inspector.set_level(level)
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            scene = inspector.scene()
            samples.append(time.perf_counter() - started)
        median = statistics.median(samples)
        records.append({
            "level": level.name.lower(),
            "nodes": len(scene.nodes),
            "edges": len(scene.edges),
            "seconds_per_scene": median,
            "total_flow": sum(node.outgoing_flow for node in scene.nodes),
        })
        console(f"  {level.label:14s} {len(scene.nodes):>7,d} Knoten  "
                f"{len(scene.edges):>7,d} Kanten  {median*1000:7.2f} ms/Szene  "
                f"Σ Fluss {records[-1]['total_flow']:.4f}")
    return {"source_nodes": len(frame.nodes), "levels": records}


def section_render(args, console) -> dict[str, Any]:
    """Echte Bildrate mit GPU-Rendering."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        console("\n[Render] Kein Display – übersprungen")
        return {"skipped": "kein Display"}
    try:
        from PyQt6 import QtWidgets
        from vispy import scene as vispy_scene

        from glassmind.visualize.render import DARK_BACKGROUND, NetworkRenderer
    except ImportError as exc:
        console(f"\n[Render] {exc} – übersprungen")
        return {"skipped": str(exc)}

    console("\n=== Zeichenpfad: echte Bildrate ===")
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    canvas = vispy_scene.SceneCanvas(size=(1400, 900), bgcolor=DARK_BACKGROUND, show=True)
    view = canvas.central_widget.add_view()
    view.camera = "panzoom"
    renderer = NetworkRenderer(view)
    records = []
    for count in args.nodes:
        frame = synthetic_frame(count)
        inspector = Inspector(_SyntheticTimeline(frame), mode=ObservationMode.TRACE)
        inspector.set_level(DetailLevel.CLUSTER)
        scene = inspector.scene()
        xmin, ymin, xmax, ymax = scene.layout.bounds
        view.camera.rect = (xmin, ymin, max(xmax - xmin, 1e-3), max(ymax - ymin, 1e-3))
        stats = renderer.update(inspector, scene)
        canvas.update()
        application.processEvents()
        samples = []
        for _ in range(args.frames):
            started = time.perf_counter()
            renderer.update(inspector, scene)
            canvas.update()
            application.processEvents()
            canvas.render()
            samples.append(time.perf_counter() - started)
        median = statistics.median(samples)
        entry = {
            "nodes": len(scene.nodes),
            "edges": len(scene.edges),
            "seconds_per_frame": median,
            "fps": 1.0 / median if median else float("inf"),
            "draw_calls": stats["draw_calls"],
            "process_rss_bytes": _rss(),
        }
        records.append(entry)
        console(f"  {len(scene.nodes):>7,d} Knoten  {len(scene.edges):>7,d} Kanten  "
                f"{median*1000:7.2f} ms/Bild  {entry['fps']:7.1f} FPS  "
                f"{stats['draw_calls']} Zeichenaufrufe  RSS {_rss()/1e6:.0f} MB")
    canvas.close()
    return {"levels": records}


def section_live(args, console) -> dict[str, Any]:
    """Wie viel kostet die Telemetrie im Modellpfad?"""
    import torch

    from glassmind.model import GlassMindLM, ModelConfig
    from glassmind.utils.reproducibility import seed_everything
    from glassmind.visualize.live import TelemetryBuffer

    console("\n=== Telemetrie-Overhead im Modellpfad ===")
    seed_everything(11)
    config = ModelConfig(vocab_size=64, d_model=args.d_model, n_layers=args.layers,
                         telemetry_clusters=4, state_interactions=True)
    model = GlassMindLM(config).eval()
    tokens = torch.randint(0, 64, (1, args.length))

    def one_round(mode: ObservationMode | None, sink: Any = None) -> float:
        bus = None
        if mode is not None:
            bus = ObservationBus(mode)
            if sink is not None:
                bus.subscribe(sink)
        started = time.perf_counter()
        with torch.inference_mode():
            for _ in range(args.repeats):
                model(tokens, observer=bus) if bus else model(tokens)
        rate = tokens.numel() * args.repeats / (time.perf_counter() - started)
        if bus is not None:
            bus.close()
        return rate

    # Referenz und Messung werden *verschränkt* erhoben. Eine einmal zu Beginn
    # gemessene Referenz driftet mit Takt und Cache-Zustand der Maschine – das
    # ergab beim ersten Versuch einen negativen Overhead, also einen
    # offensichtlich unmöglichen Wert. Abwechselnd zu messen und den Median zu
    # nehmen beseitigt genau diese Drift.
    rounds = 5
    variants: list[tuple[str, ObservationMode | None]] = [("ohne Bus", None)]
    variants += [(mode.name.lower(), mode) for mode in
                 (ObservationMode.OFF, ObservationMode.SUMMARY,
                  ObservationMode.TRACE, ObservationMode.FULL)]
    samples: dict[str, list[float]] = {name: [] for name, _ in variants}
    buffers: dict[str, Any] = {}
    for name, mode in variants:
        one_round(mode)  # Aufwärmen
    for _ in range(rounds):
        for name, mode in variants:
            sink = None
            if mode is not None:
                sink = TelemetryBuffer()
                buffers[name] = sink
            samples[name].append(one_round(mode, sink))

    baseline = statistics.median(samples["ohne Bus"])
    records = []
    for name, mode in variants:
        rate = statistics.median(samples[name])
        entry = {
            "mode": name,
            "tokens_per_second": rate,
            "tokens_per_second_spread": max(samples[name]) - min(samples[name]),
            "overhead_percent": 100 * (1 - rate / baseline),
        }
        if name in buffers:
            entry["events"] = buffers[name].received
            entry["dropped"] = buffers[name].dropped
        records.append(entry)
        console(f"  {name:9s} {rate:>9,.0f} tok/s  "
                f"Overhead {entry['overhead_percent']:6.1f} %  "
                f"Streuung {entry['tokens_per_second_spread']:>7,.0f} tok/s  "
                f"Ereignisse {entry.get('events', 0):>7,d}  "
                f"verworfen {entry.get('dropped', 0)}")
    return {"baseline_tokens_per_second": baseline, "modes": records,
            "rounds": rounds, "config": config.to_dict(), "length": args.length}


def _rss() -> int:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


SECTIONS = {"pipeline": section_pipeline, "lod": section_lod,
            "render": section_render, "live": section_live}


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4.5: Visual Inspector profilieren")
    parser.add_argument("--nodes", type=int, nargs="+",
                        default=[1_000, 10_000, 50_000, 100_000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/milestone4_5-visual.json"))
    parser.add_argument("--sections", nargs="+",
                        default=["pipeline", "lod", "render", "live"],
                        choices=sorted(SECTIONS))
    args = parser.parse_args()

    lines: list[str] = []

    def console(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    payload: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "milestone": "4.5",
        "settings": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items() if k != "output"},
        "note": (
            "Die Netzstrukturen im Abschnitt render/pipeline/lod sind synthetische "
            "Lasttests, keine Modelltelemetrie. Der Abschnitt live misst echten "
            "Modellcode."
        ),
        "sections": {},
    }
    for name in args.sections:
        payload["sections"][name] = SECTIONS[name](args, console)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    console(f"\n[Ergebnis] {args.output}")


if __name__ == "__main__":
    main()
