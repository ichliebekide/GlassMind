#!/usr/bin/env python3
"""Milestone 4.5: reproduzierbare Demo-Replays und ein Rendering-Test.

Erzeugt Traces, die sich unmittelbar im Visual Inspector öffnen lassen, und
rendert einen davon in eine PNG-Datei. Der Rendering-Test ist automatisch: Er
prüft, dass ein Bild entsteht und dass es nicht einfarbig ist – ein leeres oder
schwarzes Bild wäre ein stiller Fehlschlag.

Alle erzeugten Traces stammen aus echten Modelldurchläufen. Es gibt hier keine
synthetischen Aktivitäten; der Lasttest mit künstlichen Netzen sitzt in
``scripts/visual_bench.py`` und ist dort ausdrücklich als solcher benannt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import torch

from glassmind.data.tokenizer import ByteTokenizer
from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import ObservationBus, ObservationMode
from glassmind.observe.recorder import JSONLRecorder
from glassmind.utils.reproducibility import seed_everything


def record(
    model: GlassMindLM, tokens: torch.Tensor, path: Path, mode: ObservationMode,
    *, generate: int = 0,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    bus = ObservationBus(mode)
    recorder = JSONLRecorder(path)
    bus.subscribe(recorder)
    model.eval()
    with torch.inference_mode():
        if generate:
            model.generate(tokens, generate, temperature=0.0, observer=bus)
        else:
            model(tokens, observer=bus)
    bus.close()
    return {"path": str(path), "bytes": path.stat().st_size,
            "events": bus.events_emitted, "mode": mode.name.lower()}


def demo_tiny(args, console) -> dict[str, Any]:
    """Kleines Modell mit Speicher – zeigt alle Bereiche inklusive Memory-Bank."""
    seed_everything(args.seed)
    config = ModelConfig(
        vocab_size=64, d_model=32, n_layers=2, telemetry_clusters=4,
        state_interactions=True, memory_slots=32, memory_width=16,
        memory_key_dim=8, memory_read_k=2, memory_write_k=1,
        memory_track_usage=True,
    )
    model = GlassMindLM(config)
    tokens = torch.randint(0, 64, (1, args.tokens), generator=torch.Generator().manual_seed(args.seed))
    info = record(model, tokens, args.out / "demo-tiny-memory.jsonl", ObservationMode.TRACE)
    info["beschreibung"] = (
        "Tiny-Modell mit 32 Memory-Slots. Zeigt State-Regionen, Flusskanten und "
        "die Speicherbank mit echten Lese- und Schreibzugriffen."
    )
    info["parameter"] = model.parameter_count
    console(f"  {info['path']}  {info['events']:,} Ereignisse  {info['bytes']/1e6:.2f} MB")
    return info


def demo_full(args, console) -> dict[str, Any]:
    """Trace im Modus ``full`` – nur damit ist die Unit-Detailstufe belegt."""
    seed_everything(args.seed)
    config = ModelConfig(vocab_size=64, d_model=64, n_layers=2, telemetry_clusters=4,
                         state_interactions=True)
    model = GlassMindLM(config)
    tokens = torch.randint(0, 64, (1, max(8, args.tokens // 4)),
                           generator=torch.Generator().manual_seed(args.seed))
    info = record(model, tokens, args.out / "demo-full-units.jsonl", ObservationMode.FULL)
    info["beschreibung"] = (
        "Modus full: enthält vollständige Zustandsvektoren und schaltet damit "
        "die Detailstufe 'Units' frei. Bewusst kurz, weil dieser Modus je Token "
        "die kompletten Zustände überträgt."
    )
    info["parameter"] = model.parameter_count
    console(f"  {info['path']}  {info['events']:,} Ereignisse  {info['bytes']/1e6:.2f} MB")
    return info


def demo_language(args, console) -> dict[str, Any]:
    """Ein echtes, auf Sprache trainiertes Modell aus Milestone 4.

    Existiert noch kein Milestone-4-Checkpoint, wird das offen gemeldet statt
    ein untrainiertes Modell als Sprachmodell auszugeben.
    """
    candidates = sorted(args.checkpoints.glob("m4-*.pt")) if args.checkpoints.exists() else []
    if not candidates:
        console(f"  kein Milestone-4-Checkpoint unter {args.checkpoints} – "
                f"Sprach-Replay wird nachgereicht, sobald M4 gelaufen ist")
        return {"skipped": f"kein Checkpoint in {args.checkpoints}"}
    from glassmind.training.checkpoint import load_checkpoint

    source = candidates[0]
    model, tokenizer, meta = load_checkpoint(source, device="cpu")
    prompt = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long)
    info = record(model, prompt, args.out / "demo-language.jsonl",
                  ObservationMode.TRACE, generate=args.generate)
    info["beschreibung"] = (
        f"Sprachmodell aus {source.name}, Prompt {args.prompt!r}, "
        f"{args.generate} erzeugte Token."
    )
    info["checkpoint"] = str(source)
    info["parameter"] = model.parameter_count
    console(f"  {info['path']}  {info['events']:,} Ereignisse  "
            f"aus {source.name} ({model.parameter_count:,} Parameter)")
    return info


def render_test(args, console, trace: Path) -> dict[str, Any]:
    """Rendert einen Trace in eine PNG-Datei und prüft, dass ein Bild entsteht."""
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        console("  kein Display – Rendering-Test übersprungen")
        return {"skipped": "kein Display"}
    # Gemessene Plattformgrenze: Unter der nativen Wayland-Plattform schlägt
    # ``canvas.render`` mit "FrameBuffer attachments are incomplete" fehl, weil
    # der Framebuffer-Readback dort nicht vollständig ist. Unter XWayland
    # (``xcb``) funktioniert derselbe Code. Betroffen ist ausschließlich das
    # Abgreifen eines Bildes – die interaktive Oberfläche läuft auf Wayland
    # unverändert.
    if os.environ.get("WAYLAND_DISPLAY") and "QT_QPA_PLATFORM" not in os.environ:
        if os.environ.get("DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            console("  Hinweis: für den Bild-Export auf XWayland (xcb) umgestellt")
        else:
            console("  reines Wayland ohne XWayland – Bild-Export nicht möglich")
            return {"skipped": "Wayland ohne XWayland: kein Framebuffer-Readback"}
    import numpy as np
    from vispy import scene as vispy_scene

    from glassmind.observe.bus import ObservationMode as Mode
    from glassmind.visualize.graph import ReplayTimeline
    from glassmind.visualize.inspector import Inspector
    from glassmind.visualize.render import DARK_BACKGROUND, NetworkRenderer

    inspector = Inspector(ReplayTimeline.from_trace(trace), mode=Mode.TRACE)
    inspector.seek(min(args.render_token, len(inspector) - 1))
    # Die Leinwand muss sichtbar sein: Ein verstecktes Fenster hat auf vielen
    # Treibern keinen vollständigen Framebuffer, und ``render`` schlägt dann
    # mit "FrameBuffer attachments are incomplete" fehl.
    canvas = vispy_scene.SceneCanvas(size=(1600, 900), bgcolor=DARK_BACKGROUND, show=True)
    view = canvas.central_widget.add_view()
    view.camera = "panzoom"
    renderer = NetworkRenderer(view)
    scene = inspector.scene()
    renderer.update(inspector, scene)
    xmin, ymin, xmax, ymax = scene.layout.bounds
    view.camera.rect = (xmin, ymin, max(xmax - xmin, 1e-3), max(ymax - ymin, 1e-3))
    canvas.app.process_events()
    image = canvas.render(alpha=False)
    canvas.close()

    destination = args.out / "inspector-screenshot.png"
    try:
        import imageio.v3 as imageio

        imageio.imwrite(destination, image)
    except ImportError:
        from vispy.io import write_png

        write_png(str(destination), image)

    unique = int(np.unique(image.reshape(-1, image.shape[-1]), axis=0).shape[0])
    background = np.array([int(DARK_BACKGROUND[i:i + 2], 16) for i in (1, 3, 5)])
    non_background = int(
        (np.abs(image[..., :3].astype(int) - background).sum(axis=-1) > 12).sum()
    )
    result = {
        "path": str(destination),
        "size": list(image.shape),
        "unique_colors": unique,
        "non_background_pixels": non_background,
        "drawn_nodes": len(scene.nodes),
        "token": inspector.index,
    }
    # Ein einfarbiges Bild wäre ein stiller Fehlschlag – deshalb geprüft.
    result["passed"] = unique > 8 and non_background > 500
    console(f"  {destination}  {image.shape[1]}x{image.shape[0]}  "
            f"{unique:,} Farben  {non_background:,} gezeichnete Pixel  "
            f"{'BESTANDEN' if result['passed'] else 'FEHLGESCHLAGEN'}")
    if not result["passed"]:
        raise SystemExit("Rendering-Test fehlgeschlagen: das Bild ist praktisch leer")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4.5: Demo-Replays erzeugen")
    parser.add_argument("--out", type=Path, default=Path("runs/milestone4_5"))
    parser.add_argument("--checkpoints", type=Path, default=Path("runs/milestone4"))
    parser.add_argument("--tokens", type=int, default=48)
    parser.add_argument("--generate", type=int, default=48)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--seed", type=int, default=45)
    parser.add_argument("--render-token", type=int, default=12)
    parser.add_argument("--output", type=Path,
                        default=Path("benchmarks/milestone4_5-demos.json"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []

    def console(message: str) -> None:
        print(message, flush=True)
        lines.append(message)

    console("=== Demo-Replays ===")
    demos = {
        "tiny_memory": demo_tiny(args, console),
        "full_units": demo_full(args, console),
        "language": demo_language(args, console),
    }
    console("\n=== Rendering-Test ===")
    screenshot = render_test(args, console, Path(demos["tiny_memory"]["path"]))

    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "milestone": "4.5",
        "demos": demos,
        "screenshot": screenshot,
        "hinweis": "Alle Traces stammen aus echten Modelldurchläufen.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    console(f"\n[Ergebnis] {args.output}")
    console("\nÖffnen mit:")
    for demo in demos.values():
        if "path" in demo:
            console(f"  python -m glassmind.visualize.app --replay {demo['path']}")


if __name__ == "__main__":
    main()
