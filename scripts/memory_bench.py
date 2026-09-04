#!/usr/bin/env python3
"""Milestone 3: misst, was das externe Memory tatsächlich kostet.

Der wichtigste Nachweis steht zuerst: Der Pfad *ohne* Speicher darf gegenüber
Milestone 2.6 nicht langsamer geworden sein. Danach werden die Kosten des
aktivierten Speichers getrennt nach Lesen, Schreiben und Beobachtung
ausgewiesen – gemessen, nicht geschätzt.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import ObservationBus, ObservationMode
from glassmind.utils.device import (
    autocast_context,
    detect_device,
    peak_memory_bytes,
    reset_peak_memory,
    synchronize,
)
from glassmind.utils.reproducibility import environment_metadata, seed_everything


def throughput(model, capabilities, *, length: int, iterations: int = 5, **options) -> float:
    tokens = torch.randint(0, model.config.vocab_size, (1, length), device=capabilities.torch_device)
    with torch.inference_mode(), autocast_context(capabilities):
        for _ in range(3):
            model(tokens, **options)
        synchronize(capabilities)
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(iterations):
                model(tokens, **options)
            synchronize(capabilities)
            samples.append(tokens.numel() * iterations / (time.perf_counter() - started))
    return statistics.median(samples)


def streaming(model, capabilities, *, steps: int = 128, **options) -> dict[str, float]:
    token = torch.zeros(1, dtype=torch.long, device=capabilities.torch_device)
    with torch.inference_mode(), autocast_context(capabilities):
        state = None
        for _ in range(16):
            _, state = model.step(token, state, **options)
        synchronize(capabilities)
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(steps):
                _, state = model.step(token, state, **options)
            synchronize(capabilities)
            samples.append((time.perf_counter() - started) * 1000 / steps)
    return {"median": statistics.median(samples), "min": min(samples), "max": max(samples)}


def training_throughput(model, capabilities, *, batch: int = 16, length: int = 96, steps: int = 10) -> float:
    """Kurzer Trainingsdurchsatz; die Gewichte werden danach wiederhergestellt."""
    import torch.nn.functional as F

    snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
    was_training = model.training
    try:
        model.train()
        tokens = torch.randint(0, model.config.vocab_size, (batch, length), device=capabilities.torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def step() -> None:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tokens)
            F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size).float(), tokens.reshape(-1)
            ).backward()
            optimizer.step()

        for _ in range(2):
            step()
        synchronize(capabilities)
        started = time.perf_counter()
        for _ in range(steps):
            step()
        synchronize(capabilities)
        return tokens.numel() * steps / (time.perf_counter() - started)
    finally:
        model.load_state_dict(snapshot)
        model.train(was_training)


def operation_counts(model, capabilities, *, length: int = 8, **options) -> dict[str, float]:
    """ATen-Operationen je Token – die Größe, die bei GlassMind zählt."""
    from torch.utils._python_dispatch import TorchDispatchMode

    counter: Counter[str] = Counter()

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counter[str(func)] += 1
            return func(*args, **(kwargs or {}))

    tokens = torch.randint(0, model.config.vocab_size, (1, length), device=capabilities.torch_device)
    with torch.inference_mode(), autocast_context(capabilities), Count():
        model(tokens, **options)
    total = sum(counter.values())
    sparse = sum(count for name, count in counter.items()
                 if any(key in name for key in ("topk", "gather", "scatter")))
    return {"per_token": total / length, "sparse_per_token": sparse / length}


def memory_bandwidth(config: ModelConfig, *, read_k: int, write_k: int) -> dict[str, float]:
    """Wie viele Bytes ein Zugriff wirklich bewegt.

    Getrennt ausgewiesen: die *sparsen* Anteile (nur Top-K Werte) und das
    Ähnlichkeitsscoring, das über alle Slots läuft. Letzteres ist bei 64 bis
    128 Slots bewusst dicht – eine echte Suchstruktur wäre hier langsamer.
    """
    element = 4  # float32
    scoring = config.memory_slots * config.memory_key_dim * element
    sparse_read = read_k * config.memory_width * element
    sparse_write = write_k * (config.memory_width + config.memory_key_dim) * element
    dense_equivalent = config.memory_slots * config.memory_width * element
    return {
        "scoring_bytes_per_token": scoring,
        "sparse_read_bytes_per_token": sparse_read,
        "sparse_write_bytes_per_token": sparse_write,
        "dense_read_equivalent_bytes": dense_equivalent,
        "read_sparsity_factor": dense_equivalent / max(sparse_read, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 3: Kosten des externen Memory")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--memory-width", type=int, default=64)
    parser.add_argument("--memory-key-dim", type=int, default=32)
    parser.add_argument("--read-k", type=int, default=2)
    parser.add_argument("--write-k", type=int, default=1)
    parser.add_argument("--slot-sweep", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/milestone3-cost.json"))
    args = parser.parse_args()

    seed_everything(23)
    capabilities = detect_device(args.device, "auto")
    base = dict(vocab_size=260, d_model=args.d_model, n_layers=args.layers,
                telemetry_clusters=4, state_interactions=True)
    memory = dict(memory_slots=args.slots, memory_width=args.memory_width,
                  memory_key_dim=args.memory_key_dim, memory_read_k=args.read_k,
                  memory_write_k=args.write_k)

    def build(**overrides) -> GlassMindLM:
        return GlassMindLM(ModelConfig(**{**base, **overrides})).to(capabilities.torch_device).eval()

    print(f"[Kosten] Backend={capabilities.backend}  Gerät={capabilities.name}")
    print(f"[Kosten] d_model={args.d_model}, {args.layers} Blöcke, Länge {args.length}, "
          f"{args.slots} Slots, read_k={args.read_k}, write_k={args.write_k}\n")

    variants: list[tuple[str, dict[str, Any], dict[str, Any]]] = [
        ("M2.6 ohne Memory", {}, {}),
        ("M3 Memory deaktiviert", memory, dict(disable_memory=True)),
        ("M3 Memory aktiv", memory, {}),
        ("M3 nur Lesen (write aus)", memory, dict(disable_memory_write=True)),
        ("M3 nur Schreiben (read aus)", memory, dict(disable_memory_read=True)),
        ("M3 mit Zugriffszählern", {**memory, "memory_track_usage": True}, {}),
    ]
    records = []
    baseline = None
    header = (f"{'Variante':30s} {'Tok/s':>8s} {'ms/Tok':>8s} {'Train Tok/s':>12s} "
              f"{'VRAM':>10s} {'Ops/Tok':>8s} {'gegen M2.6':>11s}")
    print(header)
    print("-" * len(header))
    for label, overrides, options in variants:
        model = build(**overrides)
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
        reset_peak_memory(capabilities)
        tokens_per_second = throughput(model, capabilities, length=args.length, **options)
        stream = streaming(model, capabilities, **options)
        peak = peak_memory_bytes(capabilities)
        counts = operation_counts(model, capabilities, **options)
        training = training_throughput(model, capabilities) if not options else None
        if baseline is None:
            baseline = tokens_per_second
        change = 100 * (tokens_per_second / baseline - 1)
        records.append({
            "label": label, "options": {k: bool(v) for k, v in options.items()},
            "tokens_per_second": tokens_per_second,
            "streaming_ms_per_token": stream["median"],
            "streaming_ms_min": stream["min"], "streaming_ms_max": stream["max"],
            "training_tokens_per_second": training,
            "peak_memory_bytes": peak,
            "operations_per_token": counts["per_token"],
            "sparse_operations_per_token": counts["sparse_per_token"],
            "change_vs_baseline_percent": change,
            "parameter_count": model.parameter_count,
        })
        print(f"{label:30s} {tokens_per_second:8,.0f} {stream['median']:8.3f} "
              f"{(f'{training:,.0f}' if training else '–'):>12s} "
              f"{(f'{peak/2**20:.1f} MiB' if peak else '–'):>10s} "
              f"{counts['per_token']:8.1f} {change:+10.1f}%")
        del model
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()

    # Beobachtungs-Overhead getrennt.
    print()
    model = build(**memory)
    plain = throughput(model, capabilities, length=args.length)
    observed = []
    for mode in (ObservationMode.SUMMARY, ObservationMode.TRACE):
        tokens = torch.randint(0, 260, (1, args.length), device=capabilities.torch_device)
        with torch.inference_mode(), autocast_context(capabilities):
            model(tokens, observer=ObservationBus(mode))
            synchronize(capabilities)
            started = time.perf_counter()
            for _ in range(3):
                model(tokens, observer=ObservationBus(mode))
            synchronize(capabilities)
            rate = tokens.numel() * 3 / (time.perf_counter() - started)
        observed.append({"mode": mode.name.lower(), "tokens_per_second": rate,
                         "overhead_percent": 100 * (1 - rate / plain)})
        print(f"  Beobachtung {mode.name.lower():8s} {rate:8,.0f} Tok/s  "
              f"Overhead {100 * (1 - rate / plain):5.1f}%")
    del model

    # Kostenskalierung über die Slotzahl.
    print()
    sweep = []
    for slots in args.slot_sweep:
        model = build(**{**memory, "memory_slots": slots})
        rate = throughput(model, capabilities, length=args.length)
        sweep.append({"slots": slots, "tokens_per_second": rate})
        print(f"  {slots:4d} Slots: {rate:8,.0f} Tok/s")
        del model
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()

    bandwidth = memory_bandwidth(ModelConfig(**{**base, **memory}),
                                 read_k=args.read_k, write_k=args.write_k)
    print(f"\n  Bandbreite je Token: Scoring {bandwidth['scoring_bytes_per_token']:,.0f} B über alle Slots, "
          f"sparses Lesen {bandwidth['sparse_read_bytes_per_token']:,.0f} B, "
          f"sparses Schreiben {bandwidth['sparse_write_bytes_per_token']:,.0f} B")
    print(f"  Ein dichtes Lesen aller Slots wäre {bandwidth['dense_read_equivalent_bytes']:,.0f} B "
          f"({bandwidth['read_sparsity_factor']:.0f}× mehr)")

    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "milestone": "3",
        "environment": environment_metadata(capabilities, seed=23),
        "settings": {k: v for k, v in vars(args).items() if k != "output"},
        "variants": records,
        "observation": observed,
        "slot_sweep": sweep,
        "bandwidth": bandwidth,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[Ergebnis] {args.output}")


if __name__ == "__main__":
    main()
