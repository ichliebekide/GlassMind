#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import resource
import statistics
import time

import torch

from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import ObservationBus, ObservationMode
from glassmind.utils.device import autocast_context, detect_device, peak_memory_bytes, reset_peak_memory, synchronize
from glassmind.utils.reproducibility import seed_everything


def load_baseline(path: Path | None) -> dict[tuple[int, str], dict[str, object]]:
    if path is None:
        return {}
    records: dict[tuple[int, str], dict[str, object]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            key = (int(record["sequence_length"]), str(record["observation_mode"]))
            records[key] = record
    return records


def host_peak_rss_bytes() -> int | None:
    """Maximaler residenter Hauptspeicher des Prozesses, sofern verfügbar."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):  # pragma: no cover - plattformabhängig
        return None
    # Linux meldet Kibibyte, macOS Byte.
    return int(usage) * (1 if usage > 2**32 else 1024)


def measure(
    model: GlassMindLM,
    inputs: torch.Tensor,
    capabilities: object,
    mode: ObservationMode,
    iterations: int,
) -> dict[str, object]:
    observer = ObservationBus(mode)
    reset_peak_memory(capabilities)
    with torch.inference_mode(), autocast_context(capabilities):
        model(inputs, observer=observer)
        synchronize(capabilities)
        started = time.perf_counter()
        for _ in range(iterations):
            model(inputs, observer=observer)
        synchronize(capabilities)
    elapsed = time.perf_counter() - started
    token_count = inputs.numel() * iterations
    return {
        "observation_mode": mode.name.lower(),
        "tokens_per_second": token_count / elapsed,
        "elapsed_seconds": elapsed,
        "peak_memory_bytes": peak_memory_bytes(capabilities),
        "host_peak_rss_bytes": host_peak_rss_bytes(),
        "events_emitted": observer.events_emitted,
    }


def streaming_latency(
    model: GlassMindLM,
    token: torch.Tensor,
    capabilities: object,
    steps: int = 128,
    repeats: int = 5,
) -> dict[str, float]:
    """Median über mehrere Messreihen; Einzelreihen schwanken hier deutlich."""
    state = None
    samples: list[float] = []
    with torch.inference_mode(), autocast_context(capabilities):
        for _ in range(16):
            _, state = model.step(token, state)
        synchronize(capabilities)
        for _ in range(repeats):
            started = time.perf_counter()
            for _ in range(steps):
                _, state = model.step(token, state)
            synchronize(capabilities)
            samples.append((time.perf_counter() - started) * 1000 / steps)
    return {
        "streaming_latency_ms_per_token": statistics.median(samples),
        "streaming_latency_ms_per_token_min": min(samples),
        "streaming_latency_ms_per_token_max": max(samples),
        "streaming_latency_repeats": repeats,
        "streaming_latency_steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Misst GlassMind-Durchsatz und Beobachtungs-Overhead")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument(
        "--state-interactions",
        action="store_true",
        help="Aktiviert den bounded State-Interaction-Pfad aus Milestone 2",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--require-compile",
        action="store_true",
        help="Bricht ab, wenn torch.compile nicht übersetzt, statt auf Eager zurückzufallen",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Optionaler JSONL-Ausgangswert; ergänzt den prozentualen Durchsatzvergleich",
    )
    parser.add_argument("--output", type=Path, default=Path("benchmarks/latest.jsonl"))
    args = parser.parse_args()
    seed_everything(23)
    capabilities = detect_device(args.device, args.precision)
    config = ModelConfig(
        d_model=args.d_model,
        n_layers=args.layers,
        state_interactions=args.state_interactions,
    )
    model = GlassMindLM(config).to(capabilities.torch_device).eval()
    baseline_records = load_baseline(args.baseline)
    compile_active = False
    if args.compile:
        if not capabilities.compile_available:
            raise RuntimeError("torch.compile ist in diesem PyTorch-Build nicht verfügbar")
        compiled = torch.compile(model)
        try:
            # Erst der Probelauf zeigt, ob der Backend-Compiler wirklich baut.
            with torch.inference_mode(), autocast_context(capabilities):
                compiled(torch.zeros(1, 4, dtype=torch.long, device=capabilities.torch_device))
        except Exception as exc:  # pragma: no cover - umgebungsabhängig
            if args.require_compile:
                raise RuntimeError(
                    "torch.compile konnte in dieser Umgebung nicht übersetzen. Der Inductor-"
                    "Backend benötigt CPython-Header und einen passenden C-Compiler. "
                    f"Ursprüngliche Meldung: {exc}"
                ) from exc
            print(
                "[Hinweis] torch.compile ist in dieser Umgebung nicht übersetzbar "
                f"({type(exc).__name__}); die Messung läuft im normalen Eager-Pfad weiter. "
                "GlassMind benötigt torch.compile nicht."
            )
        else:
            model = compiled
            compile_active = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    print(f"[Benchmark] Backend={capabilities.backend}  Precision={capabilities.precision}  Parameter={model.parameter_count:,}")
    for length in args.lengths:
        inputs = torch.randint(0, config.vocab_size, (args.batch_size, length), device=capabilities.torch_device)
        baseline = None
        for mode in (ObservationMode.OFF, ObservationMode.SUMMARY, ObservationMode.TRACE):
            result = measure(model, inputs, capabilities, mode, args.iterations)
            if baseline is None:
                baseline = float(result["tokens_per_second"])
            result["overhead_percent"] = 100 * (1 - float(result["tokens_per_second"]) / baseline)
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "architecture": "glassmind_selective_recurrent_v1",
                "model_config": config.to_dict(),
                "parameter_count": model.parameter_count,
                "hardware": capabilities.to_dict(),
                "dtype": capabilities.precision,
                "device": capabilities.name,
                "backend": capabilities.backend,
                "batch_size": args.batch_size,
                "sequence_length": length,
                "operation": "inference",
                "compile": compile_active,
                **result,
            }
            baseline_record = baseline_records.get((length, mode.name.lower()))
            if baseline_record is not None:
                baseline_tps = float(baseline_record["tokens_per_second"])
                record["baseline_tokens_per_second"] = baseline_tps
                record["throughput_change_vs_baseline_percent"] = 100 * (
                    float(result["tokens_per_second"]) / baseline_tps - 1
                )
                record["baseline_parameter_count"] = int(baseline_record["parameter_count"])
            records.append(record)
            comparison = (
                f"  vs. Basis={float(record['throughput_change_vs_baseline_percent']):+6.1f}%"
                if "throughput_change_vs_baseline_percent" in record
                else ""
            )
            print(
                f"[Länge {length:4d}] {mode.name.lower():7s}  "
                f"{float(result['tokens_per_second']):10,.0f} tok/s  "
                f"Overhead={float(result['overhead_percent']):6.1f}%{comparison}  "
                f"Peak={result['peak_memory_bytes']}"
            )
    latency = streaming_latency(
        model,
        torch.zeros(args.batch_size, dtype=torch.long, device=capabilities.torch_device),
        capabilities,
    )
    for record in records:
        record.update(latency)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(
        f"[Ergebnis] {args.output}  Streaming-Latenz(Median)="
        f"{latency['streaming_latency_ms_per_token']:.3f} ms/Token  "
        f"(min {latency['streaming_latency_ms_per_token_min']:.3f}, "
        f"max {latency['streaming_latency_ms_per_token_max']:.3f})")


if __name__ == "__main__":
    main()
