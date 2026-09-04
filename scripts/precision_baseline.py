#!/usr/bin/env python3
"""Friert die Milestone-2.5-Referenz ein.

Alle Milestone-2.6-Vergleiche beziehen sich auf genau diese Datei. Sie wird
einmal erzeugt und danach nicht mehr verändert.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from glassmind.data.state_tasks import StateTaskVocabulary
from glassmind.precision.reference import collect_reference
from glassmind.training.checkpoint import load_checkpoint
from glassmind.utils.device import detect_device
from glassmind.utils.reproducibility import environment_metadata, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Erzeugt die eingefrorene Milestone-2.5-Referenz")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/milestone2_5-reference.json"))
    parser.add_argument("--distances", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--throughput-length", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    seed_everything(2600)
    capabilities = detect_device(args.device, "float32")
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, device=capabilities.torch_device)
    if not isinstance(tokenizer, StateTaskVocabulary):
        raise SystemExit(
            "Die Referenz braucht einen State-Task-Checkpoint; der übergebene nutzt "
            f"{type(tokenizer).__name__}."
        )
    print(f"[Referenz] Checkpoint={args.checkpoint}  Format={metadata['format_version']}")
    print(f"[Referenz] Backend={capabilities.backend}  Parameter={model.parameter_count:,}")

    record = collect_reference(
        model,
        capabilities,
        tokenizer,
        label="milestone2.5-fp32",
        distances=tuple(args.distances),
        throughput_length=args.throughput_length,
        iterations=args.iterations,
    )
    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "milestone": "2.5",
        "source_checkpoint": str(args.checkpoint),
        "checkpoint_format": metadata["format_version"],
        "environment": environment_metadata(capabilities, seed=2600),
        "tokenizer": tokenizer.to_dict(),
        "reference": record,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\n  Durchsatz          {record['tokens_per_second']:,.0f} Token/s "
          f"(Länge {record['sequence_length']})")
    print(f"  Streaming          {record['streaming_ms_per_token']:.3f} ms/Token")
    training = record.get("training_tokens_per_second")
    if isinstance(training, (int, float)):
        print(f"  Training           {training:,.0f} Token/s")
    print(f"  Gewichtsspeicher   {record['parameter_storage_bytes']:,} Byte")
    peak = record.get("peak_memory_bytes")
    print(f"  Peak-Gerätespeicher{'':1s} {peak:,} Byte" if peak else "  Peak-Gerätespeicher nicht verfügbar")
    print("  Aufgaben:")
    for entry in record["tasks"]:
        print(f"    {entry['task']:20s} Distanz {entry['distance']:5d}  "
              f"Loss={entry['loss']:.4f}  Accuracy={entry['accuracy']:.1%}")
    print("  Ablationen (Δ Accuracy):")
    for entry in record["ablations"]:
        print(f"    {entry['task']:20s} {','.join(entry['ablated_states']):9s} "
              f"{entry['accuracy_change']:+.1%}  Logit-RMS={entry['logit_rms_difference']:.4f}")
    print("  Zustandsrollen:")
    for name, values in record["state_metrics"].items():
        print(f"    {name:9s} Norm={record['final_state_norms'][name]:8.4f}  "
              f"Zeitkonstante={values['mean_estimated_time_constant']:7.2f}  "
              f"Delta={values['mean_delta']:.4f}")
    print(f"\n[Ergebnis] {args.output}")


if __name__ == "__main__":
    main()
