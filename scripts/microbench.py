#!/usr/bin/env python3
"""Misst GlassMind-typische Operationen und leitet daraus eine Precision-Empfehlung ab."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from glassmind.precision.microbench import auto_policy, run_microbenchmark
from glassmind.utils.device import detect_device
from glassmind.utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Hardware-Microbenchmark für GlassMind")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--repetitions", type=int, default=200)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    seed_everything(23)
    # Der Microbenchmark misst rohe Operationen; AMP würde nur zusätzliche
    # Casts einfügen und die Formate gegeneinander verzerren.
    capabilities = detect_device(args.device, "float32")
    report = run_microbenchmark(
        capabilities,
        d_model=args.d_model,
        batch=args.batch_size,
        sequence=args.sequence,
        repetitions=args.repetitions,
    )
    policy, _ = auto_policy(capabilities, report)
    payload = report.to_dict()
    payload["auto_policy"] = policy.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(report.format_table())
        print()
        print(f"  auto-Profil: {policy.describe()}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"\n[Ergebnis] {args.output}")


if __name__ == "__main__":
    main()
