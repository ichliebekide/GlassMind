#!/usr/bin/env python3
"""Milestone 2.6: misst systematisch, welche Precision GlassMind wirklich braucht.

Das Skript trifft keine Vorannahmen darüber, welches Format gewinnt. Es baut
aus demselben trainierten Checkpoint eine Reihe von Varianten, misst jede
gleich und stellt die Ergebnisse gegenüber.

Abschnitte:

``float``        Gleitkomma-Grundvarianten inklusive Autocast
``states``       die vollständige fast/context/semantic-dtype-Matrix
``quantization`` INT8/INT4/FP8 als Weight-Only, gesamt und je Komponente
``drift``        Langzeitdrift gegen die FP32-Referenz
``telemetry``    Abweichung der sichtbaren Aktivität über den Observation Bus
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch

from glassmind.data.state_tasks import StateTaskVocabulary
from glassmind.model import GlassMindLM, ModelConfig
from glassmind.precision.apply import apply_precision
from glassmind.precision.compare import compare_telemetry, format_telemetry_comparison
from glassmind.precision.drift import drift_summary, format_drift_table, measure_drift
from glassmind.precision.microbench import auto_policy
from glassmind.precision.policy import (
    WEIGHT_COMPONENTS,
    PrecisionPolicy,
    balanced_profile,
    experimental_profile,
    fast_profile,
    safe_profile,
)
from glassmind.precision.quantization import QuantizationUnsupported, fp8_compute_supported
from glassmind.precision.reference import collect_reference, task_accuracy
from glassmind.training.checkpoint import load_checkpoint, save_checkpoint
from glassmind.utils.device import autocast_context, detect_device
from glassmind.utils.reproducibility import environment_metadata, seed_everything

STATE_NAMES = ("fast", "context", "semantic")


# ----------------------------------------------------------------------
# Variantenaufbau
# ----------------------------------------------------------------------

class ModelFactory:
    """Baut aus einem einmal geladenen Checkpoint beliebig viele Varianten."""

    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        model, tokenizer, metadata = load_checkpoint(checkpoint, device="cpu")
        self.config: ModelConfig = model.config
        self.tokenizer = tokenizer
        self.metadata = metadata
        self.state = copy.deepcopy(model.state_dict())
        self.device = device

    def build(
        self,
        *,
        policy: PrecisionPolicy | None = None,
        model_dtype: torch.dtype | None = None,
    ) -> GlassMindLM:
        model = GlassMindLM(self.config, policy or PrecisionPolicy())
        model.load_state_dict(self.state)
        if model_dtype is not None:
            model.to(model_dtype)
        model.to(self.device).eval()
        if policy is not None and policy.quantizes_weights:
            apply_precision(model, policy, device=self.device)
        return model


def checkpoint_bytes(model: GlassMindLM, tokenizer: StateTaskVocabulary) -> int:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "variant.pt"
        save_checkpoint(path, model, tokenizer=tokenizer)
        return path.stat().st_size


def float_variants() -> list[tuple[str, PrecisionPolicy, torch.dtype | None, bool]]:
    """(Name, Policy, Modell-dtype, Autocast) – Autocast ist der 2.5-Standardpfad."""
    return [
        ("fp32 (Referenz)", safe_profile(), None, False),
        ("fp32 + AMP-bf16 (Milestone 2.5)", PrecisionPolicy(profile="milestone2.5"), None, True),
        ("Gewichte+Compute bf16", PrecisionPolicy(profile="bf16-modell"), torch.bfloat16, False),
        ("Gewichte+Compute fp16", PrecisionPolicy(profile="fp16-modell"), torch.float16, False),
        ("balanced (compute bf16, ctx+sem fp32)", balanced_profile(), None, False),
        ("fast (compute bf16, sem fp32)", fast_profile(), None, False),
        ("balanced-fp16", balanced_profile("float16"), None, False),
        ("fast-fp16", fast_profile("float16"), None, False),
    ]


def state_matrix_variants() -> Iterable[tuple[str, PrecisionPolicy]]:
    """Alle 27 Kombinationen aus fast/context/semantic × fp32/bf16/fp16."""
    names = ("float32", "bfloat16", "float16")
    short = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}
    for fast, context, semantic in itertools.product(names, repeat=3):
        label = f"fast={short[fast]} ctx={short[context]} sem={short[semantic]}"
        yield label, PrecisionPolicy(
            profile="state-matrix",
            fast_state=fast,
            context_state=context,
            semantic_state=semantic,
        )


def quantization_variants(group_size: int) -> list[tuple[str, PrecisionPolicy]]:
    variants: list[tuple[str, PrecisionPolicy]] = []
    for scheme in ("int8", "int4", "float8_e4m3", "float8_e5m2"):
        variants.append((f"alle Gewichte {scheme}", experimental_profile("float32", scheme, group_size)))
    # Ohne Dequantisierungs-Cache: zeigt den echten Laufzeitspeicher und die
    # Kosten, die der Cache verdeckt.
    for scheme in ("int8", "int4"):
        policy = experimental_profile("float32", scheme, group_size)
        variants.append(
            (
                f"alle Gewichte {scheme}, ohne Dequant-Cache",
                PrecisionPolicy(
                    profile="experimental-nocache",
                    compute=policy.compute,
                    activations=policy.activations,
                    fast_state=policy.fast_state,
                    context_state=policy.context_state,
                    semantic_state=policy.semantic_state,
                    weights=policy.weights,
                    weight_group_size=group_size,
                    dequantization_cache=False,
                ),
            )
        )
    # Gruppenweise INT4: mehr Skalen, dafür genauer. Ob sich das lohnt, zeigt
    # der Vergleich von Speicherbedarf und Drift.
    variants.append(
        (
            "alle Gewichte int4, Gruppe 8",
            experimental_profile("float32", "int4", 8),
        )
    )
    for component in WEIGHT_COMPONENTS:
        for scheme in ("int8", "int4"):
            variants.append(
                (
                    f"nur {component} {scheme}",
                    PrecisionPolicy(
                        profile="komponentenweise",
                        weights={component: scheme},
                        weight_group_size=group_size,
                    ),
                )
            )
    variants.append(
        (
            "gemischt (emb+head int8, proj int4, states bf16/fp32)",
            PrecisionPolicy(
                profile="gemischt",
                fast_state="bfloat16",
                context_state="bfloat16",
                semantic_state="float32",
                weights={
                    "embedding": "int8",
                    "lm_head": "int8",
                    "input_projection": "int4",
                    "state_projection": "int8",
                    "gate": "int8",
                    "local_mixer": "int8",
                    "output_projection": "int8",
                },
                weight_group_size=group_size,
            ),
        )
    )
    return variants


# ----------------------------------------------------------------------
# Messung
# ----------------------------------------------------------------------

def evaluate_variant(
    factory: ModelFactory,
    capabilities: Any,
    label: str,
    policy: PrecisionPolicy,
    *,
    model_dtype: torch.dtype | None = None,
    autocast: bool = False,
    distances: tuple[int, ...],
    full: bool,
    reference: dict[str, Any],
    measure_checkpoint: bool = True,
    task_batch_size: int = 32,
) -> dict[str, Any]:
    model = factory.build(policy=policy, model_dtype=model_dtype)
    context = autocast_context(capabilities) if autocast else torch.autocast(
        capabilities.torch_device.type, enabled=False
    )
    with context:
        record = collect_reference(
            model,
            capabilities,
            factory.tokenizer,
            label=label,
            distances=distances,
            include_training=full,
            include_logits=full,
            task_batch_size=task_batch_size,
        )
    record["autocast"] = autocast
    record["policy"] = policy.to_dict()
    if measure_checkpoint:
        record["checkpoint_bytes"] = checkpoint_bytes(model, factory.tokenizer)
    # Logit-Abweichung gegen die eingefrorene Referenz
    if full and "logits" in record and "logits" in reference:
        a = torch.tensor(reference["logits"]["values"])
        b = torch.tensor(record["logits"]["values"])
        finite = torch.isfinite(b)
        record["logit_finite_fraction"] = float(finite.float().mean())
        if bool(finite.all()):
            record["logit_relative_rms_vs_reference"] = float(
                (b - a).square().mean().sqrt() / a.square().mean().sqrt().clamp_min(1e-12)
            )
        else:
            # Ein nicht-endliches Logit ist kein Driftwert, sondern ein Ausfall.
            record["logit_relative_rms_vs_reference"] = float("inf")
        record["prediction_change_vs_reference"] = float(
            (torch.tensor(reference["logits"]["argmax"]) != torch.tensor(record["logits"]["argmax"]))
            .float()
            .mean()
        )
    del model
    if capabilities.backend in {"cuda", "rocm"}:
        torch.cuda.empty_cache()
    return record


def run_drift(
    factory: ModelFactory,
    capabilities: Any,
    variants: list[tuple[str, PrecisionPolicy, torch.dtype | None]],
    lengths: tuple[int, ...],
) -> list[dict[str, Any]]:
    device = capabilities.torch_device
    maximum = max(lengths)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2606)
    tokens = torch.randint(
        0, factory.config.vocab_size, (1, maximum), generator=generator
    ).to(device)
    baseline = factory.build(policy=safe_profile())
    results: list[dict[str, Any]] = []
    for label, policy, model_dtype in variants:
        candidate = factory.build(policy=policy, model_dtype=model_dtype)
        points = measure_drift(baseline, candidate, tokens, lengths, capabilities=capabilities)
        results.append(
            {
                "label": label,
                "policy": policy.to_dict(),
                "points": [point.to_dict() for point in points],
                "summary": drift_summary(points),
            }
        )
        print(f"\n  [{label}]")
        print("    " + format_drift_table(points).replace("\n", "\n    "))
        del candidate
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
    return results


# ----------------------------------------------------------------------
# Ausgabe
# ----------------------------------------------------------------------

def summary_row(record: dict[str, Any], reference: dict[str, Any], distances: tuple[int, ...]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Konfiguration": record["label"],
        "Token/s": record["tokens_per_second"],
        "ms/Token": record["streaming_ms_per_token"],
        "Training Token/s": record.get("training_tokens_per_second"),
        "VRAM": record.get("peak_memory_bytes"),
        "RAM": record.get("host_peak_rss_bytes"),
        "Gewichte B": record["parameter_storage_bytes"],
        "Checkpoint B": record.get("checkpoint_bytes"),
        "Logit-Drift": record.get("logit_relative_rms_vs_reference"),
    }
    for distance in distances:
        row[f"Recall {distance}"] = task_accuracy(record, "associative_recall", distance)
        row[f"Copy {distance}"] = task_accuracy(record, "selective_copy", distance)
    for name in STATE_NAMES:
        reference_norm = reference["final_state_norms"][name]
        row[f"Norm-Δ {name}"] = abs(
            record["final_state_norms"][name] - reference_norm
        ) / max(abs(reference_norm), 1e-12)
    return row


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    def render(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, str):
            return value
        if isinstance(value, float):
            if abs(value) < 1e-3 and value != 0:
                return f"{value:.2e}"
            return f"{value:,.4f}" if abs(value) < 10 else f"{value:,.0f}"
        return f"{value:,}"

    widths = {column: max(len(column), *(len(render(row.get(column))) for row in rows)) for column in columns}
    header = "  ".join(column.ljust(widths[column]) for column in columns)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(render(row.get(column)).ljust(widths[column]) for column in columns))


def mark_winners(records: list[dict[str, Any]], distances: tuple[int, ...]) -> dict[str, Any]:
    """Bestimmt Gewinner ausschließlich aus den gemessenen Werten.

    Die Bewertung bezieht sich auf genau die übergebenen Distanzen. Bei großen
    Distanzen braucht sie eine ausreichende Zahl an Beispielen, sonst schwankt
    die Aufgabenqualität stärker als der Unterschied zwischen den Varianten –
    ``--eval-batch-size`` steuert das.
    """
    def mean_accuracy(record: dict[str, Any]) -> float:
        values = [
            task_accuracy(record, task, distance)
            for task in ("associative_recall", "selective_copy")
            for distance in distances
        ]
        present = [value for value in values if value is not None]
        return sum(present) / len(present) if present else 0.0

    fastest = max(records, key=lambda item: item["tokens_per_second"])
    smallest = min(records, key=lambda item: item["parameter_storage_bytes"])
    most_stable = min(
        records, key=lambda item: item.get("logit_relative_rms_vs_reference", float("inf"))
    )
    # „Beste Gesamtvariante": höchste Aufgabenqualität, bei Gleichstand der
    # schnellere und kleinere Kandidat. Kein Bonus für ein bestimmtes Format.
    best_quality = max(mean_accuracy(item) for item in records)
    contenders = [item for item in records if mean_accuracy(item) >= best_quality - 1e-9]
    overall = max(contenders, key=lambda item: item["tokens_per_second"])
    return {
        "schnellste": {"label": fastest["label"], "tokens_per_second": fastest["tokens_per_second"]},
        "speichereffizienteste": {
            "label": smallest["label"],
            "parameter_storage_bytes": smallest["parameter_storage_bytes"],
        },
        "stabilste": {
            "label": most_stable["label"],
            "logit_relative_rms_vs_reference": most_stable.get("logit_relative_rms_vs_reference"),
        },
        "beste_gesamt": {
            "label": overall["label"],
            "mean_accuracy": mean_accuracy(overall),
            "tokens_per_second": overall["tokens_per_second"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 2.6: Precision- und Quantisierungsmatrix")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("benchmarks/milestone2_5-reference.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--output", type=Path, default=Path("benchmarks/milestone2_6-precision.json"))
    parser.add_argument("--distances", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument(
        "--drift-lengths", type=int, nargs="+", default=[64, 256, 1024, 4096, 8192]
    )
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=32,
        help="Batchgröße der Aufgabenmessung; kleiner macht große Distanzen bezahlbar",
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        default=["float", "states", "quantization", "drift", "telemetry"],
        choices=["float", "states", "quantization", "drift", "telemetry"],
    )
    parser.add_argument("--quick", action="store_true", help="kürzere Distanzen und Drift-Längen")
    parser.add_argument(
        "--merge",
        action="store_true",
        help="übernimmt nicht gemessene Abschnitte aus einer vorhandenen Ausgabedatei",
    )
    args = parser.parse_args()

    if args.quick:
        args.distances = [16, 64, 256]
        args.drift_lengths = [64, 256, 1024]

    seed_everything(2600)
    capabilities = detect_device(args.device, "auto")
    distances = tuple(args.distances)
    reference_payload = json.loads(args.reference.read_text(encoding="utf-8"))
    reference = reference_payload["reference"]
    factory = ModelFactory(args.checkpoint, capabilities.torch_device)

    print(f"[Matrix] Backend={capabilities.backend}  Gerät={capabilities.name}")
    print(f"[Matrix] Referenz={args.reference}  Parameter={factory.config.to_dict()['d_model']}d "
          f"×{factory.config.n_layers} Blöcke")
    supported, reason = fp8_compute_supported(capabilities.torch_device)
    print(f"[Matrix] FP8-Compute: {'ja' if supported else 'nein'} – {reason}\n")

    payload: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "milestone": "2.6",
        "eval_batch_size": args.eval_batch_size,
        "source_checkpoint": str(args.checkpoint),
        "reference_file": str(args.reference),
        "environment": environment_metadata(capabilities, seed=2600),
        "fp8_compute": {"supported": supported, "reason": reason},
        "sections": {},
    }
    float_records: list[dict[str, Any]] = []

    if "float" in args.sections:
        print("=== Gleitkomma-Grundvarianten ===")
        records = []
        for label, policy, model_dtype, autocast in float_variants():
            try:
                record = evaluate_variant(
                    factory, capabilities, label, policy,
                    model_dtype=model_dtype, autocast=autocast,
                    distances=distances, full=True, reference=reference,
                    task_batch_size=args.eval_batch_size,
                )
            except Exception as exc:  # pragma: no cover - hardwareabhängig
                print(f"  {label}: nicht messbar ({type(exc).__name__}: {exc})")
                continue
            records.append(record)
            print(f"  {label:40s} {record['tokens_per_second']:8,.0f} Tok/s  "
                  f"{record['streaming_ms_per_token']:6.3f} ms/Tok  "
                  f"{record['parameter_storage_bytes']:>9,} B")
        float_records = records
        payload["sections"]["float"] = records
        columns = (
            ["Konfiguration", "Token/s", "ms/Token", "Training Token/s", "VRAM", "Gewichte B",
             "Checkpoint B", "Logit-Drift"]
            + [f"Recall {d}" for d in distances]
            + [f"Copy {d}" for d in distances]
            + [f"Norm-Δ {name}" for name in STATE_NAMES]
        )
        print()
        print_table([summary_row(record, reference, distances) for record in records], columns)
        print()

    if "states" in args.sections:
        print("=== State-dtype-Matrix (Gewichte und Rechenpfad bleiben fp32) ===")
        records = []
        for label, policy in state_matrix_variants():
            record = evaluate_variant(
                factory, capabilities, label, policy,
                distances=distances, full=False, reference=reference,
                measure_checkpoint=False, task_batch_size=args.eval_batch_size,
            )
            records.append(record)
        payload["sections"]["states"] = records
        columns = (
            ["Konfiguration", "Token/s", "ms/Token"]
            + [f"Recall {d}" for d in distances]
            + [f"Copy {d}" for d in distances]
            + [f"Norm-Δ {name}" for name in STATE_NAMES]
        )
        print_table([summary_row(record, reference, distances) for record in records], columns)
        print()

    if "quantization" in args.sections:
        print(f"=== Weight-Only-Quantisierung (Gruppengröße {args.group_size or 'pro Kanal'}) ===")
        records = []
        seen_labels: set[str] = set()
        for label, policy in quantization_variants(args.group_size):
            if label in seen_labels:
                continue
            seen_labels.add(label)
            try:
                record = evaluate_variant(
                    factory, capabilities, label, policy,
                    distances=distances, full=True, reference=reference,
                    task_batch_size=args.eval_batch_size,
                )
            except (QuantizationUnsupported, ValueError) as exc:
                print(f"  {label:52s} nicht messbar: {exc}")
                continue
            records.append(record)
            print(f"  {label:52s} {record['tokens_per_second']:8,.0f} Tok/s  "
                  f"{record['parameter_storage_bytes']:>9,} B  "
                  f"Recall{distances[1]}={task_accuracy(record,'associative_recall',distances[1]):.1%}")
        payload["sections"]["quantization"] = records
        columns = (
            ["Konfiguration", "Token/s", "ms/Token", "VRAM", "Gewichte B", "Checkpoint B", "Logit-Drift"]
            + [f"Recall {d}" for d in distances]
            + [f"Copy {d}" for d in distances]
        )
        print()
        print_table([summary_row(record, reference, distances) for record in records], columns)
        print()
        float_records = float_records + records

    if "drift" in args.sections:
        print(f"=== Langzeitdrift gegen FP32 bei Längen {args.drift_lengths} ===")
        variants: list[tuple[str, PrecisionPolicy, torch.dtype | None]] = [
            ("Gewichte+Compute bf16", PrecisionPolicy(profile="bf16-modell"), torch.bfloat16),
            ("Gewichte+Compute fp16", PrecisionPolicy(profile="fp16-modell"), torch.float16),
            ("balanced (fast bf16)", balanced_profile(), None),
            ("fast (fast+ctx bf16)", fast_profile(), None),
            ("alle States bf16", PrecisionPolicy(profile="states-bf16", fast_state="bfloat16",
                                                 context_state="bfloat16", semantic_state="bfloat16"), None),
            ("nur semantic bf16", PrecisionPolicy(profile="sem-bf16", semantic_state="bfloat16"), None),
            ("nur semantic fp16", PrecisionPolicy(profile="sem-fp16", semantic_state="float16"), None),
            ("alle Gewichte int8", experimental_profile("float32", "int8", args.group_size), None),
            ("alle Gewichte int4", experimental_profile("float32", "int4", args.group_size), None),
        ]
        payload["sections"]["drift"] = run_drift(
            factory, capabilities, variants, tuple(args.drift_lengths)
        )
        print()

    if "telemetry" in args.sections:
        print("=== Abweichung der sichtbaren Aktivität (Observation Bus) ===")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2607)
        tokens = torch.randint(0, factory.config.vocab_size, (1, 96), generator=generator).to(
            capabilities.torch_device
        )
        baseline = factory.build(policy=safe_profile())
        comparisons = []
        for label, policy, model_dtype in (
            ("Gewichte+Compute bf16", PrecisionPolicy(profile="bf16-modell"), torch.bfloat16),
            ("balanced (fast bf16)", balanced_profile(), None),
            ("alle States bf16", PrecisionPolicy(profile="states-bf16", fast_state="bfloat16",
                                                 context_state="bfloat16", semantic_state="bfloat16"), None),
            ("alle Gewichte int8", experimental_profile("float32", "int8", args.group_size), None),
            ("alle Gewichte int4", experimental_profile("float32", "int4", args.group_size), None),
        ):
            candidate = factory.build(policy=policy, model_dtype=model_dtype)
            result = compare_telemetry(baseline, candidate, tokens)
            result["label"] = label
            comparisons.append(result)
            print(f"\n  [{label}]")
            print("    " + format_telemetry_comparison(result).replace("\n", "\n    "))
            del candidate
        payload["sections"]["telemetry"] = comparisons
        print()

    policy, report = auto_policy(capabilities)
    payload["auto_policy"] = policy.to_dict()
    payload["microbenchmark"] = report.to_dict()
    print(f"\n  auto-Profil auf dieser Hardware: {policy.describe()}")
    for note in policy.selection_notes:
        print(f"    - {note}")

    if args.merge and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        for key, value in previous.get("sections", {}).items():
            payload["sections"].setdefault(key, value)
        print(f"  Aus {args.output} übernommen: "
              f"{', '.join(sorted(set(previous.get('sections', {})) - set(args.sections))) or 'nichts'}")

    # Die Bewertung läuft über alle vergleichbaren Abschnitte, auch über die
    # aus einer früheren Messung übernommenen.
    comparable = payload["sections"].get("float", []) + payload["sections"].get("quantization", [])
    if comparable:
        payload["winners"] = mark_winners(comparable, distances)
        print("\n=== Bewertung auf Basis der gemessenen Werte ===")
        for key, value in payload["winners"].items():
            print(f"  {key:24s} {value['label']}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\n[Ergebnis] {args.output}")


if __name__ == "__main__":
    main()
