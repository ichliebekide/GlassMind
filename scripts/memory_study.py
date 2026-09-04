#!/usr/bin/env python3
"""Milestone 3: prüft, ob bounded sparse external memory seinen Platz verdient.

Das Skript trainiert dieselbe Architektur einmal ohne und einmal mit Speicher
und stellt beide gegenüber. Es setzt nicht voraus, dass der Speicher hilft –
wenn er nicht hilft, steht das im Ergebnis.

Abschnitte:

``compare``     ohne Memory gegen mit Memory über wachsende Distanzen
``ablation``    --disable-memory / --disable-memory-read / --disable-memory-write
``query``       welche Zustandsquelle die beste Query erzeugt
``capacity``    wie viele Slots und welches Top-K nötig sind
``policy``      welche Replacement-Policy gewinnt
``slots``       Slot-Ablation und tatsächliche Belegung
``cost``        Durchsatz, Latenz, Speicher und Beobachtungs-Overhead
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import torch

from glassmind.analysis.evaluation import masked_lm_metrics
from glassmind.data.state_tasks import (
    MEMORY_TASK_GENERATORS,
    StateTaskVocabulary,
    memory_task_vocabulary,
)
from glassmind.model import GlassMindLM, ModelConfig
from glassmind.model.memory import memory_utilisation
from glassmind.observe import ObservationBus, ObservationMode
from glassmind.training import StateIntelligenceTrainingConfig, train_state_intelligence
from glassmind.utils.device import (
    autocast_context,
    detect_device,
    peak_memory_bytes,
    reset_peak_memory,
    synchronize,
)
from glassmind.utils.reproducibility import environment_metadata, seed_everything

TASKS = ("delayed_binding", "multiple_bindings", "distractor_recall",
         "memory_replacement", "repeated_retrieval")
#: Die Bewertung nutzt immer dieselben Beispiele je Zelle, damit Varianten
#: vergleichbar bleiben.
EVAL_REPEATS = 3


def build(config: ModelConfig, capabilities: Any) -> GlassMindLM:
    return GlassMindLM(config).to(capabilities.torch_device)


def train(
    config: ModelConfig,
    training: StateIntelligenceTrainingConfig,
    capabilities: Any,
    vocabulary: StateTaskVocabulary,
    *,
    seed: int,
    logger: Any = None,
) -> tuple[GlassMindLM, dict[str, Any]]:
    seed_everything(seed)
    model = build(config, capabilities)
    metrics, _ = train_state_intelligence(
        model, training, capabilities, vocabulary=vocabulary, logger=logger
    )
    return model, metrics


@torch.inference_mode()
def evaluate(
    model: GlassMindLM,
    capabilities: Any,
    vocabulary: StateTaskVocabulary,
    *,
    distances: Sequence[int],
    batch_size: int = 16,
    tasks: Sequence[str] = TASKS,
    seed: int = 90_000,
    repeats: int = EVAL_REPEATS,
    token_budget: int = 200_000,
    **forward_options: Any,
) -> dict[str, dict[int, float]]:
    """Accuracy je Aufgabe und Distanz.

    Die Zahl der Beispiele richtet sich nach einem festen Tokenbudget je Zelle:
    Bei Distanz 16384 wäre eine feste Batchgröße unbezahlbar, bei Distanz 1024
    wäre sie unnötig klein. Die tatsächlich ausgewertete Beispielzahl steht in
    ``examples_per_cell``.
    """
    model.eval()
    results: dict[str, dict[int, float]] = {}
    for task in tasks:
        generator = MEMORY_TASK_GENERATORS[task]
        per_distance: dict[int, float] = {}
        for distance in distances:
            # Mindestens vier Beispiele, höchstens die gewünschte Batchgröße.
            per_batch = max(4, min(batch_size, token_budget // max(distance * repeats, 1)))
            correct = total = 0.0
            for repeat in range(repeats):
                batch = generator(
                    batch_size=per_batch,
                    distance=distance,
                    seed=seed + distance * 17 + repeat * 7,
                    vocabulary=vocabulary,
                ).to(capabilities.torch_device)
                with autocast_context(capabilities):
                    logits, _ = model(batch.input_ids, **forward_options)
                metrics = masked_lm_metrics(logits, batch)
                correct += metrics["accuracy"] * metrics["answer_tokens"]
                total += metrics["answer_tokens"]
            per_distance[distance] = correct / max(total, 1)
        results[task] = per_distance
    return results


def examples_per_cell(distance: int, batch_size: int, repeats: int, token_budget: int) -> int:
    return max(4, min(batch_size, token_budget // max(distance * repeats, 1))) * repeats


def mean_accuracy(results: dict[str, dict[int, float]], distance: int | None = None) -> float:
    values = [
        accuracy
        for per_distance in results.values()
        for key, accuracy in per_distance.items()
        if distance is None or key == distance
    ]
    return sum(values) / max(len(values), 1)


@torch.inference_mode()
def logit_difference(
    model: GlassMindLM,
    capabilities: Any,
    vocabulary: StateTaskVocabulary,
    *,
    distance: int,
    batch_size: int = 16,
    seed: int = 91_000,
    **forward_options: Any,
) -> dict[str, float]:
    """Vergleicht Logits mit und ohne die übergebene Abschaltung."""
    batch = MEMORY_TASK_GENERATORS["multiple_bindings"](
        batch_size=batch_size, distance=distance, seed=seed, vocabulary=vocabulary
    ).to(capabilities.torch_device)
    model.eval()
    with autocast_context(capabilities):
        reference, _ = model(batch.input_ids)
        altered, _ = model(batch.input_ids, **forward_options)
    a = reference[batch.loss_mask].float()
    b = altered[batch.loss_mask].float()
    return {
        "logit_rms_difference": float((b - a).square().mean().sqrt()),
        "logit_max_difference": float((b - a).abs().max()),
        "prediction_change_rate": float((a.argmax(-1) != b.argmax(-1)).float().mean()),
        "accuracy_reference": float(
            (a.argmax(-1) == batch.targets[batch.loss_mask]).float().mean()
        ),
        "accuracy_altered": float(
            (b.argmax(-1) == batch.targets[batch.loss_mask]).float().mean()
        ),
    }


def measure_cost(model: GlassMindLM, capabilities: Any, *, length: int = 128) -> dict[str, Any]:
    """Durchsatz, Streaming-Latenz, Speicher – jeweils Median mehrerer Reihen."""
    device = capabilities.torch_device
    tokens = torch.randint(0, model.config.vocab_size, (1, length), device=device)
    model.eval()
    if capabilities.backend in {"cuda", "rocm"}:
        torch.cuda.empty_cache()
    reset_peak_memory(capabilities)
    result: dict[str, Any] = {}
    with torch.inference_mode(), autocast_context(capabilities):
        for _ in range(3):
            model(tokens)
        synchronize(capabilities)
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(5):
                model(tokens)
            synchronize(capabilities)
            samples.append(tokens.numel() * 5 / (time.perf_counter() - started))
        result["tokens_per_second"] = statistics.median(samples)

        token = torch.zeros(1, dtype=torch.long, device=device)
        state = None
        for _ in range(16):
            _, state = model.step(token, state)
        synchronize(capabilities)
        samples = []
        for _ in range(5):
            started = time.perf_counter()
            for _ in range(128):
                _, state = model.step(token, state)
            synchronize(capabilities)
            samples.append((time.perf_counter() - started) * 1000 / 128)
        result["streaming_ms_per_token"] = statistics.median(samples)
    result["peak_memory_bytes"] = peak_memory_bytes(capabilities)

    # Beobachtungs-Overhead getrennt ausweisen.
    with torch.inference_mode(), autocast_context(capabilities):
        bus = ObservationBus(ObservationMode.TRACE)
        model(tokens, observer=bus)
        synchronize(capabilities)
        started = time.perf_counter()
        for _ in range(3):
            model(tokens, observer=ObservationBus(ObservationMode.TRACE))
        synchronize(capabilities)
        traced = tokens.numel() * 3 / (time.perf_counter() - started)
    result["trace_tokens_per_second"] = traced
    result["observation_overhead_percent"] = 100 * (1 - traced / result["tokens_per_second"])
    result["parameter_count"] = model.parameter_count
    result["memory_parameter_count"] = (
        0 if model.memory is None else sum(p.numel() for p in model.memory.parameters())
    )
    return result


def format_matrix(title: str, rows: list[tuple[str, dict[int, float]]], distances: Sequence[int]) -> str:
    header = f"{title:34s}" + "".join(f"{'D' + str(d):>10s}" for d in distances)
    lines = [header, "-" * len(header)]
    for label, values in rows:
        lines.append(f"{label:34s}" + "".join(f"{values.get(d, float('nan')):9.1%} " for d in distances))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Abschnitte
# ----------------------------------------------------------------------

def section_compare(args, capabilities, vocabulary, base_config, training) -> dict[str, Any]:
    """Der Kernvergleich: dieselbe Architektur ohne und mit Speicher."""
    print("=== Ohne Memory gegen mit Memory ===")
    variants = [
        ("ohne Memory", {}),
        (
            f"Memory {args.slots} Slots",
            dict(
                memory_slots=args.slots,
                memory_width=args.memory_width,
                memory_key_dim=args.memory_key_dim,
                memory_read_k=args.read_k,
                memory_write_k=args.write_k,
                memory_query_source=args.query_source,
                memory_replacement=args.replacement,
                # Für die Nutzungsanalyse werden die Zähler gebraucht.
                memory_track_usage=True,
            ),
        ),
    ]
    records = []
    for label, overrides in variants:
        config = ModelConfig(**{**base_config, **overrides})
        started = time.perf_counter()
        model, metrics = train(config, training, capabilities, vocabulary, seed=args.seed)
        results = evaluate(
            model, capabilities, vocabulary,
            distances=args.distances, batch_size=args.eval_batch_size,
        )
        cost = measure_cost(model, capabilities)
        utilisation = None
        if model.memory is not None:
            with torch.inference_mode(), autocast_context(capabilities):
                batch = MEMORY_TASK_GENERATORS["memory_replacement"](
                    batch_size=1, distance=args.distances[-1], seed=95_000, vocabulary=vocabulary
                ).to(capabilities.torch_device)
                _, state = model(batch.input_ids)
            utilisation = memory_utilisation(state.memory)
        records.append({
            "label": label,
            "config": config.to_dict(),
            "training": metrics,
            "training_seconds": time.perf_counter() - started,
            "accuracy": {task: {str(k): v for k, v in per.items()} for task, per in results.items()},
            "mean_accuracy": mean_accuracy(results),
            "cost": cost,
            "memory_utilisation": utilisation,
        })
        print(f"  {label:24s} Ø-Accuracy={mean_accuracy(results):.1%}  "
              f"{cost['tokens_per_second']:,.0f} Tok/s  {cost['streaming_ms_per_token']:.3f} ms/Tok  "
              f"Training {records[-1]['training_seconds']:.0f}s")
        del model
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
    print()
    for task in TASKS:
        rows = [(record["label"], {int(k): v for k, v in record["accuracy"][task].items()})
                for record in records]
        print(format_matrix(task, rows, args.distances))
        print()
    return {"variants": records}


def section_ablation(args, capabilities, vocabulary, base_config, training) -> dict[str, Any]:
    """Memory gilt nur als kausal relevant, wenn die Ablation etwas bewirkt."""
    print("=== Memory-Ablation ===")
    config = ModelConfig(**{**base_config, **dict(
        memory_slots=args.slots, memory_width=args.memory_width,
        memory_key_dim=args.memory_key_dim, memory_read_k=args.read_k,
        memory_write_k=args.write_k, memory_query_source=args.query_source,
        memory_replacement=args.replacement, memory_track_usage=True,
    )})
    model, _ = train(config, training, capabilities, vocabulary, seed=args.seed)
    modes = [
        ("vollständig aktiv", {}),
        ("--disable-memory", dict(disable_memory=True)),
        ("--disable-memory-read", dict(disable_memory_read=True)),
        ("--disable-memory-write", dict(disable_memory_write=True)),
    ]
    records = []
    for label, options in modes:
        results = evaluate(
            model, capabilities, vocabulary,
            distances=args.distances, batch_size=args.eval_batch_size, **options
        )
        difference = (
            logit_difference(model, capabilities, vocabulary,
                             distance=args.distances[len(args.distances) // 2], **options)
            if options else None
        )
        records.append({
            "label": label,
            "options": {k: bool(v) for k, v in options.items()},
            "accuracy": {task: {str(k): v for k, v in per.items()} for task, per in results.items()},
            "mean_accuracy": mean_accuracy(results),
            "logit_difference": difference,
        })
        extra = ""
        if difference:
            extra = (f"  ΔLogit-RMS={difference['logit_rms_difference']:.4f}  "
                     f"geänderte Vorhersagen={difference['prediction_change_rate']:.1%}")
        print(f"  {label:24s} Ø-Accuracy={mean_accuracy(results):.1%}{extra}")

    # Slot-Ablation: einzelne Zellen abschalten.
    slot_records = []
    with torch.inference_mode(), autocast_context(capabilities):
        batch = MEMORY_TASK_GENERATORS["multiple_bindings"](
            batch_size=1, distance=args.distances[len(args.distances) // 2],
            seed=95_500, vocabulary=vocabulary,
        ).to(capabilities.torch_device)
        _, state = model(batch.input_ids)
    utilisation = memory_utilisation(state.memory)
    reads = utilisation["read_distribution"]
    ranked = sorted(range(len(reads)), key=lambda index: reads[index], reverse=True)
    for slot in ranked[: args.slot_ablations]:
        difference = logit_difference(
            model, capabilities, vocabulary,
            distance=args.distances[len(args.distances) // 2],
            ablate_memory_slots=[slot],
        )
        slot_records.append({"slot": slot, "reads": reads[slot], **difference})
    if slot_records:
        print("\n  Slot-Ablation (die meistgelesenen Zellen):")
        for record in slot_records:
            print(f"    Slot {record['slot']:3d}  Reads={record['reads']:6.0f}  "
                  f"ΔLogit-RMS={record['logit_rms_difference']:.4f}  "
                  f"geänderte Vorhersagen={record['prediction_change_rate']:.1%}")
    print()
    del model
    if capabilities.backend in {"cuda", "rocm"}:
        torch.cuda.empty_cache()
    return {"modes": records, "slots": slot_records, "utilisation": utilisation}


def _variant_sweep(
    args, capabilities, vocabulary, base_config, training, title: str,
    variants: list[tuple[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Gemeinsamer Ablauf für Query-, Kapazitäts- und Policy-Studien."""
    print(f"=== {title} ===")
    records = []
    for label, overrides in variants:
        merged = dict(
            memory_slots=args.slots, memory_width=args.memory_width,
            memory_key_dim=args.memory_key_dim, memory_read_k=args.read_k,
            memory_write_k=args.write_k, memory_query_source=args.query_source,
            memory_replacement=args.replacement, memory_track_usage=True,
        )
        merged.update(overrides)
        try:
            config = ModelConfig(**{**base_config, **merged})
        except ValueError as exc:
            print(f"  {label:34s} nicht zulässig: {exc}")
            continue
        model, _ = train(config, training, capabilities, vocabulary, seed=args.seed)
        results = evaluate(
            model, capabilities, vocabulary,
            distances=args.distances, batch_size=args.eval_batch_size,
        )
        cost = measure_cost(model, capabilities)
        utilisation = None
        if model.memory is not None:
            with torch.inference_mode(), autocast_context(capabilities):
                batch = MEMORY_TASK_GENERATORS["memory_replacement"](
                    batch_size=1, distance=args.distances[-1], seed=96_000, vocabulary=vocabulary
                ).to(capabilities.torch_device)
                _, state = model(batch.input_ids)
            utilisation = memory_utilisation(state.memory)
        records.append({
            "label": label,
            "overrides": overrides,
            "mean_accuracy": mean_accuracy(results),
            "accuracy": {task: {str(k): v for k, v in per.items()} for task, per in results.items()},
            "tokens_per_second": cost["tokens_per_second"],
            "streaming_ms_per_token": cost["streaming_ms_per_token"],
            "memory_utilisation": utilisation,
        })
        occupancy = f"  belegt={utilisation['occupied_slots']}/{utilisation['slots']}" if utilisation else ""
        print(f"  {label:34s} Ø-Accuracy={mean_accuracy(results):.1%}  "
              f"{cost['tokens_per_second']:,.0f} Tok/s{occupancy}")
        del model
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
    print()
    return {"variants": records}


def section_query(args, capabilities, vocabulary, base_config, training) -> dict[str, Any]:
    return _variant_sweep(
        args, capabilities, vocabulary, base_config, training,
        "Welche Zustandsquelle erzeugt die beste Query?",
        [(f"query={source}", {"memory_query_source": source})
         for source in ("output", "fast", "context", "semantic", "fast_context", "context_semantic")],
    )


def section_capacity(args, capabilities, vocabulary, base_config, training) -> dict[str, Any]:
    variants = [(f"{slots} Slots", {"memory_slots": slots}) for slots in args.slot_sweep]
    variants += [(f"read_k={k}", {"memory_read_k": k}) for k in (1, 2, 4)]
    variants += [(f"write_k={k}", {"memory_write_k": k}) for k in (1, 2)]
    return _variant_sweep(
        args, capabilities, vocabulary, base_config, training,
        "Wie viele Slots und welches Top-K werden gebraucht?", variants,
    )


def section_policy(args, capabilities, vocabulary, base_config, training) -> dict[str, Any]:
    variants = [(f"replacement={policy}", {"memory_replacement": policy})
                for policy in ("age", "strength", "usage", "lru_strength", "learned")]
    variants += [(f"routing={mode}", {"memory_routing": mode})
                 for mode in ("cosine", "cosine_strength")]
    return _variant_sweep(
        args, capabilities, vocabulary, base_config, training,
        "Welche Replacement-Policy und welches Routing gewinnen?", variants,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 3: Nutzen des Sparse Memory messen")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--slots", type=int, default=64)
    parser.add_argument("--memory-width", type=int, default=64)
    parser.add_argument("--memory-key-dim", type=int, default=32)
    parser.add_argument("--read-k", type=int, default=2)
    parser.add_argument("--write-k", type=int, default=1)
    parser.add_argument("--query-source", default="output")
    parser.add_argument("--replacement", default="lru_strength")
    parser.add_argument("--distances", type=int, nargs="+", default=[1024, 4096, 8192, 16384])
    parser.add_argument("--train-distances", type=int, nargs="+", default=[0, 64, 256, 1024])
    parser.add_argument("--slot-sweep", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--slot-ablations", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmarks/milestone3-memory.json"))
    parser.add_argument("--merge", action="store_true")
    parser.add_argument(
        "--sections", nargs="+", default=["compare", "ablation", "query", "capacity", "policy"],
        choices=["compare", "ablation", "query", "capacity", "policy"],
    )
    args = parser.parse_args()

    seed_everything(args.seed)
    capabilities = detect_device(args.device, "auto")
    vocabulary = memory_task_vocabulary()
    base_config = dict(
        vocab_size=vocabulary.vocab_size,
        d_model=args.d_model,
        n_layers=args.layers,
        telemetry_clusters=4,
        state_interactions=True,
    )
    training = StateIntelligenceTrainingConfig(
        steps=args.steps,
        seed=args.seed,
        tasks=TASKS,
        train_distances=tuple(args.train_distances),
        batch_size=48,
        learning_rate=3e-3,
        log_every=max(args.steps // 4, 1),
    )
    print(f"[Memory-Studie] Backend={capabilities.backend}  Gerät={capabilities.name}")
    print(f"[Memory-Studie] Vokabular={vocabulary.vocab_size} Tokens, {vocabulary.key_count} Schlüssel")
    print(f"[Memory-Studie] Training: {args.steps} Schritte auf {', '.join(TASKS)}")
    print(f"[Memory-Studie] Trainingsdistanzen {args.train_distances}, "
          f"Auswertung bei {args.distances}\n")

    payload: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "milestone": "3",
        "environment": environment_metadata(capabilities, seed=args.seed),
        "tokenizer": vocabulary.to_dict(),
        "settings": {k: (list(v) if isinstance(v, list) else v)
                     for k, v in vars(args).items() if k != "output"},
        "sections": {},
    }
    runners = {
        "compare": section_compare, "ablation": section_ablation,
        "query": section_query, "capacity": section_capacity, "policy": section_policy,
    }
    for name in args.sections:
        payload["sections"][name] = runners[name](
            args, capabilities, vocabulary, base_config, training
        )

    if args.merge and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        for key, value in previous.get("sections", {}).items():
            payload["sections"].setdefault(key, value)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[Ergebnis] {args.output}")


if __name__ == "__main__":
    main()
