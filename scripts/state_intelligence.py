#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from glassmind.analysis import ClusterAnalyzer, StateMetricsAnalyzer, ablation_comparison
from glassmind.data import (
    StateTaskVocabulary,
    generate_associative_recall_batch,
    generate_selective_copy_batch,
)
from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import JSONLRecorder, ObservationBus, ObservationMode
from glassmind.training import (
    StateIntelligenceTrainingConfig,
    evaluate_state_task,
    save_checkpoint,
    train_state_intelligence,
)
from glassmind.training.run import RunDirectory
from glassmind.utils.device import autocast_context, detect_device, peak_memory_bytes
from glassmind.utils.reproducibility import environment_metadata, seed_everything


def _table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Aufgabe | Distanz | Loss | Accuracy |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['distance']} | {float(row['loss']):.4f} | {float(row['accuracy']):.1%} |"
        )
    return lines


def _summary_text(
    *,
    model: GlassMindLM,
    training: dict[str, Any],
    evaluations: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    state_metrics: dict[str, dict[str, Any]],
    trace_path: Path,
    benchmark_passed: bool,
    peak_memory: int | None,
) -> str:
    lines = [
        f"# Milestone 2 State Intelligence: {'PASS' if benchmark_passed else 'FAIL'}",
        "",
        f"- Architektur: `glassmind_selective_recurrent_v1`",
        f"- Parameter: {model.parameter_count:,}",
        f"- Trainingsschritte: {training['steps']}",
        f"- Bester Trainings-Loss: {float(training['best_loss']):.4f}",
        f"- Trainingsdurchsatz: {float(training['tokens_per_second']):,.0f} Token/s",
        f"- Peak-Gerätespeicher: {peak_memory if peak_memory is not None else 'nicht verfügbar'} Byte",
        f"- Replay-Trace: `{trace_path}`",
        "- Externes Memory: nicht vorhanden",
        "- Mixture of Experts: nicht vorhanden",
        "",
        "## Distanz-Benchmarks",
        "",
        *_table(evaluations),
        "",
        "## State-Ablation",
        "",
        "| Aufgabe | Zustand | Δ Loss | Δ Accuracy | Logit-RMS | Prediction geändert |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for result in ablations:
        lines.append(
            f"| {result['task']} | {','.join(result['ablated_states'])} | "
            f"{float(result['loss_change']):+.4f} | {float(result['accuracy_change']):+.1%} | "
            f"{float(result['logit_rms_difference']):.4f} | {float(result['prediction_change_rate']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Gemessene zeitliche State-Eigenschaften",
            "",
            "| State | Aktivierung | Delta | Update-Gate | Retention | Zeitkonstante | Persistenz | Reaktivierungen | Fluss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for state_name in ("fast", "context", "semantic"):
        values = state_metrics.get(state_name, {})
        if not values:
            continue
        lines.append(
            f"| {state_name} | {values['mean_activation_strength']:.4f} | {values['mean_delta']:.4f} | "
            f"{values['mean_update_gate']:.3f} | {values['mean_retention_activity']:.3f} | "
            f"{values['mean_estimated_time_constant']:.2f} | {values['mean_persistence']:.2f} | "
            f"{values['reactivations']} | {values['mean_information_flow']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Einordnung",
            "",
            "Zeitkonstanten sind Näherungen aus der relativen realen Zustandsänderung, keine direkten kausalen Parameter. "
            "Reaktivierungen bezeichnen erneute reale "
            "State-Updates nach mindestens einem inaktiven Token. Beides ist messbare Dynamik, aber keine semantische "
            "Interpretation eines Clusters.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiert und misst Milestone 2: State Intelligence")
    parser.add_argument("--config", type=Path, default=Path("configs/state_intelligence.json"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--minimum-accuracy", type=float, default=0.70)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    training_values = dict(raw["training"])
    training_values["train_distances"] = tuple(training_values["train_distances"])
    if args.steps is not None:
        training_values["steps"] = args.steps
    training_config = StateIntelligenceTrainingConfig(**training_values)
    evaluation_config = raw["evaluation"]
    distances = tuple(int(value) for value in evaluation_config["distances"])
    model_config = ModelConfig.from_dict(raw["model"])
    vocabulary = StateTaskVocabulary()
    if model_config.vocab_size != vocabulary.vocab_size:
        raise ValueError("Modell- und State-Task-Vokabulargröße stimmen nicht überein")
    seed_everything(training_config.seed)
    capabilities = detect_device(args.device, args.precision)
    model = GlassMindLM(model_config).to(capabilities.torch_device)
    run = RunDirectory(args.run_root, prefix="state-intelligence")
    run.write_json(
        "config.json",
        {
            "model": model_config.to_dict(),
            "training": training_config.to_dict(),
            "evaluation": evaluation_config,
            "tokenizer": vocabulary.to_dict(),
        },
    )
    run.write_json("environment.json", environment_metadata(capabilities, seed=training_config.seed))
    run.log(
        f"[Start] State Intelligence  Lauf={run.path}  Backend={capabilities.backend}  "
        f"Precision={capabilities.precision}  Parameter={model.parameter_count:,}"
    )

    training_metrics, optimizer = train_state_intelligence(
        model,
        training_config,
        capabilities,
        vocabulary=vocabulary,
        logger=run,
    )
    evaluation_kwargs = {
        "distances": distances,
        "batch_size": int(evaluation_config["batch_size"]),
        "repeats": int(evaluation_config["repeats"]),
        "seed": training_config.seed + 10_000,
        "capabilities": capabilities,
    }
    associative = evaluate_state_task(
        model,
        generate_associative_recall_batch,
        task_kwargs={"associations": training_config.associations, "vocabulary": vocabulary},
        **evaluation_kwargs,
    )
    selective = evaluate_state_task(
        model,
        generate_selective_copy_batch,
        task_kwargs={"items": training_config.copy_items, "vocabulary": vocabulary},
        **evaluation_kwargs,
    )
    evaluations = associative + selective
    for result in evaluations:
        run.metric({"event": "distance_evaluation", **result})
        run.log(
            f"[evaluation] task={result['task']}  distanz={result['distance']}  "
            f"loss={float(result['loss']):.4f}  accuracy={float(result['accuracy']):.1%}"
        )

    ablations: list[dict[str, Any]] = []
    for task_name, batch in (
        (
            "associative_recall",
            generate_associative_recall_batch(
                batch_size=64,
                distance=64,
                associations=training_config.associations,
                seed=training_config.seed + 20_000,
                vocabulary=vocabulary,
            ),
        ),
        (
            "selective_copy",
            generate_selective_copy_batch(
                batch_size=64,
                distance=64,
                items=training_config.copy_items,
                seed=training_config.seed + 30_000,
                vocabulary=vocabulary,
            ),
        ),
    ):
        device_batch = batch.to(capabilities.torch_device)
        for state_name in ("fast", "context", "semantic"):
            with autocast_context(capabilities):
                comparison = ablation_comparison(model, device_batch, (state_name,))
            comparison["task"] = task_name
            comparison["distance"] = batch.distance
            ablations.append(comparison)
            run.metric({"event": "state_ablation", **comparison})
            run.log(
                f"[ablation] task={task_name}  state={state_name}  "
                f"delta_loss={comparison['loss_change']:+.4f}  "
                f"delta_acc={comparison['accuracy_change']:+.1%}  "
                f"prediction_change={comparison['prediction_change_rate']:.1%}"
            )

    trace_batch = generate_selective_copy_batch(
        batch_size=1,
        distance=64,
        items=training_config.copy_items,
        seed=training_config.seed + 40_000,
        vocabulary=vocabulary,
    ).to(capabilities.torch_device)
    trace_path = run.traces / "state-intelligence.jsonl"
    bus = ObservationBus(ObservationMode.TRACE)
    recorder = JSONLRecorder(trace_path)
    cluster_analyzer = ClusterAnalyzer()
    state_analyzer = StateMetricsAnalyzer()
    bus.subscribe(recorder)
    bus.subscribe(cluster_analyzer)
    bus.subscribe(state_analyzer)
    model.eval()
    with torch.inference_mode(), autocast_context(capabilities):
        model(trace_batch.input_ids, observer=bus)
    bus.close()
    cluster_metrics = cluster_analyzer.summaries(vocabulary.token_name)
    state_metrics = state_analyzer.summaries()
    run.write_json("cluster_metrics.json", cluster_metrics)
    run.write_json("state_metrics.json", state_metrics)
    run.write_json(
        "results.json",
        {
            "training": training_metrics,
            "distance_evaluations": evaluations,
            "ablations": ablations,
            "state_metrics": state_metrics,
            "trace": str(trace_path),
        },
    )
    checkpoint_path = run.checkpoints / "final.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        tokenizer=vocabulary,
        optimizer=optimizer,
        step=training_config.steps,
        extra={"milestone": 2, "tasks": ["associative_recall", "selective_copy"]},
    )
    benchmark_passed = all(float(result["accuracy"]) >= args.minimum_accuracy for result in evaluations)
    summary = _summary_text(
        model=model,
        training=training_metrics,
        evaluations=evaluations,
        ablations=ablations,
        state_metrics=state_metrics,
        trace_path=trace_path,
        benchmark_passed=benchmark_passed,
        peak_memory=peak_memory_bytes(capabilities),
    )
    (run.path / "summary.md").write_text(summary, encoding="utf-8")
    run.log(f"[{'PASS' if benchmark_passed else 'FAIL'}] Ergebnisse={run.path / 'results.json'}")
    if not benchmark_passed:
        raise SystemExit(
            f"Mindestens eine Distanzmessung liegt unter {args.minimum_accuracy:.0%}; siehe {run.path / 'summary.md'}"
        )


if __name__ == "__main__":
    main()
