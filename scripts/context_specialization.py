#!/usr/bin/env python3
"""Milestone 2.5: Untersucht die kausale Rolle von ``context_state``.

Das Skript trainiert ein Modell auf Aufgaben mit mittlerer zeitlicher Struktur
(Abschnittswechsel, abschnittslokale Fakten, kollidierende Schlüssel, Wieder-
aufnahme früherer Themen, hierarchische Sequenzen) und misst anschließend per
State-Ablation, ob ``context_state`` dabei kausal relevant wird.

Das Ergebnis wird so berichtet, wie es gemessen wurde. Es gibt keinen
Architektureingriff, dessen Zweck bessere Ablationszahlen wären.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from glassmind.analysis import ClusterAnalyzer, StateMetricsAnalyzer, ablation_comparison
from glassmind.data import (
    CONTEXT_TASK_GENERATORS,
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

BASE_GENERATORS = {
    "associative_recall": generate_associative_recall_batch,
    "selective_copy": generate_selective_copy_batch,
}


def _generator_for(task: str):
    if task in BASE_GENERATORS:
        return BASE_GENERATORS[task]
    if task in CONTEXT_TASK_GENERATORS:
        return CONTEXT_TASK_GENERATORS[task]
    raise ValueError(f"Unbekannte Aufgabe: {task}")


def _task_kwargs(task: str, config: StateIntelligenceTrainingConfig, vocabulary: StateTaskVocabulary) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"vocabulary": vocabulary}
    if task == "associative_recall":
        kwargs["associations"] = config.associations
    elif task == "selective_copy":
        kwargs["items"] = config.copy_items
    elif task == "sectioned_recall":
        kwargs["sections"] = config.sections
        kwargs["facts_per_section"] = config.facts_per_section
    elif task == "topic_resumption":
        kwargs["sections"] = max(2, config.sections - 1)
    elif task == "hierarchical_scope":
        kwargs["sections"] = config.sections
    return kwargs


def _evaluation_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Aufgabe | Distanz | Loss | Accuracy |", "|---|---:|---:|---:|"]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['distance']} | {float(row['loss']):.4f} | {float(row['accuracy']):.1%} |"
        )
    return lines


def _ablation_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Aufgabe | Zustand | Δ Loss | Δ Accuracy | Logit-RMS | Prediction geändert |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {','.join(row['ablated_states'])} | "
            f"{float(row['loss_change']):+.4f} | {float(row['accuracy_change']):+.1%} | "
            f"{float(row['logit_rms_difference']):.4f} | {float(row['prediction_change_rate']):.1%} |"
        )
    return lines


#: Unterhalb dieser mittleren Accuracy ist eine Ablationsaussage nicht belastbar –
#: an einem Modell, das die Aufgabe gar nicht löst, misst man kein Lösungsverfahren.
RELIABLE_ACCURACY = 0.75


def _verdict(
    ablations: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    context_tasks: set[str],
) -> tuple[str, dict[str, float]]:
    """Formuliert die gemessene Aussage über ``context_state`` ohne Beschönigung."""

    def mean(rows: list[dict[str, Any]], field: str) -> float:
        return sum(float(row[field]) for row in rows) / max(len(rows), 1)

    context_rows = [
        row
        for row in ablations
        if row["ablated_states"] == ["context"] and row["task"] in context_tasks
    ]
    base_rows = [
        row
        for row in ablations
        if row["ablated_states"] == ["context"] and row["task"] not in context_tasks
    ]
    other_rows = [
        row
        for row in ablations
        if row["ablated_states"] != ["context"] and row["task"] in context_tasks
    ]
    context_accuracy = mean(
        [row for row in evaluations if row["task"] in context_tasks], "accuracy"
    )
    numbers = {
        "context_task_mean_accuracy": context_accuracy,
        "context_mean_accuracy_change_context_tasks": mean(context_rows, "accuracy_change"),
        "context_mean_loss_change_context_tasks": mean(context_rows, "loss_change"),
        "context_mean_accuracy_change_base_tasks": mean(base_rows, "accuracy_change"),
        "context_mean_prediction_change_rate_context_tasks": mean(context_rows, "prediction_change_rate"),
        "other_states_mean_accuracy_change_context_tasks": mean(other_rows, "accuracy_change"),
        "other_states_mean_loss_change_context_tasks": mean(other_rows, "loss_change"),
    }
    drop = -numbers["context_mean_accuracy_change_context_tasks"]
    others = -numbers["other_states_mean_accuracy_change_context_tasks"]
    if drop >= 0.10 and drop >= 0.25 * max(others, 1e-9):
        verdict = (
            "context_state wird auf den Aufgaben mit mittlerer zeitlicher Struktur kausal "
            f"relevant: seine Ablation kostet dort im Mittel {drop:.1%} Accuracy "
            f"(andere Zustände: {others:.1%})."
        )
    elif drop >= 0.02:
        verdict = (
            "context_state trägt auf den neuen Aufgaben messbar, aber deutlich schwächer bei "
            f"als fast_state und semantic_state ({drop:.1%} gegenüber {others:.1%} Accuracy-Verlust)."
        )
    else:
        verdict = (
            "context_state bleibt auch auf Aufgaben mit expliziter Abschnittsstruktur kausal "
            f"nahezu wirkungslos ({drop:.1%} Accuracy-Verlust bei Ablation). Die Aufgaben "
            "werden offenbar weiterhin über fast_state und die gebundene semantische "
            "Interaktion gelöst."
        )
    if context_accuracy < RELIABLE_ACCURACY:
        verdict += (
            f" **Einschränkung:** Das Modell löst die Kontextaufgaben nur zu {context_accuracy:.1%} "
            f"(Schwelle {RELIABLE_ACCURACY:.0%}). Eine Ablation an einem Modell, das die Aufgabe "
            "nicht beherrscht, zeigt bestenfalls, worauf sein unvollständiges Verfahren beruht – "
            "nicht, welcher Zustand für eine Lösung nötig wäre. Diese Zahlen sind entsprechend "
            "vorsichtig zu lesen."
        )
        numbers["reliable"] = 0.0
    else:
        numbers["reliable"] = 1.0
    return verdict, numbers


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 2.5: Kontextspezialisierung messen")
    parser.add_argument("--config", type=Path, default=Path("configs/context_specialization.json"))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--minimum-accuracy", type=float, default=0.0,
                        help="Optionale Abschlussgrenze; 0 meldet nur, ohne zu scheitern")
    parser.add_argument("--only-context", action="store_true",
                        help="Trainiert ausschließlich die Aufgaben mit Abschnittsstruktur")
    parser.add_argument("--ablation-distance", type=int, default=64)
    parser.add_argument("--ablation-batch-size", type=int, default=64)
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    training_values = dict(raw["training"])
    training_values["train_distances"] = tuple(training_values["train_distances"])
    training_values["tasks"] = tuple(training_values["tasks"])
    if args.steps is not None:
        training_values["steps"] = args.steps
    if args.only_context:
        training_values["tasks"] = tuple(
            task for task in training_values["tasks"] if task in CONTEXT_TASK_GENERATORS
        )
        if not training_values["tasks"]:
            raise SystemExit("--only-context benötigt mindestens eine Kontextaufgabe in der Config")
    training_config = StateIntelligenceTrainingConfig(**training_values)
    evaluation_config = raw["evaluation"]
    distances = tuple(int(value) for value in evaluation_config["distances"])
    model_config = ModelConfig.from_dict(raw["model"])
    vocabulary = StateTaskVocabulary(section_count=int(raw["tokenizer"]["section_count"]))
    if model_config.vocab_size != vocabulary.vocab_size:
        raise ValueError(
            f"Modellvokabular {model_config.vocab_size} passt nicht zum Aufgabenvokabular "
            f"{vocabulary.vocab_size}"
        )

    seed_everything(training_config.seed)
    capabilities = detect_device(args.device, args.precision)
    model = GlassMindLM(model_config).to(capabilities.torch_device)
    run = RunDirectory(args.run_root, prefix="context-specialization")
    run.write_json(
        "config.json",
        {
            "milestone": 2.5,
            "model": model_config.to_dict(),
            "training": training_config.to_dict(),
            "evaluation": evaluation_config,
            "tokenizer": vocabulary.to_dict(),
        },
    )
    run.write_json("environment.json", environment_metadata(capabilities, seed=training_config.seed))
    run.log(
        f"[Start] Kontextspezialisierung  Lauf={run.path}  Backend={capabilities.backend}  "
        f"Precision={capabilities.precision}  Parameter={model.parameter_count:,}  "
        f"Aufgaben={', '.join(training_config.tasks)}"
    )

    training_metrics, optimizer = train_state_intelligence(
        model, training_config, capabilities, vocabulary=vocabulary, logger=run
    )

    evaluations: list[dict[str, Any]] = []
    for task in training_config.tasks:
        evaluations.extend(
            evaluate_state_task(
                model,
                _generator_for(task),
                distances=distances,
                batch_size=int(evaluation_config["batch_size"]),
                repeats=int(evaluation_config["repeats"]),
                seed=training_config.seed + 10_000,
                capabilities=capabilities,
                task_kwargs=_task_kwargs(task, training_config, vocabulary),
            )
        )
    for result in evaluations:
        run.metric({"event": "distance_evaluation", **result})
        run.log(
            f"[evaluation] task={result['task']}  distanz={result['distance']}  "
            f"loss={float(result['loss']):.4f}  accuracy={float(result['accuracy']):.1%}"
        )

    ablations: list[dict[str, Any]] = []
    for index, task in enumerate(training_config.tasks):
        batch = _generator_for(task)(
            batch_size=args.ablation_batch_size,
            distance=args.ablation_distance,
            seed=training_config.seed + 20_000 + index * 1_000,
            **_task_kwargs(task, training_config, vocabulary),
        ).to(capabilities.torch_device)
        for state_name in ("fast", "context", "semantic"):
            with autocast_context(capabilities):
                comparison = ablation_comparison(model, batch, (state_name,))
            comparison["task"] = task
            comparison["distance"] = args.ablation_distance
            ablations.append(comparison)
            run.metric({"event": "state_ablation", **comparison})
            run.log(
                f"[ablation] task={task}  state={state_name}  "
                f"delta_loss={comparison['loss_change']:+.4f}  "
                f"delta_acc={comparison['accuracy_change']:+.1%}  "
                f"prediction_change={comparison['prediction_change_rate']:.1%}"
            )

    context_tasks = set(CONTEXT_TASK_GENERATORS)
    verdict, verdict_numbers = _verdict(ablations, evaluations, context_tasks)
    run.log(f"[Befund] {verdict}")

    trace_task = next(
        (task for task in training_config.tasks if task in context_tasks), training_config.tasks[0]
    )
    trace_batch = _generator_for(trace_task)(
        batch_size=1,
        distance=args.ablation_distance,
        seed=training_config.seed + 40_000,
        **_task_kwargs(trace_task, training_config, vocabulary),
    ).to(capabilities.torch_device)
    trace_path = run.traces / "context-specialization.jsonl"
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
            "context_state_verdict": verdict,
            "context_state_numbers": verdict_numbers,
            "trace": str(trace_path),
        },
    )
    save_checkpoint(
        run.checkpoints / "final.pt",
        model,
        tokenizer=vocabulary,
        optimizer=optimizer,
        step=training_config.steps,
        extra={"milestone": 2.5, "tasks": list(training_config.tasks)},
    )

    passed = all(float(result["accuracy"]) >= args.minimum_accuracy for result in evaluations)
    lines = [
        f"# Milestone 2.5 Kontextspezialisierung: {'PASS' if passed else 'FAIL'}",
        "",
        f"- Architektur: `glassmind_selective_recurrent_v1`",
        f"- Parameter: {model.parameter_count:,}",
        f"- Trainingsaufgaben: {', '.join(training_config.tasks)}",
        f"- Trainingsschritte: {training_metrics['steps']}",
        f"- Bester Trainings-Loss: {float(training_metrics['best_loss']):.4f}",
        f"- Trainingsdurchsatz: {float(training_metrics['tokens_per_second']):,.0f} Token/s",
        f"- Peak-Gerätespeicher: {peak_memory_bytes(capabilities) or 'nicht verfügbar'} Byte",
        f"- Replay-Trace: `{trace_path}`",
        "",
        "## Distanz-Benchmarks",
        "",
        *_evaluation_table(evaluations),
        "",
        f"## State-Ablation bei Distanz {args.ablation_distance}",
        "",
        *_ablation_table(ablations),
        "",
        "## Befund zu `context_state`",
        "",
        verdict,
        "",
        "Die Aufgaben wurden gebaut, um eine mittlere Zeitskala *nützlich* zu machen: gleiche",
        "Schlüssel tragen in verschiedenen Abschnitten verschiedene Werte, Themen werden nach",
        "einer Unterbrechung wieder aufgenommen, und eine dokumentweite Konstante steht neben",
        "abschnittslokalen Fakten. Ob `context_state` diese Rolle übernimmt, entscheidet das",
        "Training; die Architektur wurde dafür nicht angepasst.",
        "",
        "## Gemessene zeitliche State-Eigenschaften",
        "",
        "| State | Aktivierung | Delta | Update-Gate | Retention | Zeitkonstante | Persistenz | Reaktivierungen | Fluss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
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
            "Zeitkonstanten sind Näherungen aus der relativen realen Zustandsänderung, keine direkt",
            "identifizierten kausalen Parameter.",
            "",
        ]
    )
    (run.path / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    run.log(f"[{'PASS' if passed else 'FAIL'}] Ergebnisse={run.path / 'results.json'}")
    if not passed:
        raise SystemExit(
            f"Mindestens eine Distanzmessung liegt unter {args.minimum_accuracy:.0%}; "
            f"siehe {run.path / 'summary.md'}"
        )


if __name__ == "__main__":
    main()
