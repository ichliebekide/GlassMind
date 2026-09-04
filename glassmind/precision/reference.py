"""Erhebt die vollständige Kennzahlenlage eines Modells für Precision-Vergleiche.

Alles, was Milestone 2.6 vergleichen soll, wird hier an einer Stelle gemessen:
Durchsatz, Latenz, Speicher, Aufgabenqualität, Zustandsrollen und Logits. Die
Funktion wird sowohl für die eingefrorene Milestone-2.5-Referenz als auch für
jede geprüfte Precision-Variante verwendet – nur so sind die Zahlen vergleichbar.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch
import torch.nn.functional as F

from glassmind.analysis.evaluation import ablation_comparison, masked_lm_metrics
from glassmind.data.state_tasks import (
    StateTaskVocabulary,
    generate_associative_recall_batch,
    generate_selective_copy_batch,
)
from glassmind.observe.bus import ObservationBus, ObservationMode
from glassmind.precision.apply import precision_report, state_dtype_report
from glassmind.precision.quantization import parameter_storage_bytes
from glassmind.utils.device import (
    DeviceCapabilities,
    peak_memory_bytes,
    reset_peak_memory,
    synchronize,
)

if TYPE_CHECKING:  # Nur für Typprüfung – zur Laufzeit gäbe es einen Ringschluss.
    from glassmind.model.lm import GlassMindLM

STATE_NAMES = ("fast", "context", "semantic")
DEFAULT_DISTANCES = (16, 64, 256, 1024)


def _host_rss_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (ImportError, AttributeError, OSError):  # pragma: no cover - plattformabhängig
        return None
    return int(usage) * (1 if usage > 2**32 else 1024)


@torch.inference_mode()
def measure_throughput(
    model: "GlassMindLM",
    capabilities: DeviceCapabilities,
    *,
    batch_size: int = 1,
    length: int = 128,
    iterations: int = 5,
) -> dict[str, float | int | None]:
    device = capabilities.torch_device
    tokens = torch.randint(0, model.config.vocab_size, (batch_size, length), device=device)
    model(tokens)
    synchronize(capabilities)
    started = time.perf_counter()
    for _ in range(iterations):
        model(tokens)
    synchronize(capabilities)
    elapsed = time.perf_counter() - started
    return {
        "tokens_per_second": tokens.numel() * iterations / elapsed,
        "sequence_length": length,
        "batch_size": batch_size,
    }


@torch.inference_mode()
def measure_streaming(
    model: "GlassMindLM",
    capabilities: DeviceCapabilities,
    *,
    batch_size: int = 1,
    steps: int = 128,
    repeats: int = 5,
) -> dict[str, float]:
    import statistics

    device = capabilities.torch_device
    token = torch.zeros(batch_size, dtype=torch.long, device=device)
    state = None
    for _ in range(16):
        _, state = model.step(token, state)
    synchronize(capabilities)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(steps):
            _, state = model.step(token, state)
        synchronize(capabilities)
        samples.append((time.perf_counter() - started) * 1000 / steps)
    return {
        "streaming_ms_per_token": statistics.median(samples),
        "streaming_ms_per_token_min": min(samples),
        "streaming_ms_per_token_max": max(samples),
    }


def measure_training_throughput(
    model: "GlassMindLM",
    capabilities: DeviceCapabilities,
    *,
    batch_size: int = 16,
    length: int = 96,
    steps: int = 12,
) -> dict[str, Any]:
    """Kurzer Trainingsdurchsatz. Quantisierte Gewichte sind nicht trainierbar.

    Die Messung führt echte Optimizer-Schritte aus und verändert dabei die
    Gewichte. Sie werden anschließend exakt wiederhergestellt, damit alle
    übrigen Kennzahlen am unveränderten Modell gemessen werden. Ohne diese
    Sicherung würden reduzierte Formate bereits an der Messung selbst
    scheitern – FP16 divergiert hier ohne Loss-Scaling.
    """
    import copy

    from glassmind.precision.quantization import iter_quantized

    if any(True for _ in iter_quantized(model)):
        return {
            "training_tokens_per_second": None,
            "training_note": "Weight-Only-quantisierte Module haben keine trainierbaren Gewichte",
        }
    device = capabilities.torch_device
    snapshot = {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}
    was_training = model.training
    try:
        model.train()
        tokens = torch.randint(0, model.config.vocab_size, (batch_size, length), device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        finite = True

        def train_step() -> None:
            nonlocal finite
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(tokens)
            loss = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size).float(), tokens.reshape(-1)
            )
            if not torch.isfinite(loss):
                finite = False
            loss.backward()
            optimizer.step()

        for _ in range(2):
            train_step()
        synchronize(capabilities)
        started = time.perf_counter()
        for _ in range(steps):
            train_step()
        synchronize(capabilities)
        elapsed = time.perf_counter() - started
        result: dict[str, Any] = {
            "training_tokens_per_second": tokens.numel() * steps / elapsed,
            "training_loss_finite": finite,
            "training_peak_memory_bytes": peak_memory_bytes(capabilities),
        }
        if not finite:
            result["training_note"] = (
                "Der Trainings-Loss wurde nicht-endlich. In diesem Format ist Training ohne "
                "Loss-Scaling nicht möglich."
            )
        return result
    finally:
        model.load_state_dict(snapshot)
        model.train(was_training)


@torch.inference_mode()
def measure_tasks(
    model: "GlassMindLM",
    capabilities: DeviceCapabilities,
    vocabulary: StateTaskVocabulary,
    *,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    batch_size: int = 32,
    seed: int = 51_000,
    associations: int = 3,
    copy_items: int = 2,
) -> list[dict[str, Any]]:
    """Assoziativer Abruf und selektives Kopieren über alle Distanzen."""
    generators: tuple[tuple[str, Callable[..., Any], dict[str, Any]], ...] = (
        ("associative_recall", generate_associative_recall_batch, {"associations": associations}),
        ("selective_copy", generate_selective_copy_batch, {"items": copy_items}),
    )
    results: list[dict[str, Any]] = []
    for task, generator, kwargs in generators:
        for distance in distances:
            batch = generator(
                batch_size=batch_size,
                distance=distance,
                seed=seed + distance * 1009,
                vocabulary=vocabulary,
                **kwargs,
            ).to(capabilities.torch_device)
            logits, _ = model(batch.input_ids)
            metrics = masked_lm_metrics(logits, batch)
            results.append({"task": task, "distance": distance, **metrics})
    return results


@torch.inference_mode()
def measure_ablations(
    model: "GlassMindLM",
    capabilities: DeviceCapabilities,
    vocabulary: StateTaskVocabulary,
    *,
    distance: int = 64,
    batch_size: int = 64,
    seed: int = 52_000,
    associations: int = 3,
    copy_items: int = 2,
) -> list[dict[str, Any]]:
    batches = (
        (
            "associative_recall",
            generate_associative_recall_batch(
                batch_size=batch_size, distance=distance, associations=associations,
                seed=seed, vocabulary=vocabulary,
            ),
        ),
        (
            "selective_copy",
            generate_selective_copy_batch(
                batch_size=batch_size, distance=distance, items=copy_items,
                seed=seed + 1000, vocabulary=vocabulary,
            ),
        ),
    )
    results: list[dict[str, Any]] = []
    for task, batch in batches:
        device_batch = batch.to(capabilities.torch_device)
        for state_name in STATE_NAMES:
            comparison = ablation_comparison(model, device_batch, (state_name,))
            comparison["task"] = task
            comparison["distance"] = distance
            results.append(comparison)
    return results


@torch.inference_mode()
def measure_state_roles(
    model: "GlassMindLM",
    capabilities: DeviceCapabilities,
    vocabulary: StateTaskVocabulary,
    *,
    distance: int = 64,
    seed: int = 53_000,
    copy_items: int = 2,
) -> dict[str, Any]:
    """Zustandsnormen und Zeitkonstanten aus realer Telemetrie."""
    from glassmind.analysis.clusters import StateMetricsAnalyzer

    batch = generate_selective_copy_batch(
        batch_size=1, distance=distance, items=copy_items, seed=seed, vocabulary=vocabulary
    ).to(capabilities.torch_device)
    analyzer = StateMetricsAnalyzer()
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(analyzer)
    model.eval()
    logits, final_state = model(batch.input_ids, observer=bus)
    bus.close()
    summaries = analyzer.summaries()
    norms = {
        name: float(
            torch.linalg.vector_norm(
                torch.stack(
                    [getattr(block, name).float().flatten() for block in final_state.blocks]
                )
            )
        )
        for name in STATE_NAMES
    }
    return {
        "state_metrics": {
            name: {
                "mean_activation_strength": values["mean_activation_strength"],
                "mean_delta": values["mean_delta"],
                "mean_update_gate": values["mean_update_gate"],
                "mean_retention_activity": values["mean_retention_activity"],
                "mean_estimated_time_constant": values["mean_estimated_time_constant"],
                "mean_persistence": values["mean_persistence"],
                "reactivations": values["reactivations"],
                "mean_information_flow": values["mean_information_flow"],
            }
            for name, values in summaries.items()
        },
        "final_state_norms": norms,
    }


@torch.inference_mode()
def reference_logits(
    model: "GlassMindLM",
    capabilities: DeviceCapabilities,
    vocabulary: StateTaskVocabulary,
    *,
    seed: int = 54_000,
    distance: int = 64,
    batch_size: int = 8,
    associations: int = 3,
) -> dict[str, Any]:
    """Ein fester, reproduzierbarer Logit-Ausschnitt als Vergleichsanker."""
    batch = generate_associative_recall_batch(
        batch_size=batch_size, distance=distance, associations=associations,
        seed=seed, vocabulary=vocabulary,
    ).to(capabilities.torch_device)
    logits, _ = model(batch.input_ids)
    selected = logits[batch.loss_mask].float().cpu()
    return {
        "seed": seed,
        "distance": distance,
        "shape": list(selected.shape),
        "values": selected.tolist(),
        "argmax": selected.argmax(dim=-1).tolist(),
    }


def collect_reference(
    model: "GlassMindLM",
    capabilities: DeviceCapabilities,
    vocabulary: StateTaskVocabulary,
    *,
    label: str,
    distances: Sequence[int] = DEFAULT_DISTANCES,
    throughput_length: int = 128,
    iterations: int = 5,
    include_training: bool = True,
    include_logits: bool = True,
    task_batch_size: int = 32,
) -> dict[str, Any]:
    """Erhebt die vollständige Kennzahlenlage eines Modells."""
    model.eval()
    # Der Allocator-Cache früherer Varianten würde den Spitzenwert verfälschen.
    if capabilities.backend in {"cuda", "rocm"}:
        torch.cuda.empty_cache()
    reset_peak_memory(capabilities)
    fallback = next(
        (parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()),
        torch.float32,
    )
    record: dict[str, Any] = {
        "label": label,
        "model_config": model.config.to_dict(),
        "parameter_count": model.parameter_count,
        "parameter_storage_bytes": parameter_storage_bytes(model),
        "precision": {**state_dtype_report(model, fallback),
                      "weight_dtype": str(fallback).removeprefix("torch.")},
        "precision_report": precision_report(model),
        "hardware": capabilities.to_dict(),
        **measure_throughput(
            model, capabilities, length=throughput_length, iterations=iterations
        ),
        **measure_streaming(model, capabilities),
        "tasks": measure_tasks(
            model, capabilities, vocabulary, distances=distances, batch_size=task_batch_size
        ),
        "ablations": measure_ablations(model, capabilities, vocabulary),
        **measure_state_roles(model, capabilities, vocabulary),
    }
    if include_logits:
        record["logits"] = reference_logits(model, capabilities, vocabulary)
    # Spitzenwert der reinen Inferenz – die Trainingsmessung darunter allokiert
    # Gradienten und Optimizer-Zustände und würde den Vergleich verzerren.
    record["peak_memory_bytes"] = peak_memory_bytes(capabilities)
    record["host_peak_rss_bytes"] = _host_rss_bytes()
    if include_training:
        record.update(measure_training_throughput(model, capabilities))
    return record


def task_accuracy(record: dict[str, Any], task: str, distance: int) -> float | None:
    for entry in record.get("tasks", []):
        if entry["task"] == task and entry["distance"] == distance:
            return float(entry["accuracy"])
    return None
