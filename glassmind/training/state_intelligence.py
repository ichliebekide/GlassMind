from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Callable

import torch
from torch import Tensor
import torch.nn.functional as F

from glassmind.analysis.evaluation import masked_lm_metrics
from glassmind.data.state_tasks import (
    CONTEXT_TASK_GENERATORS,
    MEMORY_TASK_GENERATORS,
    StateTaskBatch,
    StateTaskVocabulary,
    generate_associative_recall_batch,
    generate_selective_copy_batch,
)
from glassmind.model.lm import GlassMindLM
from glassmind.utils.device import DeviceCapabilities, autocast_context, synchronize


@dataclass(frozen=True)
class StateIntelligenceTrainingConfig:
    steps: int = 600
    batch_size: int = 48
    learning_rate: float = 5e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_every: int = 25
    seed: int = 41
    train_distances: tuple[int, ...] = (0, 8, 16, 32, 64)
    associations: int = 3
    copy_items: int = 2
    tokens_per_batch: int = 8192
    # Die Reihenfolge bestimmt, welche Aufgabe ein Schritt zieht. Der Standard
    # entspricht exakt dem Milestone-2-Wechsel zwischen beiden Aufgaben.
    tasks: tuple[str, ...] = ("associative_recall", "selective_copy")
    sections: int = 3
    facts_per_section: int = 2
    # Milestone 3: Parameter der Memory-Aufgaben.
    bindings: int = 4
    distractors: int = 6
    replacement_facts: int = 24
    retrievals: int = 3

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["train_distances"] = list(self.train_distances)
        data["tasks"] = list(self.tasks)
        return data


def masked_loss(logits: Tensor, batch: StateTaskBatch) -> Tensor:
    return F.cross_entropy(logits[batch.loss_mask].float(), batch.targets[batch.loss_mask])


def _scaler(capabilities: DeviceCapabilities) -> torch.amp.GradScaler | None:
    if capabilities.precision != "float16" or capabilities.backend not in {"cuda", "rocm", "xpu"}:
        return None
    device_type = "cuda" if capabilities.backend in {"cuda", "rocm"} else "xpu"
    return torch.amp.GradScaler(device_type)


def _estimated_length(
    config: StateIntelligenceTrainingConfig, task: str, distance: int, task_items: int
) -> int:
    """Grobe Sequenzlänge, um die Tokenzahl pro Batch stabil zu halten."""
    if task == "sectioned_recall":
        return distance + config.sections * (1 + 3 * config.facts_per_section) + 8
    if task == "topic_resumption":
        return distance + (config.sections + 1) * 4 + 8
    if task == "hierarchical_scope":
        return distance + 5 + config.sections * 4 + 8
    if task == "delayed_binding":
        return distance + 10
    if task == "multiple_bindings":
        return distance + 3 * config.bindings + 8
    if task == "distractor_recall":
        return distance + 3 * (config.distractors + 1) + 8
    if task == "memory_replacement":
        return distance + 3 * config.replacement_facts + 8
    if task == "repeated_retrieval":
        return distance + 4 * config.retrievals + 6
    return distance + 3 * max(task_items, config.copy_items) + 8


def _training_batch(
    config: StateIntelligenceTrainingConfig,
    step: int,
    vocabulary: StateTaskVocabulary,
) -> StateTaskBatch:
    progress = step / max(config.steps - 1, 1)
    if progress < 0.20:
        task_items = 1
    elif progress < 0.40:
        task_items = 2
    else:
        task_items = config.associations
    curriculum_size = min(
        len(config.train_distances),
        1 + max(0, int((progress - 0.25) * len(config.train_distances) / 0.75)),
    )
    task_count = len(config.tasks)
    distance = config.train_distances[(step // task_count) % curriculum_size]
    task = config.tasks[step % task_count]
    effective_batch_size = min(
        config.batch_size,
        max(4, config.tokens_per_batch // max(_estimated_length(config, task, distance, task_items), 1)),
    )
    seed = config.seed + step * 101
    if task == "associative_recall":
        return generate_associative_recall_batch(
            batch_size=effective_batch_size,
            distance=distance,
            associations=min(task_items, config.associations),
            seed=seed,
            vocabulary=vocabulary,
        )
    if task == "selective_copy":
        return generate_selective_copy_batch(
            batch_size=effective_batch_size,
            distance=distance,
            items=min(task_items, config.copy_items),
            seed=seed,
            vocabulary=vocabulary,
        )
    generator = CONTEXT_TASK_GENERATORS.get(task) or MEMORY_TASK_GENERATORS.get(task)
    if generator is None:
        raise ValueError(f"Unbekannte Trainingsaufgabe: {task}")
    kwargs: dict[str, Any] = {}
    if task in CONTEXT_TASK_GENERATORS:
        kwargs["sections"] = config.sections
        if task == "sectioned_recall":
            kwargs["facts_per_section"] = config.facts_per_section
        if task == "topic_resumption":
            kwargs["sections"] = max(2, config.sections - 1)
    elif task == "multiple_bindings":
        kwargs["bindings"] = config.bindings
    elif task == "distractor_recall":
        kwargs["distractors"] = config.distractors
    elif task == "memory_replacement":
        kwargs["facts"] = config.replacement_facts
    elif task == "repeated_retrieval":
        kwargs["retrievals"] = config.retrievals
    return generator(
        batch_size=effective_batch_size,
        distance=distance,
        seed=seed,
        vocabulary=vocabulary,
        **kwargs,
    )


def train_state_intelligence(
    model: GlassMindLM,
    config: StateIntelligenceTrainingConfig,
    capabilities: DeviceCapabilities,
    *,
    vocabulary: StateTaskVocabulary | None = None,
    logger: object | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[dict[str, float | int], torch.optim.Optimizer]:
    vocabulary = vocabulary or StateTaskVocabulary()
    optimizer = optimizer or torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = _scaler(capabilities)
    model.train()
    started = time.perf_counter()
    best_loss = math.inf
    final_loss = math.inf
    final_accuracy = 0.0
    seen_tokens = 0
    for step in range(config.steps):
        batch = _training_batch(config, step, vocabulary).to(capabilities.torch_device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(capabilities):
            logits, _ = model(batch.input_ids)
            loss = masked_loss(logits, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nicht-endlicher State-Intelligence-Loss in Schritt {step + 1}")
        if scaler is None:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        with torch.no_grad():
            predictions = logits[batch.loss_mask].argmax(dim=-1)
            final_accuracy = float((predictions == batch.targets[batch.loss_mask]).float().mean().item())
        final_loss = float(loss.detach().item())
        best_loss = min(best_loss, final_loss)
        seen_tokens += batch.input_ids.numel()
        current_step = step + 1
        if logger is not None and (
            current_step == 1
            or current_step % config.log_every == 0
            or current_step == config.steps
        ):
            elapsed = max(time.perf_counter() - started, 1e-9)
            message = (
                f"[state-training] Schritt {current_step}/{config.steps}  "
                f"task={batch.task}  distanz={batch.distance}  loss={final_loss:.4f}  "
                f"acc={final_accuracy:.1%}  tok/s={seen_tokens / elapsed:,.0f}"
            )
            logger.log(message)
            logger.metric(
                {
                    "event": "state_training_step",
                    "step": current_step,
                    "task": batch.task,
                    "distance": batch.distance,
                    "loss": final_loss,
                    "accuracy": final_accuracy,
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": seen_tokens / elapsed,
                }
            )
    synchronize(capabilities)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return (
        {
            "steps": config.steps,
            "final_loss": final_loss,
            "best_loss": best_loss,
            "final_accuracy": final_accuracy,
            "tokens_per_second": seen_tokens / elapsed,
        },
        optimizer,
    )


@torch.inference_mode()
def evaluate_state_task(
    model: GlassMindLM,
    generator: Callable[..., StateTaskBatch],
    *,
    distances: tuple[int, ...],
    batch_size: int,
    repeats: int,
    seed: int,
    capabilities: DeviceCapabilities,
    task_kwargs: dict[str, Any],
) -> list[dict[str, float | int | str]]:
    model.eval()
    results: list[dict[str, float | int | str]] = []
    for distance in distances:
        loss_sum = 0.0
        correct_sum = 0.0
        answer_tokens = 0
        started = time.perf_counter()
        for repeat in range(repeats):
            batch = generator(
                batch_size=batch_size,
                distance=distance,
                seed=seed + distance * 1009 + repeat,
                **task_kwargs,
            ).to(capabilities.torch_device)
            with autocast_context(capabilities):
                logits, _ = model(batch.input_ids)
            metrics = masked_lm_metrics(logits, batch)
            count = int(metrics["answer_tokens"])
            loss_sum += float(metrics["loss"]) * count
            correct_sum += float(metrics["accuracy"]) * count
            answer_tokens += count
        synchronize(capabilities)
        elapsed = time.perf_counter() - started
        results.append(
            {
                "task": generator.__name__.removeprefix("generate_").removesuffix("_batch"),
                "distance": distance,
                "loss": loss_sum / answer_tokens,
                "accuracy": correct_sum / answer_tokens,
                "answer_tokens": answer_tokens,
                "elapsed_seconds": elapsed,
            }
        )
    return results
