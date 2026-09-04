from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Iterable

import torch
from torch import Tensor
import torch.nn.functional as F

from glassmind.model.lm import GlassMindLM
from glassmind.observe.bus import ObservationBus, ObservationMode
from glassmind.utils.device import DeviceCapabilities, autocast_context, peak_memory_bytes, reset_peak_memory, synchronize


@dataclass(frozen=True)
class TrainingConfig:
    steps: int = 200
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    log_every: int = 20
    seed: int = 17

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _grad_scaler(capabilities: DeviceCapabilities) -> torch.amp.GradScaler | None:
    if capabilities.precision != "float16" or capabilities.backend not in {"cuda", "rocm", "xpu"}:
        return None
    device_type = "cuda" if capabilities.backend in {"cuda", "rocm"} else "xpu"
    return torch.amp.GradScaler(device_type)


def train_steps(
    model: GlassMindLM,
    batches: Iterable[tuple[Tensor, Tensor]],
    config: TrainingConfig,
    capabilities: DeviceCapabilities,
    *,
    logger: object | None = None,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float | int | bool]:
    model.train()
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = _grad_scaler(capabilities)
    iterator = iter(batches)
    best_loss = math.inf
    final_loss = math.inf
    seen_tokens = 0
    started = time.perf_counter()
    reset_peak_memory(capabilities)
    for step in range(1, config.steps + 1):
        try:
            inputs, targets = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            inputs, targets = next(iterator)
        inputs = inputs.to(capabilities.torch_device)
        targets = targets.to(capabilities.torch_device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(capabilities):
            logits, _ = model(inputs)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nicht-endlicher Loss in Schritt {step}: {loss.item()}")
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
        final_loss = float(loss.detach().item())
        best_loss = min(best_loss, final_loss)
        seen_tokens += targets.numel()
        if logger is not None and (step == 1 or step % config.log_every == 0 or step == config.steps):
            elapsed = max(time.perf_counter() - started, 1e-9)
            message = f"[training] Schritt {step}/{config.steps}  loss={final_loss:.4f}  tok/s={seen_tokens / elapsed:,.0f}  grad_norm={float(grad_norm):.3f}"
            logger.log(message)
            logger.metric({"event": "training_step", "step": step, "loss": final_loss, "best_loss": best_loss, "tokens_per_second": seen_tokens / elapsed, "grad_norm": float(grad_norm)})
    synchronize(capabilities)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "final_loss": final_loss,
        "best_loss": best_loss,
        "tokens_per_second": seen_tokens / elapsed,
        "peak_memory_bytes": peak_memory_bytes(capabilities) or 0,
        "steps": config.steps,
        "nan_inf": False,
    }
