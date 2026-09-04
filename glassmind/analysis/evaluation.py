from __future__ import annotations

from collections.abc import Collection
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F

from glassmind.data.state_tasks import StateTaskBatch
from glassmind.model.lm import GlassMindLM


def masked_lm_metrics(logits: Tensor, batch: StateTaskBatch) -> dict[str, float]:
    selected_logits = logits[batch.loss_mask]
    selected_targets = batch.targets[batch.loss_mask]
    if selected_targets.numel() == 0:
        raise ValueError("Die Aufgabenmaske enthält keine Zieltoken")
    loss = F.cross_entropy(selected_logits.float(), selected_targets)
    accuracy = (selected_logits.argmax(dim=-1) == selected_targets).float().mean()
    return {
        "loss": float(loss.detach().item()),
        "accuracy": float(accuracy.detach().item()),
        "answer_tokens": int(selected_targets.numel()),
    }


@torch.inference_mode()
def ablation_comparison(
    model: GlassMindLM,
    batch: StateTaskBatch,
    states: Collection[str],
) -> dict[str, Any]:
    baseline_logits, _ = model(batch.input_ids)
    ablated_logits, _ = model(batch.input_ids, ablate_states=states)
    baseline = masked_lm_metrics(baseline_logits, batch)
    ablated = masked_lm_metrics(ablated_logits, batch)
    selected_baseline = baseline_logits[batch.loss_mask].float()
    selected_ablated = ablated_logits[batch.loss_mask].float()
    difference = selected_ablated - selected_baseline
    changed = selected_baseline.argmax(dim=-1) != selected_ablated.argmax(dim=-1)
    return {
        "ablated_states": sorted(set(states)),
        "baseline": baseline,
        "ablated": ablated,
        "loss_change": ablated["loss"] - baseline["loss"],
        "accuracy_change": ablated["accuracy"] - baseline["accuracy"],
        "logit_mean_absolute_difference": float(difference.abs().mean().item()),
        "logit_rms_difference": float(difference.square().mean().sqrt().item()),
        "prediction_change_rate": float(changed.float().mean().item()),
    }
