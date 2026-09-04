from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor


def scalar(value: Tensor) -> float:
    return float(value.detach().float().item())


def tensor_summary(value: Tensor) -> dict[str, float | bool]:
    data = value.detach().float()
    # Sämtliche Reduktionen bleiben bis zu einem einzigen kompakten Transfer auf dem Gerät.
    safe = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    packed = torch.stack(
        (
            safe.mean(),
            safe.std(unbiased=False),
            safe.min(),
            safe.max(),
            torch.linalg.vector_norm(safe),
            (safe.abs() < 1e-6).float().mean(),
            (safe.abs() > 0.99).float().mean(),
            torch.isnan(data).any().float(),
            torch.isinf(data).any().float(),
        )
    ).cpu().tolist()
    return {
        "mean": packed[0],
        "std": packed[1],
        "min": packed[2],
        "max": packed[3],
        "norm": packed[4],
        "sparsity": packed[5],
        "saturation": packed[6],
        "has_nan": bool(packed[7]),
        "has_inf": bool(packed[8]),
    }


# Reihenfolge der je Cluster gepackten Kennzahlen. Die Konstante hält
# Berechnung (auf dem Gerät) und Auswertung (auf dem Host) synchron.
CLUSTER_STATISTIC_COUNT = 12


def _cluster_view(value: Tensor, cluster_count: int) -> Tensor | None:
    """Formt [batch, breite] zu [batch, cluster, kanäle], wenn das exakt aufgeht."""
    width = value.shape[-1]
    if width % cluster_count:
        return None
    return value.reshape(*value.shape[:-1], cluster_count, width // cluster_count)


def cluster_statistics(
    state: Tensor,
    delta: Tensor,
    gate: Tensor,
    incoming_flow: Tensor,
    cluster_count: int,
) -> Tensor:
    """Liefert [cluster_count, CLUSTER_STATISTIC_COUNT] – ohne Host-Transfer.

    Bei gleichmäßig teilbarer Breite laufen alle Cluster in einem einzigen
    Reduktionsaufruf statt in ``cluster_count`` einzelnen Aufrufen. Der
    Aufteilungspfad bleibt als Fallback erhalten und liefert dieselben Zahlen.
    """
    tensors = [value.detach().float() for value in (state, delta, gate, incoming_flow)]
    views = [_cluster_view(value, cluster_count) for value in tensors]
    if all(view is not None for view in views):
        state_v, delta_v, gate_v, flow_v = views
        reduce = (0, 2) if state_v.ndim == 3 else tuple(range(state_v.ndim - 1))
        return torch.stack(
            (
                state_v.square().mean(dim=reduce).sqrt(),
                delta_v.square().mean(dim=reduce).sqrt(),
                gate_v.mean(dim=reduce),
                flow_v.square().mean(dim=reduce).sqrt(),
                state_v.mean(dim=reduce),
                state_v.std(dim=reduce, unbiased=False),
                state_v.amin(dim=reduce),
                state_v.amax(dim=reduce),
                (state_v.abs() < 1e-6).float().mean(dim=reduce),
                (state_v - delta_v).square().mean(dim=reduce).sqrt(),
                torch.linalg.vector_norm(state_v, dim=reduce),
                torch.linalg.vector_norm(delta_v, dim=reduce),
            ),
            dim=-1,
        )
    chunks = [torch.tensor_split(value, cluster_count, dim=-1) for value in tensors]
    rows = []
    for state_part, delta_part, gate_part, flow_part in zip(*chunks, strict=True):
        rows.append(
            torch.stack(
                (
                    state_part.square().mean().sqrt(),
                    delta_part.square().mean().sqrt(),
                    gate_part.mean(),
                    flow_part.square().mean().sqrt(),
                    state_part.mean(),
                    state_part.std(unbiased=False),
                    state_part.min(),
                    state_part.max(),
                    (state_part.abs() < 1e-6).float().mean(),
                    (state_part - delta_part).square().mean().sqrt(),
                    torch.linalg.vector_norm(state_part),
                    torch.linalg.vector_norm(delta_part),
                )
            )
        )
    return torch.stack(rows)


def cluster_flow_rms(flow: Tensor, cluster_count: int) -> Tensor:
    """Effektivwert des Flusses je Clustergruppe, als ein Gerätetensor."""
    value = flow.detach().float()
    view = _cluster_view(value, cluster_count)
    if view is not None:
        reduce = (0, 2) if view.ndim == 3 else tuple(range(view.ndim - 1))
        return view.square().mean(dim=reduce).sqrt()
    return torch.stack(
        [chunk.square().mean().sqrt() for chunk in torch.tensor_split(value, cluster_count, dim=-1)]
    )


def build_cluster_nodes(
    *,
    layer: int,
    state_name: str,
    rows: list[list[float]],
    activity_weights: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """Baut die Knoten-Dicts aus bereits übertragenen Kennzahlen."""
    nodes: list[dict[str, Any]] = []
    for cluster, packed in enumerate(rows):
        activation, change, gate_activity, flow = packed[:4]
        activity = sum(
            weight * component for weight, component in zip(activity_weights, packed[:4], strict=True)
        )
        forget_activity = min(1.0, change / max(packed[9], 1e-6))
        # Ein Gate ist je nach State ein Interpolations- oder Schreib-Gate. Eine
        # scheinbar exakte Retention aus ``1 - gate`` wäre daher irreführend.
        # Dieser backendunabhängige Proxy misst stattdessen die relative reale
        # Zustandsänderung zum vorherigen Token.
        retention = max(0.0, min(1.0, 1.0 - forget_activity))
        time_constant = -1.0 / math.log(max(retention, 1e-6)) if retention < 0.999999 else 1_000_000.0
        nodes.append(
            {
                "id": f"core.{layer}.{state_name}.cluster.{cluster}",
                "kind": state_name,
                "cluster": cluster,
                "activity": activity,
                "components": {
                    "activation_rms": activation,
                    "delta_rms": change,
                    "gate_mean": gate_activity,
                    "incoming_flow_rms": flow,
                    "activation_mean": packed[4],
                    "activation_std": packed[5],
                    "activation_min": packed[6],
                    "activation_max": packed[7],
                    "activation_sparsity": packed[8],
                    "previous_state_rms": packed[9],
                    "state_norm": packed[10],
                    "delta_norm": packed[11],
                    "update_gate_activity": gate_activity,
                    "retention_activity": retention,
                    "forget_activity": forget_activity,
                    "estimated_time_constant": time_constant,
                    "information_flow": flow,
                },
            }
        )
    return nodes


_SUMMARY_FIELDS = ("mean", "std", "min", "max", "norm", "sparsity", "saturation", "has_nan", "has_inf")


def sequence_tensor_summary(value: Tensor) -> list[dict[str, float | bool]]:
    """``tensor_summary`` für jede Position von [batch, sequence, ...] auf einmal.

    Erspart einen eigenen Host-Transfer je Token.
    """
    data = value.detach().float()
    reduce = (0,) + tuple(range(2, data.ndim))
    safe = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    packed = torch.stack(
        (
            safe.mean(dim=reduce),
            safe.std(dim=reduce, unbiased=False),
            safe.amin(dim=reduce),
            safe.amax(dim=reduce),
            torch.linalg.vector_norm(safe, dim=reduce),
            (safe.abs() < 1e-6).float().mean(dim=reduce),
            (safe.abs() > 0.99).float().mean(dim=reduce),
            torch.isnan(data).any(dim=reduce).float(),
            torch.isinf(data).any(dim=reduce).float(),
        ),
        dim=-1,
    ).cpu().tolist()
    summaries: list[dict[str, float | bool]] = []
    for row in packed:
        summary: dict[str, float | bool] = dict(zip(_SUMMARY_FIELDS[:7], row[:7], strict=True))
        summary["has_nan"] = bool(row[7])
        summary["has_inf"] = bool(row[8])
        summaries.append(summary)
    return summaries


def clustered_state_nodes(
    *,
    layer: int,
    state_name: str,
    state: Tensor,
    delta: Tensor,
    gate: Tensor,
    incoming_flow: Tensor,
    cluster_count: int,
    activity_weights: tuple[float, float, float, float],
) -> list[dict[str, Any]]:
    """Bequemer Einzelaufruf mit genau einem Host-Transfer."""
    rows = cluster_statistics(state, delta, gate, incoming_flow, cluster_count).cpu().tolist()
    return build_cluster_nodes(
        layer=layer, state_name=state_name, rows=rows, activity_weights=activity_weights
    )
