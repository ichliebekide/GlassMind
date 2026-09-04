from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Callable

from glassmind.observe.events import ObservationEvent


@dataclass
class _Aggregate:
    count: int = 0
    active_count: int = 0
    activity_sum: float = 0.0
    activation_strength_sum: float = 0.0
    state_norm_sum: float = 0.0
    max_activity: float = 0.0
    delta_sum: float = 0.0
    persistence_sum: float = 0.0
    update_sum: float = 0.0
    forget_sum: float = 0.0
    retention_sum: float = 0.0
    flow_sum: float = 0.0
    reactivations: int = 0
    average_duration_latest: float = 0.0
    token_counts: Counter[int] = field(default_factory=Counter)

    def add(self, node: dict[str, Any], token_id: int | None) -> None:
        components = node.get("components", {})
        active = bool(components.get("update_active", False))
        self.count += 1
        self.activity_sum += float(node.get("activity", 0.0))
        self.activation_strength_sum += float(components.get("activation_rms", 0.0))
        self.state_norm_sum += float(components.get("state_norm", 0.0))
        self.max_activity = max(self.max_activity, float(node.get("activity", 0.0)))
        self.delta_sum += float(components.get("delta_rms", 0.0))
        self.persistence_sum += float(components.get("persistence_duration", 0.0))
        self.update_sum += float(components.get("update_gate_activity", components.get("gate_mean", 0.0)))
        self.forget_sum += float(components.get("forget_activity", 0.0))
        self.retention_sum += float(components.get("retention_activity", 0.0))
        self.flow_sum += float(components.get("information_flow", components.get("incoming_flow_rms", 0.0)))
        self.reactivations += int(bool(components.get("reactivation", False)))
        self.average_duration_latest = float(
            node.get("cluster_statistics", {}).get(
                "average_activation_duration", self.average_duration_latest
            )
        )
        if active:
            self.active_count += 1
            if token_id is not None:
                self.token_counts[token_id] += 1

    def summary(self, token_name: Callable[[int], str] | None = None) -> dict[str, Any]:
        divisor = max(self.count, 1)
        frequent = sorted(self.token_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        mean_retention = self.retention_sum / divisor
        estimated_time_constant = (
            -1.0 / math.log(max(mean_retention, 1e-6))
            if mean_retention < 0.999999
            else 1_000_000.0
        )
        return {
            "samples": self.count,
            "activation_count": self.active_count,
            "activation_rate": self.active_count / divisor,
            "mean_activity": self.activity_sum / divisor,
            "mean_activation_strength": self.activation_strength_sum / divisor,
            "mean_state_norm": self.state_norm_sum / divisor,
            "max_activity": self.max_activity,
            "mean_delta": self.delta_sum / divisor,
            "mean_persistence": self.persistence_sum / divisor,
            "average_activation_duration": self.average_duration_latest,
            "mean_update_gate": self.update_sum / divisor,
            "mean_forget_activity": self.forget_sum / divisor,
            "mean_retention_activity": mean_retention,
            "mean_information_flow": self.flow_sum / divisor,
            "mean_estimated_time_constant": estimated_time_constant,
            "reactivations": self.reactivations,
            "frequent_tokens": [
                {
                    "token_id": token_id,
                    "token": token_name(token_id) if token_name else str(token_id),
                    "count": count,
                }
                for token_id, count in frequent
            ],
        }


class ClusterAnalyzer:
    """Reproduzierbarer Ereignisempfänger ohne semantische Clusterlabels."""

    def __init__(self) -> None:
        self._clusters: dict[str, _Aggregate] = defaultdict(_Aggregate)

    def __call__(self, event: ObservationEvent) -> None:
        if event.event != "network_step":
            return
        for node in event.payload.get("nodes", []):
            if ".cluster." in str(node.get("id", "")):
                self._clusters[str(node["id"])].add(node, event.token_id)

    def summaries(self, token_name: Callable[[int], str] | None = None) -> dict[str, dict[str, Any]]:
        return {
            node_id: self._clusters[node_id].summary(token_name)
            for node_id in sorted(self._clusters)
        }


class StateMetricsAnalyzer:
    def __init__(self) -> None:
        self._states: dict[str, _Aggregate] = defaultdict(_Aggregate)

    def __call__(self, event: ObservationEvent) -> None:
        if event.event != "network_step":
            return
        for node in event.payload.get("nodes", []):
            state_name = str(node.get("kind", ""))
            if state_name in {"fast", "context", "semantic"}:
                self._states[state_name].add(node, event.token_id)

    def summaries(self) -> dict[str, dict[str, Any]]:
        return {
            name: self._states[name].summary()
            for name in ("fast", "context", "semantic")
            if name in self._states
        }
