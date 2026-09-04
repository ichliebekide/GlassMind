from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class _ClusterHistory:
    active: bool = False
    ever_active: bool = False
    streak: int = 0
    completed_active_tokens: int = 0
    episodes: int = 0
    reactivations: int = 0
    activation_count: int = 0
    activity_sum: float = 0.0
    max_activity: float = 0.0
    token_counts: Counter[int] = field(default_factory=Counter)


class ActivityTracker:
    """Deterministischer, rein beobachtender Verlauf für Clusteraktivität."""

    def __init__(self) -> None:
        self._history: dict[str, _ClusterHistory] = {}

    def annotate(
        self,
        nodes: list[dict[str, Any]],
        *,
        token_id: int,
        update_threshold: float,
    ) -> None:
        for node in nodes:
            if ".cluster." not in str(node.get("id", "")):
                continue
            node_id = str(node["id"])
            components = node.setdefault("components", {})
            update_strength = float(components.get("delta_rms", 0.0))
            active = update_strength >= update_threshold
            history = self._history.setdefault(node_id, _ClusterHistory())
            reactivated = active and not history.active and history.ever_active
            if active:
                if not history.active:
                    history.episodes += 1
                history.streak += 1
                history.ever_active = True
                history.activation_count += 1
                activity = float(node.get("activity", 0.0))
                history.activity_sum += activity
                history.max_activity = max(history.max_activity, activity)
                history.token_counts[token_id] += 1
                if reactivated:
                    history.reactivations += 1
            elif history.active:
                history.completed_active_tokens += history.streak
                history.streak = 0
            history.active = active
            active_tokens = history.completed_active_tokens + history.streak
            average_duration = active_tokens / history.episodes if history.episodes else 0.0
            frequent_tokens = sorted(
                history.token_counts.items(), key=lambda item: (-item[1], item[0])
            )[:5]
            node["persistence"] = history.streak
            node["reactivation"] = reactivated
            components.update(
                {
                    "update_active": active,
                    "persistence_duration": history.streak,
                    "reactivation": reactivated,
                    "reactivation_count": history.reactivations,
                }
            )
            node["cluster_statistics"] = {
                "mean_activity": history.activity_sum / max(history.activation_count, 1),
                "max_activity": history.max_activity,
                "activation_count": history.activation_count,
                "average_activation_duration": average_duration,
                "reactivations": history.reactivations,
                "frequent_token_ids": [
                    {"token_id": frequent_id, "count": count}
                    for frequent_id, count in frequent_tokens
                ],
            }
