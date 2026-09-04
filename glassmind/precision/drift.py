"""Numerische Drift eines rekurrenten Modells gegenüber einer FP32-Referenz.

GlassMind trägt seinen Zustand über die gesamte Sequenz. Ein Präzisionsverlust
verschwindet deshalb nicht, sondern kann sich aufschaukeln. Dieses Modul misst
genau das: Zustand für Zustand, Länge für Länge.

Gemessen wird immer gegen dieselbe Eingabe und dieselben Gewichte. Der einzige
Unterschied zwischen Referenz und Prüfling ist die Zahlendarstellung.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import Tensor

from glassmind.utils.device import DeviceCapabilities, synchronize

if TYPE_CHECKING:  # Nur für Typprüfung – zur Laufzeit gäbe es einen Ringschluss.
    from glassmind.model.lm import GlassMindLM, ModelState

STATE_NAMES = ("fast", "context", "semantic")


def _relative_rms(test: Tensor, reference: Tensor) -> float:
    difference = (test.float() - reference.float()).square().mean().sqrt()
    scale = reference.float().square().mean().sqrt().clamp_min(1e-12)
    return float(difference / scale)


def _norm(value: Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()))


@dataclass
class DriftPoint:
    length: int
    state_relative_rms: dict[str, float] = field(default_factory=dict)
    state_absolute_rms: dict[str, float] = field(default_factory=dict)
    state_norm_reference: dict[str, float] = field(default_factory=dict)
    state_norm_test: dict[str, float] = field(default_factory=dict)
    state_norm_drift: dict[str, float] = field(default_factory=dict)
    logit_relative_rms: float = 0.0
    logit_max_absolute: float = 0.0
    prediction_change_rate: float = 0.0
    has_nan: bool = False
    has_inf: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "length": self.length,
            "state_relative_rms": self.state_relative_rms,
            "state_absolute_rms": self.state_absolute_rms,
            "state_norm_reference": self.state_norm_reference,
            "state_norm_test": self.state_norm_test,
            "state_norm_drift": self.state_norm_drift,
            "logit_relative_rms": self.logit_relative_rms,
            "logit_max_absolute": self.logit_max_absolute,
            "prediction_change_rate": self.prediction_change_rate,
            "has_nan": self.has_nan,
            "has_inf": self.has_inf,
        }


def _compare_states(reference: ModelState, test: ModelState) -> dict[str, dict[str, float]]:
    """Mittelt die Abweichung je Zustandsart über alle Blöcke."""
    relative: dict[str, list[float]] = {name: [] for name in STATE_NAMES}
    absolute: dict[str, list[float]] = {name: [] for name in STATE_NAMES}
    norm_reference: dict[str, list[float]] = {name: [] for name in STATE_NAMES}
    norm_test: dict[str, list[float]] = {name: [] for name in STATE_NAMES}
    for reference_block, test_block in zip(reference.blocks, test.blocks, strict=True):
        for name in STATE_NAMES:
            a = getattr(reference_block, name)
            b = getattr(test_block, name)
            relative[name].append(_relative_rms(b, a))
            absolute[name].append(float((b.float() - a.float()).square().mean().sqrt()))
            norm_reference[name].append(_norm(a))
            norm_test[name].append(_norm(b))

    def mean(values: dict[str, list[float]]) -> dict[str, float]:
        return {name: sum(items) / max(len(items), 1) for name, items in values.items()}

    reference_norms = mean(norm_reference)
    test_norms = mean(norm_test)
    return {
        "relative": mean(relative),
        "absolute": mean(absolute),
        "norm_reference": reference_norms,
        "norm_test": test_norms,
        "norm_drift": {
            name: abs(test_norms[name] - reference_norms[name]) / max(reference_norms[name], 1e-12)
            for name in STATE_NAMES
        },
    }


@torch.inference_mode()
def measure_drift(
    reference_model: GlassMindLM,
    test_model: GlassMindLM,
    tokens: Tensor,
    checkpoints: Sequence[int],
    *,
    capabilities: DeviceCapabilities | None = None,
    segment: int = 512,
) -> list[DriftPoint]:
    """Läuft beide Modelle einmal durch und misst an den angegebenen Längen.

    Die Sequenz wird in Segmente zerlegt und der Zustand weitergereicht. Damit
    kostet die Messung aller Längen so viel wie ein einziger Durchlauf der
    längsten Sequenz, nicht die Summe aller Längen.
    """
    reference_model.eval()
    test_model.eval()
    marks = sorted({int(value) for value in checkpoints if 0 < int(value) <= tokens.shape[1]})
    if not marks:
        raise ValueError("Es liegt keine gültige Prüflänge innerhalb der Sequenz")
    points: list[DriftPoint] = []
    reference_state: ModelState | None = None
    test_state: ModelState | None = None
    position = 0
    pending = list(marks)
    while pending:
        target = pending[0]
        while position < target:
            width = min(segment, target - position)
            chunk = tokens[:, position : position + width]
            reference_logits, reference_state = reference_model(chunk, reference_state)
            test_logits, test_state = test_model(chunk, test_state)
            position += width
        assert reference_state is not None and test_state is not None
        comparison = _compare_states(reference_state, test_state)
        a = reference_logits[:, -1].float()
        b = test_logits[:, -1].float()
        point = DriftPoint(
            length=target,
            state_relative_rms=comparison["relative"],
            state_absolute_rms=comparison["absolute"],
            state_norm_reference=comparison["norm_reference"],
            state_norm_test=comparison["norm_test"],
            state_norm_drift=comparison["norm_drift"],
            logit_relative_rms=_relative_rms(b, a),
            logit_max_absolute=float((b - a).abs().max()),
            prediction_change_rate=float((a.argmax(-1) != b.argmax(-1)).float().mean()),
            has_nan=bool(torch.isnan(b).any()) or any(
                bool(torch.isnan(getattr(block, name)).any())
                for block in test_state.blocks
                for name in STATE_NAMES
            ),
            has_inf=bool(torch.isinf(b).any()) or any(
                bool(torch.isinf(getattr(block, name)).any())
                for block in test_state.blocks
                for name in STATE_NAMES
            ),
        )
        points.append(point)
        pending.pop(0)
    if capabilities is not None:
        synchronize(capabilities)
    return points


def drift_summary(points: Sequence[DriftPoint]) -> dict[str, Any]:
    """Verdichtet die Messreihe: wo beginnt die Drift wirklich?"""
    if not points:
        return {}
    final = points[-1]
    onset: dict[str, int | None] = {}
    for name in STATE_NAMES:
        # Erste Länge, ab der die relative Abweichung ein Prozent übersteigt.
        onset[name] = next(
            (point.length for point in points if point.state_relative_rms.get(name, 0.0) > 0.01),
            None,
        )
    logit_onset = next((point.length for point in points if point.logit_relative_rms > 0.01), None)
    prediction_onset = next(
        (point.length for point in points if point.prediction_change_rate > 0.0), None
    )
    return {
        "max_length": final.length,
        "final_state_relative_rms": final.state_relative_rms,
        "final_logit_relative_rms": final.logit_relative_rms,
        "final_prediction_change_rate": final.prediction_change_rate,
        "state_drift_onset_length": onset,
        "logit_drift_onset_length": logit_onset,
        "prediction_change_onset_length": prediction_onset,
        "any_nan": any(point.has_nan for point in points),
        "any_inf": any(point.has_inf for point in points),
    }


def format_drift_table(points: Sequence[DriftPoint]) -> str:
    header = (
        f"{'Länge':>7s} {'fast':>10s} {'context':>10s} {'semantic':>10s} "
        f"{'Logit':>10s} {'Pred.':>7s} {'NaN/Inf':>8s}"
    )
    lines = [header, "-" * len(header)]
    for point in points:
        lines.append(
            f"{point.length:7d} "
            f"{point.state_relative_rms.get('fast', 0.0):10.3e} "
            f"{point.state_relative_rms.get('context', 0.0):10.3e} "
            f"{point.state_relative_rms.get('semantic', 0.0):10.3e} "
            f"{point.logit_relative_rms:10.3e} "
            f"{point.prediction_change_rate:6.1%} "
            f"{('ja' if point.has_nan or point.has_inf else 'nein'):>8s}"
        )
    return "\n".join(lines)
