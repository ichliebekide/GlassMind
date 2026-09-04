"""Hardware-Microbenchmark mit GlassMind-typischen Formen.

Große synthetische Matrixmultiplikationen sagen über diesen Kern wenig aus: Er
besteht überwiegend aus kleinen, sequenziell abhängigen Operationen. Deshalb
misst dieses Modul genau die Formen, die im Tokenpfad tatsächlich auftreten.

Aus den Messwerten leitet ``recommend`` eine Empfehlung ab. Sie ist eine
Beobachtung über diese Hardware, keine allgemeine Aussage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import statistics
import time
from typing import Any, Callable

import torch
from torch import nn

from glassmind.precision.policy import FLOAT_DTYPES, PrecisionPolicy
from glassmind.precision.quantization import (
    QuantizedLinear,
    dequantize_weight,
    fp8_compute_supported,
    quantize_weight,
)
from glassmind.utils.device import DeviceCapabilities, synchronize


@dataclass
class BenchmarkResult:
    name: str
    group: str
    dtype: str
    microseconds: float
    repetitions: int
    note: str = ""
    #: Streuung über die Messreihen. Ohne sie lässt sich ein echter Vorsprung
    #: nicht von Messrauschen unterscheiden.
    minimum: float = 0.0
    maximum: float = 0.0

    @property
    def spread_percent(self) -> float:
        return 100.0 * (self.maximum - self.minimum) / self.microseconds if self.microseconds else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "dtype": self.dtype,
            "microseconds": self.microseconds,
            "microseconds_min": self.minimum,
            "microseconds_max": self.maximum,
            "spread_percent": self.spread_percent,
            "repetitions": self.repetitions,
            "note": self.note,
        }


@dataclass
class MicrobenchmarkReport:
    device: dict[str, Any]
    results: list[BenchmarkResult] = field(default_factory=list)
    recommendation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "results": [result.to_dict() for result in self.results],
            "recommendation": self.recommendation,
        }

    def best(self, group: str) -> BenchmarkResult | None:
        candidates = [item for item in self.results if item.group == group]
        return min(candidates, key=lambda item: item.microseconds) if candidates else None

    def lookup(self, group: str, dtype: str) -> BenchmarkResult | None:
        return next(
            (item for item in self.results if item.group == group and item.dtype == dtype), None
        )

    def format_table(self) -> str:
        lines = [f"Microbenchmark auf {self.device.get('name', 'unbekannt')}", ""]
        width = max((len(item.name) for item in self.results), default=10)
        current = None
        for item in self.results:
            if item.group != current:
                current = item.group
                lines.append(f"  [{current}]")
            note = f"   {item.note}" if item.note else ""
            lines.append(
                f"    {item.name:<{width}}  {item.dtype:<11s} {item.microseconds:9.2f} µs "
                f"(±{item.spread_percent:4.1f} %){note}"
            )
        if self.recommendation:
            lines.extend(["", "  Empfohlen auf dieser Hardware:"])
            for key, value in self.recommendation.items():
                if key != "notes":
                    lines.append(f"    {key:<14} = {value}")
            for note in self.recommendation.get("notes", ()):
                lines.append(f"    Hinweis: {note}")
        return "\n".join(lines)


def _time(
    fn: Callable[[], Any],
    capabilities: DeviceCapabilities,
    repetitions: int,
    warmup: int = 16,
    series: int = 7,
) -> tuple[float, float, float]:
    """(Median, Minimum, Maximum) in Mikrosekunden je Aufruf.

    Die Messung schließt den Python-Dispatch bewusst ein: GlassMind ist bei
    diesen Formen dispatch-gebunden, eine reine Kernelzeit wäre nicht das,
    was der Tokenpfad tatsächlich kostet.
    """
    for _ in range(warmup):
        fn()
    synchronize(capabilities)
    samples: list[float] = []
    for _ in range(series):
        started = time.perf_counter()
        for _ in range(repetitions):
            fn()
        synchronize(capabilities)
        samples.append((time.perf_counter() - started) * 1e6 / repetitions)
    return statistics.median(samples), min(samples), max(samples)


def _record(
    report: "MicrobenchmarkReport",
    name: str,
    group: str,
    dtype: str,
    timing: tuple[float, float, float],
    repetitions: int,
    note: str = "",
) -> None:
    median, minimum, maximum = timing
    report.results.append(
        BenchmarkResult(
            name=name,
            group=group,
            dtype=dtype,
            microseconds=median,
            repetitions=repetitions,
            note=note,
            minimum=minimum,
            maximum=maximum,
        )
    )


def _supported_dtypes(capabilities: DeviceCapabilities) -> list[str]:
    names = ["float32", "bfloat16", "float16"]
    if capabilities.backend == "cpu":
        # Float16 ist auf CPU zwar darstellbar, aber ohne echte Rechenwerke; der
        # Vergleich wäre irreführend.
        names.remove("float16")
    return names


def run_microbenchmark(
    capabilities: DeviceCapabilities,
    *,
    d_model: int = 64,
    batch: int = 1,
    sequence: int = 128,
    repetitions: int = 200,
    large: bool = True,
) -> MicrobenchmarkReport:
    device = capabilities.torch_device
    report = MicrobenchmarkReport(device=capabilities.to_dict())
    dtypes = _supported_dtypes(capabilities)

    # --- Lineare Abbildungen in den Formen des Tokenpfads ---------------
    shapes = [
        ("linear_klein_state", batch, d_model, d_model + max(2, int(d_model ** 0.5)) + 1),
        ("linear_klein_integrator", batch, 2 * d_model + d_model, d_model),
        ("linear_sequenz_input", batch * sequence, d_model, 5 * d_model),
    ]
    if large:
        shapes.append(("linear_gross", batch * sequence, 4 * d_model, 16 * d_model))
    for label, rows, in_features, out_features in shapes:
        for name in dtypes:
            dtype = FLOAT_DTYPES[name]
            x = torch.randn(rows, in_features, device=device, dtype=dtype)
            w = torch.randn(out_features, in_features, device=device, dtype=dtype)
            _record(
                report,
                f"{label} [{rows}x{in_features}]→{out_features}",
                label,
                name,
                _time(lambda: torch.nn.functional.linear(x, w), capabilities, repetitions),
                repetitions,
            )

    # --- Zustandsaktualisierung: genau die Kette aus dem Tokenpfad -------
    for name in dtypes:
        dtype = FLOAT_DTYPES[name]
        state = torch.randn(batch, d_model, device=device, dtype=dtype)
        value = torch.randn(batch, d_model, device=device, dtype=dtype)
        gate = torch.rand(batch, d_model, device=device, dtype=dtype)
        _record(
            report,
            f"state_update lerp(tanh(v+s), g) [{batch}x{d_model}]",
            "state_update",
            name,
            _time(lambda: torch.lerp(state, torch.tanh(value + state), gate), capabilities, repetitions),
            repetitions,
        )

    # --- Elementweise Gates ---------------------------------------------
    for name in dtypes:
        dtype = FLOAT_DTYPES[name]
        a = torch.randn(batch, d_model, device=device, dtype=dtype)
        b = torch.randn(batch, d_model, device=device, dtype=dtype)
        _record(
            report,
            f"gate silu(a)*sigmoid(b) [{batch}x{d_model}]",
            "elementwise_gate",
            name,
            _time(lambda: torch.nn.functional.silu(a) * torch.sigmoid(b), capabilities, repetitions),
            repetitions,
        )

    # --- Präzisionswechsel ----------------------------------------------
    source = torch.randn(batch * sequence, d_model, device=device, dtype=torch.float32)
    for name in dtypes:
        if name == "float32":
            continue
        dtype = FLOAT_DTYPES[name]
        _record(
            report,
            f"cast float32→{name} [{batch*sequence}x{d_model}]",
            "cast",
            name,
            _time(lambda: source.to(dtype), capabilities, repetitions),
            repetitions,
        )

    # --- Dequantisierung --------------------------------------------------
    dense = torch.randn(5 * d_model, d_model, device=device, dtype=torch.float32)
    for scheme in ("int8", "int4"):
        packed, scales = quantize_weight(dense, scheme, 0)
        packed, scales = packed.to(device), scales.to(device)
        shape = dense.shape
        for name in dtypes:
            dtype = FLOAT_DTYPES[name]
            _record(
                report,
                f"dequantisieren {scheme} [{shape[0]}x{shape[1]}]→{name}",
                f"dequantize_{scheme}",
                name,
                _time(
                    lambda: dequantize_weight(packed, scales, scheme, shape, dtype),
                    capabilities,
                    max(repetitions // 4, 20),
                ),
                max(repetitions // 4, 20),
            )

    # --- INT8-Linear im Ganzen (Dequantisierung plus Matmul) --------------
    reference = nn.Linear(d_model, 5 * d_model, bias=False).to(device)
    x_small = torch.randn(batch, d_model, device=device, dtype=torch.float32)
    for scheme in ("none", "int8", "int4"):
        if scheme == "none":
            module: nn.Module = reference
            note = "dichte FP32-Referenz"
        else:
            module = QuantizedLinear.from_linear(reference, scheme, 0, cache=False).to(device)
            note = "ohne Dequantisierungs-Cache"
        _record(
            report,
            f"weight_linear {scheme} [{batch}x{d_model}]→{5*d_model}",
            "weight_linear",
            scheme,
            _time(lambda: module(x_small), capabilities, max(repetitions // 2, 40)),
            max(repetitions // 2, 40),
            note,
        )
    cached = QuantizedLinear.from_linear(reference, "int8", 0, cache=True).to(device)
    cached(x_small)
    _record(
        report,
        f"weight_linear int8 [{batch}x{d_model}]→{5*d_model}",
        "weight_linear",
        "int8_cached",
        _time(lambda: cached(x_small), capabilities, max(repetitions // 2, 40)),
        max(repetitions // 2, 40),
        "mit Dequantisierungs-Cache (Speichervorteil entfällt zur Laufzeit)",
    )

    report.recommendation = recommend(report, capabilities)
    return report


def recommend(report: MicrobenchmarkReport, capabilities: DeviceCapabilities) -> dict[str, Any]:
    """Leitet compute-/State-dtypes aus den Messwerten ab.

    Es gibt hier keine Vorannahme, welches Format gewinnt. Ein Format wird nur
    dann empfohlen, wenn sein Vorsprung die gemessene Streuung deutlich
    übersteigt – sonst bleibt das genauere float32 stehen.
    """
    notes: list[str] = []
    token_groups = ("linear_klein_state", "linear_klein_integrator", "state_update", "elementwise_gate")
    totals: dict[str, float] = {}
    spreads: dict[str, float] = {}
    for group in token_groups:
        for item in report.results:
            if item.group == group:
                totals[item.dtype] = totals.get(item.dtype, 0.0) + item.microseconds
                spreads[item.dtype] = spreads.get(item.dtype, 0.0) + (item.maximum - item.minimum)
    if not totals:  # pragma: no cover - nur bei leerem Bericht
        return {"compute": "float32", "states": "float32", "notes": ["Keine Messwerte"]}

    reference = totals.get("float32")
    candidate = min(totals, key=lambda key: totals[key])
    compute = "float32"
    if reference:
        gain = 100.0 * (1.0 - totals[candidate] / reference)
        # Die Streuung beider Kandidaten ist die Untergrenze dessen, was ein
        # Vorsprung überhaupt bedeuten kann.
        noise = 100.0 * (spreads.get(candidate, 0.0) + spreads.get("float32", 0.0)) / reference
        threshold = max(5.0, noise)
        if candidate != "float32" and gain >= threshold:
            compute = candidate
            notes.append(
                f"{candidate} ist im Tokenpfad {gain:.1f} % schneller als float32 "
                f"(Messrauschen {noise:.1f} %)."
            )
        else:
            notes.append(
                f"Kein reduziertes Format schlägt float32 im Tokenpfad deutlich: bester Kandidat "
                f"{candidate} mit {gain:.1f} % bei {noise:.1f} % Messrauschen. "
                "Die Formen sind so klein, dass der Dispatch-Overhead dominiert."
            )

    dense = report.lookup("weight_linear", "none")
    for label, key in (("ohne Cache", "int8"), ("mit Cache", "int8_cached"), ("ohne Cache", "int4")):
        item = report.lookup("weight_linear", key)
        if dense and item:
            notes.append(
                f"{key} {label} kostet das {item.microseconds / dense.microseconds:.2f}-fache "
                "eines dichten Linear."
            )
    notes.append(
        "Weight-Only-Quantisierung dequantisiert vor dem Matmul zurück; ihr Gewinn liegt im "
        "Speicherbedarf, nicht automatisch in der Rechenzeit."
    )

    supported, reason = fp8_compute_supported(capabilities.torch_device)
    notes.append(("FP8-Compute verfügbar: " if supported else "FP8-Compute nicht verfügbar: ") + reason)

    state_dtype = "float32"
    state_reference = report.lookup("state_update", "float32")
    state_best = report.best("state_update")
    if state_reference and state_best and state_best.dtype != "float32":
        state_gain = 100.0 * (1.0 - state_best.microseconds / state_reference.microseconds)
        state_noise = 100.0 * (
            (state_best.maximum - state_best.minimum)
            + (state_reference.maximum - state_reference.minimum)
        ) / state_reference.microseconds
        if state_gain >= max(5.0, state_noise):
            state_dtype = state_best.dtype
            notes.append(
                f"State-Update ist in {state_best.dtype} {state_gain:.1f} % schneller "
                f"(Rauschen {state_noise:.1f} %)."
            )
        else:
            notes.append(
                f"State-Update gewinnt in {state_best.dtype} nur {state_gain:.1f} % bei "
                f"{state_noise:.1f} % Rauschen; die Zustände bleiben float32."
            )
    return {"compute": compute, "states": state_dtype, "notes": notes}


def auto_policy(
    capabilities: DeviceCapabilities, report: MicrobenchmarkReport | None = None, **kwargs: Any
) -> tuple[PrecisionPolicy, MicrobenchmarkReport]:
    """Wählt eine Policy anhand gemessener Werte, nicht anhand von Annahmen.

    ``semantic_state`` bleibt grundsätzlich float32: Er akkumuliert über die
    gesamte Sequenz und ist der einzige Zustand, dessen Drift sich nicht in
    wenigen Schritten auswäscht. Ob das gerechtfertigt ist, prüft die
    Driftmessung in ``glassmind.precision.drift``.
    """
    report = report or run_microbenchmark(capabilities, **kwargs)
    recommendation = report.recommendation
    compute = str(recommendation.get("compute", "float32"))
    states = str(recommendation.get("states", "float32"))
    policy = PrecisionPolicy(
        profile="auto",
        compute=compute,
        activations=compute,
        fast_state=states,
        context_state=states if states != "float32" else "float32",
        semantic_state="float32",
        selection_notes=tuple(recommendation.get("notes", ())),
    )
    return policy, report
