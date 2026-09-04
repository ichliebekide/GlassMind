"""Precision- und Quantisierungsschicht von GlassMind (Milestone 2.6).

``glassmind.precision.reference`` wird hier bewusst nicht re-exportiert: Es
misst ein vollständiges Modell und hängt deshalb an ``glassmind.model`` und
``glassmind.analysis``. Ein Import auf Paketebene ergäbe einen Ringschluss.
Es wird direkt importiert::

    from glassmind.precision.reference import collect_reference
"""

from glassmind.precision.compare import compare_telemetry, format_telemetry_comparison
from glassmind.precision.drift import (
    DriftPoint,
    drift_summary,
    format_drift_table,
    measure_drift,
)
from glassmind.precision.microbench import (
    MicrobenchmarkReport,
    auto_policy,
    run_microbenchmark,
)
from glassmind.precision.apply import (
    apply_precision,
    component_modules,
    precision_report,
    state_dtype_report,
)
from glassmind.precision.policy import (
    FLOAT_DTYPES,
    INHERIT,
    PROFILE_NAMES,
    STATIC_PROFILES,
    WEIGHT_COMPONENTS,
    WEIGHT_SCHEMES,
    PrecisionPolicy,
    balanced_profile,
    experimental_profile,
    fast_profile,
    resolve_dtype,
    safe_profile,
)
from glassmind.precision.quantization import (
    QuantizationUnsupported,
    QuantizedEmbedding,
    QuantizedLinear,
    dequantize_weight,
    fake_quantize,
    fp8_compute_supported,
    fp8_storage_supported,
    parameter_storage_bytes,
    quantization_report,
    quantize_weight,
)

__all__ = [
    "DriftPoint",
    "FLOAT_DTYPES",
    "MicrobenchmarkReport",
    "INHERIT",
    "PROFILE_NAMES",
    "STATIC_PROFILES",
    "WEIGHT_COMPONENTS",
    "WEIGHT_SCHEMES",
    "PrecisionPolicy",
    "QuantizationUnsupported",
    "QuantizedEmbedding",
    "QuantizedLinear",
    "apply_precision",
    "auto_policy",
    "balanced_profile",
    "compare_telemetry",
    "component_modules",
    "dequantize_weight",
    "drift_summary",
    "experimental_profile",
    "fake_quantize",
    "fast_profile",
    "format_drift_table",
    "format_telemetry_comparison",
    "fp8_compute_supported",
    "fp8_storage_supported",
    "measure_drift",
    "parameter_storage_bytes",
    "precision_report",
    "quantization_report",
    "quantize_weight",
    "resolve_dtype",
    "run_microbenchmark",
    "safe_profile",
    "state_dtype_report",
]
