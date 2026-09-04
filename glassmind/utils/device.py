from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import json
from typing import Any, ContextManager

import torch


@dataclass(frozen=True)
class DeviceCapabilities:
    backend: str
    device: str
    name: str
    precision: str
    amp: bool
    compile_available: bool
    total_memory_bytes: int | None
    pytorch_version: str
    cuda_version: str | None
    rocm_version: str | None
    optimizations: tuple[str, ...] = ()

    @property
    def torch_device(self) -> torch.device:
        return torch.device(self.device)

    @property
    def dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[self.precision]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _xpu_available() -> bool:
    return bool(hasattr(torch, "xpu") and torch.xpu.is_available())


def _mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def available_backends() -> dict[str, bool]:
    cuda_available = torch.cuda.is_available()
    return {
        "cuda": cuda_available and torch.version.hip is None,
        "rocm": cuda_available and torch.version.hip is not None,
        "xpu": _xpu_available(),
        "mps": _mps_available(),
        "cpu": True,
    }


def detect_device(requested: str = "auto", precision: str = "auto") -> DeviceCapabilities:
    available = available_backends()
    requested = requested.lower()
    if requested == "auto":
        backend = next(name for name in ("cuda", "rocm", "xpu", "mps", "cpu") if available[name])
    else:
        if requested not in available:
            raise ValueError(f"Unbekanntes Backend: {requested}")
        if not available[requested]:
            raise RuntimeError(f"Backend {requested} ist im installierten PyTorch-Build nicht verfügbar")
        backend = requested

    device_name, total_memory = _device_details(backend)
    selected_precision, amp = _select_precision(backend, precision)
    device = "cuda" if backend in {"cuda", "rocm"} else backend
    optimizations: list[str] = []
    if amp:
        optimizations.append("AMP")
    return DeviceCapabilities(
        backend=backend,
        device=device,
        name=device_name,
        precision=selected_precision,
        amp=amp,
        compile_available=hasattr(torch, "compile"),
        total_memory_bytes=total_memory,
        pytorch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        rocm_version=torch.version.hip,
        optimizations=tuple(optimizations),
    )


def _device_details(backend: str) -> tuple[str, int | None]:
    if backend in {"cuda", "rocm"}:
        properties = torch.cuda.get_device_properties(0)
        return properties.name, int(properties.total_memory)
    if backend == "xpu":
        properties = torch.xpu.get_device_properties(0)
        return properties.name, int(getattr(properties, "total_memory", 0)) or None
    if backend == "mps":
        return "Apple Metal Performance Shaders", None
    return "CPU", None


def _select_precision(backend: str, requested: str) -> tuple[str, bool]:
    requested = requested.lower()
    allowed = {"auto", "float32", "float16", "bfloat16"}
    if requested not in allowed:
        raise ValueError(f"Unbekannte Precision: {requested}")
    if requested != "auto":
        if backend == "cpu" and requested == "float16":
            raise RuntimeError("float16 ist für den portablen CPU-Pfad nicht freigegeben")
        amp_requested = requested != "float32"
        if amp_requested and not _autocast_available(backend):
            raise RuntimeError(f"AMP ist für Backend {backend} in diesem PyTorch-Build nicht verfügbar")
        return requested, amp_requested
    if not _autocast_available(backend):
        return "float32", False
    if backend == "cuda":
        return ("bfloat16" if torch.cuda.is_bf16_supported() else "float16"), True
    if backend == "rocm":
        bf16 = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
        return ("bfloat16" if bf16 else "float16"), True
    if backend == "xpu":
        return "bfloat16", True
    if backend == "mps":
        return "float16", True
    return "float32", False


def _autocast_available(backend: str) -> bool:
    if backend == "cpu":
        return True
    device_type = "cuda" if backend in {"cuda", "rocm"} else backend
    checker = getattr(getattr(torch.amp, "autocast_mode", object()), "is_autocast_available", None)
    return bool(checker(device_type)) if checker is not None else backend in {"cuda", "rocm"}


def autocast_context(capabilities: DeviceCapabilities) -> ContextManager[Any]:
    if not capabilities.amp:
        return nullcontext()
    device_type = "cuda" if capabilities.backend in {"cuda", "rocm"} else capabilities.backend
    return torch.autocast(device_type=device_type, dtype=capabilities.dtype)


def synchronize(capabilities: DeviceCapabilities) -> None:
    if capabilities.backend in {"cuda", "rocm"}:
        torch.cuda.synchronize()
    elif capabilities.backend == "xpu":
        torch.xpu.synchronize()
    elif capabilities.backend == "mps" and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def reset_peak_memory(capabilities: DeviceCapabilities) -> None:
    if capabilities.backend in {"cuda", "rocm"}:
        torch.cuda.reset_peak_memory_stats()
    elif capabilities.backend == "xpu" and hasattr(torch.xpu, "reset_peak_memory_stats"):
        torch.xpu.reset_peak_memory_stats()


def peak_memory_bytes(capabilities: DeviceCapabilities) -> int | None:
    if capabilities.backend in {"cuda", "rocm"}:
        return int(torch.cuda.max_memory_allocated())
    if capabilities.backend == "xpu" and hasattr(torch.xpu, "max_memory_allocated"):
        return int(torch.xpu.max_memory_allocated())
    # MPS stellt derzeit keinen echten Peak-Zähler bereit; ein aktueller Wert wäre irreführend.
    return None


def format_device_report(capabilities: DeviceCapabilities) -> str:
    available = available_backends()
    gib = capabilities.total_memory_bytes / 2**30 if capabilities.total_memory_bytes else None
    lines = [
        "GlassMind-Gerätediagnose",
        f"  PyTorch:             {capabilities.pytorch_version}",
        f"  Verfügbare Backends: {', '.join(name for name, found in available.items() if found)}",
        f"  Gewähltes Backend:   {capabilities.backend}",
        f"  Gerät:               {capabilities.name}",
        f"  Precision:           {capabilities.precision}",
        f"  AMP:                 {'aktiv' if capabilities.amp else 'inaktiv'}",
        f"  torch.compile:       {'verfügbar' if capabilities.compile_available else 'nicht verfügbar'}",
        f"  Aktive Optimierungen:{' ' + ', '.join(capabilities.optimizations) if capabilities.optimizations else ' keine'}",
    ]
    if gib is not None:
        lines.append(f"  Gerätespeicher:      {gib:.2f} GiB")
    if capabilities.cuda_version:
        lines.append(f"  CUDA-Runtime:        {capabilities.cuda_version}")
    if capabilities.rocm_version:
        lines.append(f"  ROCm-Runtime:        {capabilities.rocm_version}")
    lines.append("  Referenzpfad:        portable PyTorch-Operationen")
    lines.extend(_precision_lines(capabilities))
    return "\n".join(lines)


def _precision_lines(capabilities: DeviceCapabilities) -> list[str]:
    """Meldet, welche Zahlenformate hier wirklich tragen – ohne Beschönigung."""
    # Der Import steht bewusst hier: ``glassmind.precision`` benutzt dieses
    # Modul, ein Import auf Modulebene wäre ein Ringschluss.
    from glassmind.precision.quantization import (
        FP8_DTYPES,
        fp8_compute_supported,
        fp8_storage_supported,
    )

    device = capabilities.torch_device
    lines = ["  Zahlenformate:"]
    lines.append("    float32:            immer verfügbar")
    for name, dtype in (("bfloat16", torch.bfloat16), ("float16", torch.float16)):
        if capabilities.backend == "cpu" and name == "float16":
            lines.append(f"    {name + ':':<19s} darstellbar, aber ohne CPU-Rechenwerke – nicht freigegeben")
            continue
        try:
            torch.zeros(2, device=device, dtype=dtype).sum()
            lines.append(f"    {name + ':':<19s} verfügbar")
        except Exception as exc:  # pragma: no cover - backendabhängig
            lines.append(f"    {name + ':':<19s} nicht verfügbar ({type(exc).__name__})")
    lines.append(
        "    INT8/INT4 Gewichte: portabel (Weight-Only, Dequantisierung vor dem Matmul)"
    )
    storage = "verfügbar" if fp8_storage_supported() else "nicht in diesem PyTorch-Build"
    lines.append(f"    FP8 als Speicher:   {storage}"
                 + (f" ({', '.join(FP8_DTYPES)})" if FP8_DTYPES else ""))
    supported, reason = fp8_compute_supported(device)
    lines.append(f"    FP8-Rechenwerke:    {'verfügbar' if supported else 'nicht verfügbar'} – {reason}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Erkennt GlassMind-Geräte und Fähigkeiten")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    capabilities = detect_device(args.device, args.precision)
    print(json.dumps(capabilities.to_dict(), indent=2) if args.json else format_device_report(capabilities))


if __name__ == "__main__":
    main()
