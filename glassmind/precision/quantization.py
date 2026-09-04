"""Portable Weight-Only-Quantisierung für GlassMind.

Enthalten sind INT8, INT4 (in ``uint8`` gepackt), reine Gleitkomma-
Speicherformate (FP16/BF16) und die vorbereitete FP8-Schnittstelle.

Alle Pfade bestehen aus gewöhnlichen ATen-Operationen und laufen deshalb auf
jedem Backend, das PyTorch anbietet. Es gibt keine herstellerspezifischen
Kernel und keinen stillen Fallback: Ein nicht unterstütztes Format führt zu
einem Fehler, nicht zu einem stillschweigend falschen Ergebnis.

Wichtig für die Einordnung: Weight-Only-Quantisierung dequantisiert vor dem
Matmul zurück in den Rechendatentyp. Sie spart also **Speicher für die
Gewichte**, nicht zwangsläufig Rechenzeit. Ob sie schneller ist, misst
``scripts/precision_matrix.py``.
"""
from __future__ import annotations

from typing import Iterator

import torch
from torch import Tensor, nn

#: FP8-Typen existieren erst ab neueren PyTorch-Versionen.
FP8_DTYPES: dict[str, torch.dtype] = {
    name: dtype
    for name, dtype in (
        ("float8_e4m3", getattr(torch, "float8_e4m3fn", None)),
        ("float8_e5m2", getattr(torch, "float8_e5m2", None)),
    )
    if dtype is not None
}

STORAGE_FLOAT_DTYPES: dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


class QuantizationUnsupported(RuntimeError):
    """Das gewünschte Format wird von Build, Backend oder Hardware nicht getragen."""


# ----------------------------------------------------------------------
# Unterstützungsabfragen
# ----------------------------------------------------------------------

def fp8_storage_supported() -> bool:
    """Kennt dieser PyTorch-Build FP8 überhaupt als Datentyp?"""
    return bool(FP8_DTYPES)


def fp8_compute_supported(device: torch.device | str = "cpu") -> tuple[bool, str]:
    """Prüft echte FP8-*Rechenwerke*, nicht nur den Datentyp.

    Gibt ``(unterstützt, Begründung)`` zurück. Ein FP8-Speicherformat mit
    anschließender Dequantisierung ist ausdrücklich **kein** FP8-Compute und
    wird hier nicht als solches gemeldet.
    """
    if not fp8_storage_supported():
        return False, "Dieser PyTorch-Build kennt keine FP8-Datentypen"
    if not hasattr(torch, "_scaled_mm"):
        return False, "torch._scaled_mm ist nicht verfügbar"
    device = torch.device(device)
    if device.type != "cuda":
        return False, f"FP8-Matmul ist für Backend {device.type!r} nicht implementiert"
    if not torch.cuda.is_available():
        return False, "Kein CUDA/ROCm-Gerät verfügbar"
    major, minor = torch.cuda.get_device_capability(device.index or 0)
    if (major, minor) < (8, 9):
        return False, (
            f"Compute Capability {major}.{minor} besitzt keine FP8-Tensorkerne "
            "(benötigt 8.9 oder neuer)"
        )
    return True, f"Compute Capability {major}.{minor} unterstützt FP8-Matmul"


def require_scheme_supported(scheme: str, device: torch.device | str = "cpu") -> None:
    if scheme in ("none", "int8", "int4"):
        return
    if scheme in STORAGE_FLOAT_DTYPES:
        return
    if scheme in ("float8_e4m3", "float8_e5m2"):
        if scheme not in FP8_DTYPES:
            raise QuantizationUnsupported(
                f"{scheme} ist in PyTorch {torch.__version__} nicht vorhanden. "
                "GlassMind aktiviert FP8 nicht stillschweigend mit einem Ersatzformat."
            )
        return
    raise QuantizationUnsupported(f"Unbekanntes Gewichtsschema: {scheme}")


def storage_layout(
    scheme: str, out_features: int, in_features: int, group_size: int
) -> tuple[torch.Size, torch.dtype, int]:
    """Form, dtype und Gruppenzahl der Ablage – identisch zu ``quantize_weight``.

    Die Puffer werden damit schon im Konstruktor korrekt angelegt, sodass
    ``load_state_dict`` weder Form noch dtype anpassen muss.
    """
    require_scheme_supported(scheme)
    if scheme in STORAGE_FLOAT_DTYPES:
        return torch.Size((out_features, in_features)), STORAGE_FLOAT_DTYPES[scheme], 1
    if scheme in FP8_DTYPES:
        return torch.Size((out_features, in_features)), FP8_DTYPES[scheme], 1
    groups = 1 if group_size <= 0 or group_size >= in_features else in_features // group_size
    if groups > 1 and in_features % group_size:
        raise ValueError(
            f"weight_group_size={group_size} teilt die Eingangsbreite {in_features} nicht"
        )
    if scheme == "int8":
        return torch.Size((out_features, in_features)), torch.int8, groups
    if in_features % 2:
        raise ValueError("INT4 benötigt eine gerade Eingangsbreite")
    return torch.Size((out_features, in_features // 2)), torch.uint8, groups


# ----------------------------------------------------------------------
# Quantisierung und Rücktransformation
# ----------------------------------------------------------------------

def _grouped(weight: Tensor, group_size: int) -> tuple[Tensor, int]:
    """Formt [out, in] zu [out, gruppen, gruppengröße]."""
    out_features, in_features = weight.shape
    if group_size <= 0 or group_size >= in_features:
        return weight.reshape(out_features, 1, in_features), 1
    if in_features % group_size:
        raise ValueError(
            f"weight_group_size={group_size} teilt die Eingangsbreite {in_features} nicht"
        )
    groups = in_features // group_size
    return weight.reshape(out_features, groups, group_size), groups


def quantize_weight(weight: Tensor, scheme: str, group_size: int) -> tuple[Tensor, Tensor]:
    """Erzeugt (gepackte Werte, Skalen) für ein Gewicht [out, in].

    INT8 und INT4 arbeiten symmetrisch: der Nullpunkt bleibt exakt null, damit
    ein Nullgewicht auch nach der Rücktransformation null ist.
    """
    require_scheme_supported(scheme)
    source = weight.detach().float()
    if scheme in STORAGE_FLOAT_DTYPES:
        return source.to(STORAGE_FLOAT_DTYPES[scheme]), torch.ones(1, dtype=torch.float32)
    if scheme in FP8_DTYPES:
        # FP8 hat einen sehr kleinen Wertebereich; eine Kanalskala hält die
        # Gewichte im darstellbaren Bereich.
        limit = torch.finfo(FP8_DTYPES[scheme]).max
        peak = source.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
        scale = peak / limit
        # Skalen liegen für jedes Schema als [out, gruppen] vor; FP8 nutzt eine
        # Gruppe je Ausgangskanal.
        return (source / scale).to(FP8_DTYPES[scheme]), scale.contiguous()

    blocks, groups = _grouped(source, group_size)
    levels = 127.0 if scheme == "int8" else 7.0
    peak = blocks.abs().amax(dim=2, keepdim=True).clamp_min(1e-12)
    scale = peak / levels
    quantized = torch.round(blocks / scale).clamp_(-levels - 1.0, levels)
    scales = scale.reshape(source.shape[0], groups).contiguous()
    if scheme == "int8":
        return quantized.reshape(source.shape).to(torch.int8), scales
    # INT4: zwei Werte je Byte, Vorzeichen über einen Offset von 8.
    values = (quantized.reshape(source.shape).to(torch.int16) + 8).clamp_(0, 15).to(torch.uint8)
    if values.shape[1] % 2:
        raise ValueError("INT4 benötigt eine gerade Eingangsbreite")
    packed = values[:, 0::2] | (values[:, 1::2] << 4)
    return packed.contiguous(), scales


def dequantize_weight(
    packed: Tensor, scales: Tensor, scheme: str, shape: torch.Size, dtype: torch.dtype
) -> Tensor:
    """Baut das Gewicht [out, in] im gewünschten Rechendatentyp zurück."""
    if scheme in STORAGE_FLOAT_DTYPES or scheme in FP8_DTYPES:
        restored = packed.to(torch.float32)
        if scheme in FP8_DTYPES:
            restored = restored * scales
        return restored.to(dtype)
    out_features, in_features = shape
    if scheme == "int8":
        values = packed.to(torch.float32)
    else:
        low = (packed & 0x0F).to(torch.int16) - 8
        high = ((packed >> 4) & 0x0F).to(torch.int16) - 8
        values = torch.stack((low, high), dim=2).reshape(out_features, in_features).to(torch.float32)
    groups = scales.shape[1]
    if groups == 1:
        restored = values * scales
    else:
        restored = (values.reshape(out_features, groups, in_features // groups) * scales.unsqueeze(2))
        restored = restored.reshape(out_features, in_features)
    return restored.to(dtype)


def fake_quantize(weight: Tensor, scheme: str, group_size: int = 0) -> Tensor:
    """Quantisiert und dequantisiert differenzierbar (Straight-Through).

    Grundlage für späteres Quantization Aware Training. In diesem Milestone
    wird die Funktion nur getestet, nicht im Training verwendet.
    """
    if scheme == "none":
        return weight
    packed, scales = quantize_weight(weight, scheme, group_size)
    restored = dequantize_weight(packed, scales, scheme, weight.shape, weight.dtype)
    return weight + (restored - weight).detach()


# ----------------------------------------------------------------------
# Module
# ----------------------------------------------------------------------

class QuantizedLinear(nn.Module):
    """Weight-Only-quantisierter Ersatz für ``nn.Linear``.

    Die Bias-Werte bleiben in voller Präzision: Sie sind winzig und ihre
    Quantisierung brächte weder Speicher- noch Rechenvorteil.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        scheme: str,
        group_size: int = 0,
        bias: bool = True,
        cache: bool = True,
    ) -> None:
        super().__init__()
        require_scheme_supported(scheme)
        self.in_features = in_features
        self.out_features = out_features
        self.scheme = scheme
        self.group_size = group_size
        self.cache_enabled = cache
        shape, storage_dtype, groups = storage_layout(scheme, out_features, in_features, group_size)
        self.register_buffer("packed_weight", torch.zeros(shape, dtype=storage_dtype), persistent=True)
        self.register_buffer("weight_scales", torch.ones(out_features, groups), persistent=True)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._cached: Tensor | None = None

    @property
    def weight_shape(self) -> torch.Size:
        return torch.Size((self.out_features, self.in_features))

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, scheme: str, group_size: int = 0, cache: bool = True
    ) -> "QuantizedLinear":
        module = cls(
            linear.in_features,
            linear.out_features,
            scheme=scheme,
            group_size=group_size,
            bias=linear.bias is not None,
            cache=cache,
        )
        packed, scales = quantize_weight(linear.weight, scheme, group_size)
        module.packed_weight.copy_(packed)
        module.weight_scales.copy_(scales)
        if linear.bias is not None:
            with torch.no_grad():
                module.bias.copy_(linear.bias.detach())
        return module

    def invalidate_cache(self) -> None:
        self._cached = None

    def dequantized_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        target = dtype or torch.get_default_dtype()
        cached = self._cached
        if cached is not None and cached.dtype == target and cached.device == self.packed_weight.device:
            return cached
        restored = dequantize_weight(
            self.packed_weight, self.weight_scales, self.scheme, self.weight_shape, target
        )
        if self.cache_enabled:
            self._cached = restored
        return restored

    def forward(self, x: Tensor) -> Tensor:
        return torch.nn.functional.linear(x, self.dequantized_weight(x.dtype), self.bias)

    def stored_bytes(self) -> int:
        total = self.packed_weight.numel() * self.packed_weight.element_size()
        total += self.weight_scales.numel() * self.weight_scales.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scheme={self.scheme}, group_size={self.group_size}"
        )


class QuantizedEmbedding(nn.Module):
    """Weight-Only-quantisierte Nachschlagetabelle."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        scheme: str,
        group_size: int = 0,
        cache: bool = True,
    ) -> None:
        super().__init__()
        require_scheme_supported(scheme)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.scheme = scheme
        self.group_size = group_size
        self.cache_enabled = cache
        shape, storage_dtype, groups = storage_layout(scheme, num_embeddings, embedding_dim, group_size)
        self.register_buffer("packed_weight", torch.zeros(shape, dtype=storage_dtype), persistent=True)
        self.register_buffer("weight_scales", torch.ones(num_embeddings, groups), persistent=True)
        self._cached: Tensor | None = None

    @property
    def weight_shape(self) -> torch.Size:
        return torch.Size((self.num_embeddings, self.embedding_dim))

    @classmethod
    def from_embedding(
        cls, embedding: nn.Embedding, scheme: str, group_size: int = 0, cache: bool = True
    ) -> "QuantizedEmbedding":
        module = cls(
            embedding.num_embeddings,
            embedding.embedding_dim,
            scheme=scheme,
            group_size=group_size,
            cache=cache,
        )
        packed, scales = quantize_weight(embedding.weight, scheme, group_size)
        module.packed_weight.copy_(packed)
        module.weight_scales.copy_(scales)
        return module

    def invalidate_cache(self) -> None:
        self._cached = None

    def dequantized_weight(self, dtype: torch.dtype | None = None) -> Tensor:
        target = dtype or torch.get_default_dtype()
        cached = self._cached
        if cached is not None and cached.dtype == target and cached.device == self.packed_weight.device:
            return cached
        restored = dequantize_weight(
            self.packed_weight, self.weight_scales, self.scheme, self.weight_shape, target
        )
        if self.cache_enabled:
            self._cached = restored
        return restored

    @property
    def weight(self) -> Tensor:
        return self.dequantized_weight(torch.get_default_dtype())

    def forward(self, indices: Tensor) -> Tensor:
        return torch.nn.functional.embedding(indices, self.dequantized_weight())

    def stored_bytes(self) -> int:
        return (
            self.packed_weight.numel() * self.packed_weight.element_size()
            + self.weight_scales.numel() * self.weight_scales.element_size()
        )

    def extra_repr(self) -> str:
        return (
            f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, "
            f"scheme={self.scheme}, group_size={self.group_size}"
        )


QUANTIZED_MODULES = (QuantizedLinear, QuantizedEmbedding)


def linear_weight(module: nn.Module, dtype: torch.dtype) -> Tensor:
    """Gibt die Gewichtsmatrix eines Linear- oder QuantizedLinear-Moduls zurück.

    Der rekurrente Kern ruft das je Block und Sequenz auf – im Streaming also
    einmal pro Token. Der Normalfall ``nn.Linear`` mit passendem dtype geht
    deshalb über einen Typidentitätsvergleich und kostet keinen
    Vererbungsdurchlauf.
    """
    if type(module) is nn.Linear:
        weight = module.weight
        return weight if weight.dtype is dtype else weight.to(dtype)
    if isinstance(module, QuantizedLinear):
        return module.dequantized_weight(dtype)
    weight = module.weight
    return weight if weight.dtype is dtype else weight.to(dtype)


def iter_quantized(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, QUANTIZED_MODULES):
            yield name, module


def invalidate_quantization_caches(model: nn.Module) -> None:
    for _, module in iter_quantized(model):
        module.invalidate_cache()


def quantization_report(model: nn.Module) -> dict[str, dict[str, object]]:
    """Listet auf, welches Modul mit welchem Schema abgelegt ist."""
    report: dict[str, dict[str, object]] = {}
    for name, module in iter_quantized(model):
        report[name] = {
            "scheme": module.scheme,
            "group_size": module.group_size,
            "shape": list(module.weight_shape),
            "stored_bytes": module.stored_bytes(),
        }
    return report


def parameter_storage_bytes(model: nn.Module) -> int:
    """Tatsächlicher Speicherbedarf der Gewichte inklusive Quantisierung.

    ``model.parameter_count`` zählt weiterhin logische Parameter; diese Zahl
    beschreibt, wie viele Bytes davon wirklich gehalten werden. Geteilte
    Gewichte (``tie_embeddings``) werden nur einmal gezählt.
    """
    seen: set[int] = set()
    total = 0
    for _, module in iter_quantized(model):
        if id(module.packed_weight) in seen:
            continue
        seen.add(id(module.packed_weight))
        total += module.stored_bytes()
    for parameter in model.parameters():
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        total += parameter.numel() * parameter.element_size()
    return total
