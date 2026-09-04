"""Wendet eine :class:`PrecisionPolicy` auf ein GlassMind-Modell an.

Die gesamte Zuordnung „Modulgruppe → konkrete Module" steht hier und nur hier.
Der Modellcode enthält keine Quantisierungsfallunterscheidungen; er fragt beim
Rechnen lediglich über ``linear_weight`` nach der Gewichtsmatrix.
"""
from __future__ import annotations

from typing import Iterator

import torch
from torch import nn

from glassmind.precision.policy import INHERIT, PrecisionPolicy, resolve_dtype
from glassmind.precision.quantization import (
    QuantizationUnsupported,
    QuantizedEmbedding,
    QuantizedLinear,
    parameter_storage_bytes,
    quantization_report,
    require_scheme_supported,
)


def component_modules(model: nn.Module) -> dict[str, list[tuple[nn.Module, str]]]:
    """Ordnet jede Modulgruppe den (Elternmodul, Attributname)-Paaren zu.

    Die Gruppennamen folgen der Architektur, nicht einer Wunschvorstellung:
    Wert- und Gate-Anteile eines Blocks teilen sich seit Milestone 2.5 die
    Matrix ``input_proj``, weshalb ``gate`` nur noch das Gate des lokalen
    Mixers bezeichnet.
    """
    groups: dict[str, list[tuple[nn.Module, str]]] = {
        "embedding": [(model, "embedding")],
        "local_mixer": [(model.local_mixer, "channel_mix")],
        "gate": [(model.local_mixer, "gate")],
        "input_projection": [],
        "state_projection": [],
        "output_projection": [],
        "lm_head": [(model, "lm_head")],
    }
    for block in model.blocks:
        groups["input_projection"].append((block, "input_proj"))
        groups["state_projection"].extend(
            [
                (block, "pre_state_proj"),
                (block, "post_state_proj"),
                (block, "context_recurrent"),
            ]
        )
        if not block.state_interactions:
            groups["state_projection"].extend(
                [(block, "semantic_from_context"), (block, "semantic_recurrent")]
            )
        groups["output_projection"].append((block, "integrator"))
    return groups


def _tied(model: nn.Module) -> bool:
    return bool(
        getattr(model.config, "tie_embeddings", False)
        and isinstance(model.embedding, nn.Embedding)
        and isinstance(model.lm_head, nn.Linear)
        and model.lm_head.weight is model.embedding.weight
    )


def apply_precision(
    model: nn.Module, policy: PrecisionPolicy, *, device: torch.device | str | None = None
) -> nn.Module:
    """Baut das Modell gemäß Policy um. Verändert ``model`` in place.

    Eine neutrale Policy lässt das Modell unverändert – das ist der Normalfall
    und garantiert, dass die Milestone-2.5-Baseline erhalten bleibt.
    """
    target = torch.device(device) if device is not None else next(model.parameters()).device
    for scheme in set(policy.weights.values()):
        if scheme != "none":
            require_scheme_supported(scheme, target)

    model.precision = policy
    for block in model.blocks:
        block.precision = policy

    if not policy.quantizes_weights:
        return model

    tied = _tied(model)
    if tied:
        embedding_scheme = policy.scheme_for("embedding")
        head_scheme = policy.scheme_for("lm_head")
        if embedding_scheme != head_scheme:
            if "none" not in (embedding_scheme, head_scheme):
                raise QuantizationUnsupported(
                    "Bei tie_embeddings teilen Embedding und LM-Head dieselbe Matrix. "
                    "Sie können nicht unterschiedlich quantisiert werden "
                    f"(embedding={embedding_scheme}, lm_head={head_scheme}). "
                    "Setze beide gleich oder deaktiviere tie_embeddings."
                )
            # Nur eine Seite gesetzt: Da beide dieselbe Matrix sind, gilt das
            # Schema zwangsläufig für beide. Das ist keine stille Ausweitung,
            # sondern die einzige mögliche Lesart – und sie steht im Bericht.
            shared = embedding_scheme if embedding_scheme != "none" else head_scheme
            policy = policy.with_weights(embedding=shared, lm_head=shared)
            model.precision = policy
            for block in model.blocks:
                block.precision = policy

    groups = component_modules(model)
    for component, entries in groups.items():
        scheme = policy.scheme_for(component)
        if scheme == "none":
            continue
        for parent, attribute in entries:
            module = getattr(parent, attribute)
            if isinstance(module, (QuantizedLinear, QuantizedEmbedding)):
                continue
            if isinstance(module, nn.Embedding):
                replacement: nn.Module = QuantizedEmbedding.from_embedding(
                    module, scheme, policy.weight_group_size, policy.dequantization_cache
                )
            elif isinstance(module, nn.Linear):
                replacement = QuantizedLinear.from_linear(
                    module, scheme, policy.weight_group_size, policy.dequantization_cache
                )
            else:
                raise QuantizationUnsupported(
                    f"{component}.{attribute} ist vom Typ {type(module).__name__} "
                    "und wird von der Weight-Only-Quantisierung nicht abgedeckt"
                )
            setattr(parent, attribute, replacement.to(target))

    if tied and policy.scheme_for("embedding") != "none":
        # Die Bindung bleibt bestehen: beide Seiten greifen auf dieselben
        # gepackten Werte zu, statt sie doppelt zu halten.
        model.lm_head.packed_weight = model.embedding.packed_weight
        model.lm_head.weight_scales = model.embedding.weight_scales
    return model


def quantized_model(
    model: nn.Module, policy: PrecisionPolicy, *, device: torch.device | str | None = None
) -> nn.Module:
    return apply_precision(model, policy, device=device)


def precision_report(model: nn.Module) -> dict[str, object]:
    """Fasst zusammen, wie das Modell tatsächlich abgelegt ist."""
    policy: PrecisionPolicy = getattr(model, "precision", PrecisionPolicy())
    parameter_dtypes: dict[str, int] = {}
    for parameter in model.parameters():
        key = str(parameter.dtype).removeprefix("torch.")
        parameter_dtypes[key] = parameter_dtypes.get(key, 0) + parameter.numel()
    return {
        "policy": policy.to_dict(),
        "quantized_modules": quantization_report(model),
        "parameter_dtypes": parameter_dtypes,
        "parameter_storage_bytes": parameter_storage_bytes(model),
    }


def state_dtype_report(model: nn.Module, fallback: torch.dtype) -> dict[str, str]:
    policy: PrecisionPolicy = getattr(model, "precision", PrecisionPolicy())
    fast, context, semantic = policy.state_dtypes(fallback)
    compute = fallback if policy.compute == INHERIT else resolve_dtype(policy.compute, fallback)
    return {
        "compute": str(compute).removeprefix("torch."),
        "fast_state": str(fast).removeprefix("torch."),
        "context_state": str(context).removeprefix("torch."),
        "semantic_state": str(semantic).removeprefix("torch."),
    }


def iter_linear_like(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, QuantizedLinear, QuantizedEmbedding)):
            yield name, module
