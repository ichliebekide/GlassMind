from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from glassmind.data.tokenizer import ByteTokenizer
from glassmind.data.state_tasks import StateTaskVocabulary
from glassmind.model.config import ModelConfig
from glassmind.model.lm import GlassMindLM
from glassmind.precision.apply import apply_precision, state_dtype_report
from glassmind.precision.policy import PrecisionPolicy
from glassmind.precision.quantization import dequantize_weight, quantization_report


CHECKPOINT_FORMAT_VERSION = 4
SUPPORTED_CHECKPOINT_FORMATS = (1, 2, 3, 4)


def memory_metadata(model: GlassMindLM) -> dict[str, Any]:
    """Beschreibt den konfigurierten Speicher; leer, wenn keiner existiert."""
    memory = getattr(model, "memory", None)
    if memory is None:
        return {"enabled": False}
    policy: PrecisionPolicy = getattr(model, "precision", PrecisionPolicy())
    return {
        "enabled": True,
        "slots": memory.slots,
        "width": memory.width,
        "key_dim": memory.key_dim,
        "read_top_k": memory.read_k,
        "write_top_k": memory.write_k,
        "replacement_policy": memory.replacement,
        "routing": memory.routing,
        "query_source": memory.query_source,
        "decay": memory.decay,
        "layer": model.config.memory_layer_index,
        "precision": {
            "memory_value": policy.memory_value,
            "memory_key": policy.memory_key,
            "memory_score": policy.memory_score,
        },
        "parameter_count": sum(p.numel() for p in memory.parameters()),
    }


def _backend_metadata(model: GlassMindLM) -> dict[str, Any]:
    """Woher der Checkpoint stammt. Rein informativ – das Laden bleibt portabel."""
    reference = next(
        (parameter for parameter in model.parameters() if parameter.is_floating_point()), None
    )
    device = reference.device if reference is not None else torch.device("cpu")
    metadata: dict[str, Any] = {
        "device_type": device.type,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "rocm_version": torch.version.hip,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        metadata["device_name"] = torch.cuda.get_device_name(device.index or 0)
        major, minor = torch.cuda.get_device_capability(device.index or 0)
        metadata["compute_capability"] = f"{major}.{minor}"
    return metadata


def dequantize_state_dict(
    state: dict[str, Any], quantization: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Wandelt quantisierte Puffer zurück in dichte Gewichte.

    Damit lässt sich ein quantisiertes Modell auch dort laden, wo das Schema
    nicht gebraucht oder nicht gewünscht ist – etwa auf einem anderen Backend.
    """
    restored = dict(state)
    for name, info in quantization.items():
        packed_key, scale_key = f"{name}.packed_weight", f"{name}.weight_scales"
        if packed_key not in restored:
            continue
        shape = torch.Size(info["shape"])
        weight = dequantize_weight(
            restored.pop(packed_key),
            restored.pop(scale_key),
            str(info["scheme"]),
            shape,
            torch.float32,
        )
        restored[f"{name}.weight"] = weight
    return restored


def migrate_block_parameters(
    state: dict[str, torch.Tensor], config: ModelConfig
) -> dict[str, torch.Tensor]:
    """Überführt Format 1 (entbündelte Projektionen) in Format 2 (fusioniert).

    Die Fusion legt nur zusammen, was ohnehin dieselbe Eingabe liest. Alle
    übernommenen Gewichte behalten ihre Werte exakt; verworfen werden
    ausschließlich Parameter, die im gebundenen Pfad nie gelesen wurden.
    """
    d = config.d_model
    rank, width = config.binding_rank, config.semantic_width
    parts = 4 if config.state_interactions else 6
    migrated: dict[str, torch.Tensor] = {}
    handled: set[str] = set()
    for layer in range(config.n_layers):
        prefix = f"blocks.{layer}."
        if f"{prefix}fast_recurrent.weight" not in state:
            continue
        block = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        handled.update(f"{prefix}{name}" for name in block)

        # Eingangsprojektion und Ausgangs-Gate teilen dieselbe Eingabe.
        migrated[f"{prefix}input_proj.weight"] = torch.cat(
            (block["input_proj.weight"][: parts * d], block["output_gate.weight"]), dim=0
        )
        migrated[f"{prefix}input_proj.bias"] = torch.cat(
            (block["input_proj.bias"][: parts * d], block["output_gate.bias"]), dim=0
        )
        # Alles, was aus dem vorherigen bzw. neuen fast_state gelesen wird.
        pre = [block["fast_recurrent.weight"]]
        post = [block["context_from_fast.weight"]]
        if config.state_interactions:
            pre.append(block["key_projection.weight"])
            post.append(block["value_projection.weight"])
            write_gate = block["binding_write_gate.weight"]
            pre.append(write_gate[:, :d])
            post.append(write_gate[:, d:])
            migrated[f"{prefix}binding_gate_bias"] = block["binding_write_gate.bias"]
            # Der Lesecode wird in die Integrator-Matrix aufgenommen; die
            # Spalten jenseits der Bindungsbreite waren dauerhaft null.
            migrated[f"{prefix}integrator.weight"] = torch.cat(
                (
                    block["integrator.weight"][:, : 2 * d + width],
                    block["read_to_output.weight"],
                ),
                dim=1,
            )
        else:
            migrated[f"{prefix}integrator.weight"] = block["integrator.weight"]
            for name in ("semantic_from_context.weight", "semantic_recurrent.weight", "semantic_bias"):
                migrated[f"{prefix}{name}"] = block[name]
        migrated[f"{prefix}pre_state_proj.weight"] = torch.cat(pre, dim=0)
        migrated[f"{prefix}post_state_proj.weight"] = torch.cat(post, dim=0)
        for name in ("norm.weight", "norm.bias", "context_recurrent.weight", "integrator.bias", "fast_bias", "context_bias"):
            migrated[f"{prefix}{name}"] = block[name]
    if not migrated:
        return dict(state)
    result = {key: value for key, value in state.items() if key not in handled}
    result.update(migrated)
    return result


MEMORY_STATE_FIELDS = (
    "keys", "values", "strength", "age", "usage_count",
    "read_count", "write_count", "last_read_step", "last_write_step", "occupied",
)


def _memory_state_tree(state: Any) -> dict[str, Any]:
    return {name: getattr(state, name) for name in MEMORY_STATE_FIELDS} | {"step": state.step}


def restore_memory_state(data: dict[str, Any]) -> Any:
    """Baut einen gespeicherten Laufzeitspeicher zurück."""
    from glassmind.model.memory import MemoryState

    return MemoryState(*(data[name] for name in MEMORY_STATE_FIELDS), step=int(data.get("step", 0)))


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def save_checkpoint(
    path: str | Path,
    model: GlassMindLM,
    *,
    tokenizer: ByteTokenizer | StateTaskVocabulary | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
    step: int = 0,
    epoch: int = 0,
    extra: dict[str, Any] | None = None,
    memory_state: Any = None,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    policy: PrecisionPolicy = getattr(model, "precision", PrecisionPolicy())
    reference = next(
        (parameter for parameter in model.parameters() if parameter.is_floating_point()), None
    )
    weight_dtype = str(reference.dtype).removeprefix("torch.") if reference is not None else "float32"
    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": "glassmind_selective_recurrent_v1",
        "model_config": model.config.to_dict(),
        "precision_policy": policy.to_dict(),
        "precision": {
            "weight_dtype": weight_dtype,
            **state_dtype_report(model, reference.dtype if reference is not None else torch.float32),
        },
        # Schema, Gruppengröße, Form und Bytezahl je quantisiertem Modul. Die
        # Skalen selbst liegen als Puffer im ``model_state``.
        "quantization": quantization_report(model),
        # Die Speicher*konfiguration* und die gelernten Projektionen gehören zum
        # Modell. Der Speicher*inhalt* ist Laufzeitzustand wie ``fast_state``
        # und wird bewusst nicht mitgespeichert – er ist batchabhängig und
        # beschreibt einen einzelnen Lauf, nicht das Modell. Wer ihn für Replay
        # oder Analyse braucht, übergibt ``memory_state`` ausdrücklich.
        "memory": memory_metadata(model),
        "memory_state": _cpu_tree(_memory_state_tree(memory_state)) if memory_state else None,
        "backend": _backend_metadata(model),
        "model_state": _cpu_tree(model.state_dict()),
        "optimizer_state": _cpu_tree(optimizer.state_dict()) if optimizer else None,
        "scheduler_state": _cpu_tree(scheduler.state_dict()) if scheduler else None,
        "tokenizer": (tokenizer or ByteTokenizer()).to_dict(),
        "step": step,
        "epoch": epoch,
        "extra": extra or {},
    }
    torch.save(checkpoint, destination)


def load_checkpoint(
    path: str | Path,
    *,
    device: torch.device | str = "cpu",
    dequantize: bool = False,
) -> tuple[GlassMindLM, ByteTokenizer | StateTaskVocabulary, dict[str, Any]]:
    """Lädt einen Checkpoint backendunabhängig.

    ``dequantize=True`` baut ein quantisiertes Modell als dichtes FP32-Modell
    auf. Das ist der Weg, einen quantisierten Checkpoint auf einem Backend zu
    verwenden, das das Schema nicht tragen soll.
    """
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=False)
    version = checkpoint.get("format_version")
    if version not in SUPPORTED_CHECKPOINT_FORMATS:
        raise ValueError(f"Nicht unterstütztes Checkpoint-Format: {version}")
    config = ModelConfig.from_dict(checkpoint["model_config"])
    model_state = checkpoint["model_state"]
    if version < 2:
        model_state = migrate_block_parameters(model_state, config)
    policy = PrecisionPolicy.from_dict(checkpoint.get("precision_policy") or {})
    quantization = checkpoint.get("quantization") or {}
    if quantization and dequantize:
        model_state = dequantize_state_dict(model_state, quantization)
        policy = PrecisionPolicy(
            profile=policy.profile,
            compute=policy.compute,
            activations=policy.activations,
            fast_state=policy.fast_state,
            context_state=policy.context_state,
            semantic_state=policy.semantic_state,
            selection_notes=(*policy.selection_notes, "beim Laden dequantisiert"),
        )
        quantization = {}
        checkpoint["quantization"] = {}
        checkpoint["dequantized_on_load"] = True
        checkpoint["precision_policy"] = policy.to_dict()
    model = GlassMindLM(config, policy)
    if quantization:
        # Die quantisierten Module müssen vor dem Laden existieren, damit Form
        # und dtype der Puffer passen.
        apply_precision(model, policy, device="cpu")
    model.load_state_dict(model_state)
    model.to(device)
    tokenizer_data = checkpoint["tokenizer"]
    if tokenizer_data.get("kind") == "state_task":
        tokenizer = StateTaskVocabulary.from_dict(tokenizer_data)
    else:
        tokenizer = ByteTokenizer.from_dict(tokenizer_data)
    return model, tokenizer, checkpoint
