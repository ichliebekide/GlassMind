"""Regressionstests für die Milestone-2.5-Optimierungen.

Die Optimierungen sind Umformungen: sequenzweite Vorprojektion, zusammengelegte
Projektionen mit gemeinsamer Eingabe und gebündelte Telemetrie-Transfers. Diese
Tests halten fest, dass sie die Rechnung nicht verändern.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import ObservationBus, ObservationMode
from glassmind.observe import metrics as metrics_module
from glassmind.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    migrate_block_parameters,
    save_checkpoint,
)
from glassmind.utils.device import autocast_context, detect_device
from glassmind.utils.reproducibility import seed_everything

CONFIGURATIONS = (
    ModelConfig(vocab_size=48, d_model=32, n_layers=2, telemetry_clusters=4, state_interactions=False),
    ModelConfig(vocab_size=48, d_model=32, n_layers=2, telemetry_clusters=4, state_interactions=True),
)


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=["milestone1_pfad", "state_interactions"])
def test_fused_block_matches_slow_reference(config: ModelConfig) -> None:
    seed_everything(11)
    model = GlassMindLM(config).eval()
    x = torch.randn(3, 13, config.d_model)
    for block in model.blocks:
        with torch.no_grad():
            fused, fused_state, _ = block(x)
            reference, reference_state = block.reference_forward(x)
        assert torch.allclose(fused, reference, atol=1e-5, rtol=1e-5)
        for name in ("fast", "context", "semantic"):
            assert torch.allclose(
                getattr(fused_state, name), getattr(reference_state, name), atol=1e-5, rtol=1e-5
            )


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=["milestone1_pfad", "state_interactions"])
@pytest.mark.parametrize("ablation", ["fast", "context", "semantic"])
def test_fused_block_matches_reference_under_ablation(config: ModelConfig, ablation: str) -> None:
    seed_everything(13)
    model = GlassMindLM(config).eval()
    block = model.blocks[0]
    x = torch.randn(2, 9, config.d_model)
    states = frozenset({ablation})
    with torch.no_grad():
        fused, _, _ = block(x, ablate_states=states)
        reference, _ = block.reference_forward(x, ablate_states=states)
    assert torch.allclose(fused, reference, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=["milestone1_pfad", "state_interactions"])
def test_semantic_state_width_matches_the_active_path(config: ModelConfig) -> None:
    model = GlassMindLM(config).eval()
    state = model.initial_state(2)
    assert state.blocks[0].semantic.shape[-1] == config.semantic_width
    assert state.blocks[0].fast.shape[-1] == config.d_model
    if config.state_interactions:
        # Der gebundene Pfad führt keinen unerreichbaren semantischen Zweig mehr mit.
        names = dict(model.blocks[0].named_parameters())
        assert "semantic_from_context.weight" not in names
        assert "semantic_recurrent.weight" not in names
        assert "semantic_bias" not in names


def test_batched_sequence_matches_token_by_token_streaming() -> None:
    """Die sequenzweite Vorprojektion darf den Streaming-Pfad nicht verändern."""
    seed_everything(19)
    config = CONFIGURATIONS[1]
    model = GlassMindLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 21))
    with torch.no_grad():
        full, final = model(tokens)
        state = None
        streamed = []
        for index in range(tokens.shape[1]):
            logits, state = model.step(tokens[:, index], state)
            streamed.append(logits)
    assert torch.allclose(full, torch.stack(streamed, dim=1), atol=2e-5, rtol=2e-5)
    for name in ("fast", "context", "semantic"):
        assert torch.allclose(
            getattr(final.blocks[-1], name), getattr(state.blocks[-1], name), atol=2e-5, rtol=2e-5
        )


def _legacy_state_dict(model: GlassMindLM, config: ModelConfig) -> dict[str, torch.Tensor]:
    """Baut die Format-1-Ablage aus den fusionierten Gewichten zurück."""
    d, rank, width = config.d_model, config.binding_rank, config.semantic_width
    parts = 4 if config.state_interactions else 6
    current = model.state_dict()
    legacy = {key: value for key, value in current.items() if not key.startswith("blocks.")}
    for layer, block in enumerate(model.blocks):
        prefix = f"blocks.{layer}."
        pre, post = block.pre_state_proj.weight, block.post_state_proj.weight
        legacy[f"{prefix}input_proj.weight"] = block.input_proj.weight[: parts * d]
        legacy[f"{prefix}input_proj.bias"] = block.input_proj.bias[: parts * d]
        if config.state_interactions:
            # Format 1 führte auch im gebundenen Pfad einen semantischen Zweig
            # mit, den kein Codepfad je gelesen hat.
            legacy[f"{prefix}input_proj.weight"] = torch.cat(
                (legacy[f"{prefix}input_proj.weight"], torch.zeros(2 * d, d)), dim=0
            )
            legacy[f"{prefix}input_proj.bias"] = torch.cat(
                (legacy[f"{prefix}input_proj.bias"], torch.zeros(2 * d)), dim=0
            )
        legacy[f"{prefix}output_gate.weight"] = block.input_proj.weight[parts * d :]
        legacy[f"{prefix}output_gate.bias"] = block.input_proj.bias[parts * d :]
        legacy[f"{prefix}fast_recurrent.weight"] = pre[:d]
        legacy[f"{prefix}context_from_fast.weight"] = post[:d]
        legacy[f"{prefix}integrator.bias"] = block.integrator.bias
        if config.state_interactions:
            legacy[f"{prefix}key_projection.weight"] = pre[d : d + rank]
            legacy[f"{prefix}value_projection.weight"] = post[d : d + rank]
            legacy[f"{prefix}binding_write_gate.weight"] = torch.cat(
                (pre[d + rank :], post[d + rank :]), dim=1
            )
            legacy[f"{prefix}binding_write_gate.bias"] = block.binding_gate_bias
            # Die Spalten jenseits der Bindungsbreite waren dauerhaft null.
            legacy[f"{prefix}integrator.weight"] = torch.cat(
                (block.integrator.weight[:, : 2 * d + width], torch.zeros(d, d - width)), dim=1
            )
            legacy[f"{prefix}read_to_output.weight"] = block.integrator.weight[:, 2 * d + width :]
            legacy[f"{prefix}semantic_from_context.weight"] = torch.zeros(d, d)
            legacy[f"{prefix}semantic_recurrent.weight"] = torch.zeros(d, d)
            legacy[f"{prefix}semantic_bias"] = torch.zeros(d)
        else:
            legacy[f"{prefix}integrator.weight"] = block.integrator.weight
            for name in ("semantic_from_context.weight", "semantic_recurrent.weight", "semantic_bias"):
                legacy[f"{prefix}{name}"] = current[f"{prefix}{name}"]
        for name in ("norm.weight", "norm.bias", "context_recurrent.weight", "fast_bias", "context_bias"):
            legacy[f"{prefix}{name}"] = current[f"{prefix}{name}"]
    return legacy


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=["milestone1_pfad", "state_interactions"])
def test_legacy_checkpoint_migrates_without_changing_predictions(
    config: ModelConfig, tmp_path: Path
) -> None:
    """Format 1 legt dieselben Gewichte nur anders ab – die Ausgabe bleibt gleich."""
    seed_everything(23)
    model = GlassMindLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 15))
    with torch.no_grad():
        expected, _ = model(tokens)

    restored = GlassMindLM(config)
    restored.load_state_dict(migrate_block_parameters(_legacy_state_dict(model, config), config))
    restored.eval()
    with torch.no_grad():
        migrated, _ = restored(tokens)
    assert torch.allclose(expected, migrated, atol=1e-6, rtol=1e-6)

    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, step=3)
    reloaded, _, metadata = load_checkpoint(path)
    assert metadata["format_version"] == CHECKPOINT_FORMAT_VERSION
    reloaded.eval()
    with torch.no_grad():
        again, _ = reloaded(tokens)
    assert torch.equal(expected, again)


def test_batched_cluster_statistics_match_per_cluster_reduction() -> None:
    """Die gebündelte Telemetrie muss dieselben Zahlen liefern wie einzeln."""
    torch.manual_seed(29)
    for width, clusters in ((64, 8), (25, 4), (32, 4)):
        tensors = [torch.randn(2, width) for _ in range(4)]
        fast = metrics_module.cluster_statistics(*tensors, clusters)
        flow_fast = metrics_module.cluster_flow_rms(tensors[0], clusters)
        original = metrics_module._cluster_view
        metrics_module._cluster_view = lambda *_: None
        try:
            slow = metrics_module.cluster_statistics(*tensors, clusters)
            flow_slow = metrics_module.cluster_flow_rms(tensors[0], clusters)
        finally:
            metrics_module._cluster_view = original
        assert fast.shape == (clusters, metrics_module.CLUSTER_STATISTIC_COUNT)
        assert torch.allclose(fast, slow, atol=1e-6, rtol=1e-6)
        assert torch.allclose(flow_fast, flow_slow, atol=1e-6, rtol=1e-6)


def test_sequence_summary_matches_single_position_summary() -> None:
    torch.manual_seed(31)
    values = torch.randn(3, 6, 9)
    values[0, 2, 4] = float("inf")
    summaries = metrics_module.sequence_tensor_summary(values)
    for index, summary in enumerate(summaries):
        single = metrics_module.tensor_summary(values[:, index])
        for field, value in single.items():
            assert summary[field] == pytest.approx(value, abs=1e-6)


def test_observation_stays_numerically_neutral_in_every_mode() -> None:
    seed_everything(37)
    config = CONFIGURATIONS[1]
    model = GlassMindLM(config).eval()
    tokens = torch.randint(0, config.vocab_size, (2, 11))
    with torch.no_grad():
        baseline, _ = model(tokens)
        for mode in (ObservationMode.OFF, ObservationMode.SUMMARY, ObservationMode.TRACE, ObservationMode.FULL):
            observed, _ = model(tokens, observer=ObservationBus(mode))
            assert torch.equal(baseline, observed), mode


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Kein CUDA/ROCm-Backend verfügbar")
def test_training_step_runs_on_the_accelerator() -> None:
    """Fängt Rückwärtspfade ab, die nur auf dem Beschleuniger scheitern.

    Ein Beispiel aus der Praxis: ``torch.bmm`` wird in aktuellen Builds über
    einen Triton-Kernel rückwärts geführt, der ohne CPython-Header nicht baubar
    ist. Auf CPU fällt das nicht auf.
    """
    capabilities = detect_device("auto", "auto")
    seed_everything(41)
    config = CONFIGURATIONS[1]
    model = GlassMindLM(config).to(capabilities.torch_device).train()
    tokens = torch.randint(0, config.vocab_size, (2, 9), device=capabilities.torch_device)
    with autocast_context(capabilities):
        logits, _ = model(tokens)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size).float(), tokens.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
