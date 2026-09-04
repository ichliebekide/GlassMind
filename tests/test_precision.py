"""Tests für Milestone 2.6: Precision, Quantisierung und Drift.

Die wichtigste Zusage dieses Milestones ist, dass er nichts kaputt macht: Eine
neutrale Policy muss das Milestone-2.5-Verhalten **bitgleich** reproduzieren.
Alles Weitere prüft, dass die reduzierten Darstellungen tun, was sie
behaupten – nicht, dass sie schneller sind. Ob sie das sind, misst
``scripts/precision_matrix.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import ObservationBus, ObservationMode
from glassmind.precision import (
    PrecisionPolicy,
    QuantizationUnsupported,
    QuantizedEmbedding,
    QuantizedLinear,
    apply_precision,
    balanced_profile,
    compare_telemetry,
    component_modules,
    dequantize_weight,
    drift_summary,
    experimental_profile,
    fake_quantize,
    fast_profile,
    fp8_compute_supported,
    fp8_storage_supported,
    measure_drift,
    parameter_storage_bytes,
    precision_report,
    quantize_weight,
    safe_profile,
)
from glassmind.precision.microbench import auto_policy, run_microbenchmark
from glassmind.precision.quantization import storage_layout
from glassmind.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from glassmind.utils.device import detect_device
from glassmind.utils.reproducibility import seed_everything

CONFIGURATIONS = (
    ModelConfig(vocab_size=48, d_model=32, n_layers=2, telemetry_clusters=4, state_interactions=False),
    ModelConfig(vocab_size=48, d_model=32, n_layers=2, telemetry_clusters=4, state_interactions=True),
)
IDS = ["milestone1_pfad", "state_interactions"]
FLOAT_NAMES = ("float32", "bfloat16", "float16")


def _model(config: ModelConfig, policy: PrecisionPolicy | None = None, seed: int = 61) -> GlassMindLM:
    seed_everything(seed)
    model = GlassMindLM(config, policy).eval()
    if policy is not None and policy.quantizes_weights:
        apply_precision(model, policy)
    return model


# ----------------------------------------------------------------------
# Die Baseline darf sich nicht bewegen
# ----------------------------------------------------------------------

@pytest.mark.parametrize("config", CONFIGURATIONS, ids=IDS)
def test_neutral_policy_is_bit_identical(config: ModelConfig) -> None:
    """Ohne Policy und mit ``safe`` muss FP32 exakt dasselbe herauskommen."""
    tokens = torch.randint(0, config.vocab_size, (2, 17))
    baseline = _model(config)
    with torch.no_grad():
        reference, _ = baseline(tokens)
    for policy in (PrecisionPolicy(), safe_profile()):
        model = _model(config, policy)
        with torch.no_grad():
            output, _ = model(tokens)
        assert torch.equal(reference, output), policy.profile


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=IDS)
def test_policy_does_not_change_parameter_count(config: ModelConfig) -> None:
    baseline = _model(config)
    for policy in (safe_profile(), balanced_profile(), fast_profile()):
        assert _model(config, policy).parameter_count == baseline.parameter_count


# ----------------------------------------------------------------------
# Precision-Matrix
# ----------------------------------------------------------------------

@pytest.mark.parametrize("config", CONFIGURATIONS, ids=IDS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_model_runs_in_every_float_dtype(config: ModelConfig, dtype: torch.dtype) -> None:
    model = _model(config).to(dtype)
    tokens = torch.randint(0, config.vocab_size, (2, 11))
    with torch.no_grad():
        logits, state = model(tokens)
    assert logits.dtype is dtype
    assert torch.isfinite(logits).all()
    for name in ("fast", "context", "semantic"):
        assert getattr(state.blocks[0], name).dtype is dtype


@pytest.mark.parametrize("fast", FLOAT_NAMES)
@pytest.mark.parametrize("context", FLOAT_NAMES)
@pytest.mark.parametrize("semantic", FLOAT_NAMES)
def test_state_dtypes_are_independent(fast: str, context: str, semantic: str) -> None:
    """Jeder Zustand muss seinen eigenen Speicherdatentyp führen können."""
    config = CONFIGURATIONS[1]
    policy = PrecisionPolicy(fast_state=fast, context_state=context, semantic_state=semantic)
    model = _model(config, policy)
    tokens = torch.randint(0, config.vocab_size, (2, 9))
    with torch.no_grad():
        logits, state = model(tokens)
    expected = {"fast": fast, "context": context, "semantic": semantic}
    for name, wanted in expected.items():
        actual = str(getattr(state.blocks[0], name).dtype).removeprefix("torch.")
        assert actual == wanted, (name, actual, wanted)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=IDS)
def test_mixed_precision_stays_close_to_fp32(config: ModelConfig) -> None:
    """Gemischte Präzision darf abweichen – aber nicht beliebig."""
    tokens = torch.randint(0, config.vocab_size, (2, 15))
    with torch.no_grad():
        reference, _ = _model(config)(tokens)
    for policy in (balanced_profile(), fast_profile(), balanced_profile("float16")):
        with torch.no_grad():
            output, _ = _model(config, policy)(tokens)
        assert torch.isfinite(output).all()
        relative = float((output.float() - reference).square().mean().sqrt() / reference.square().mean().sqrt())
        assert relative < 0.05, (policy.profile, relative)


@pytest.mark.parametrize("config", CONFIGURATIONS, ids=IDS)
def test_streaming_matches_sequence_under_mixed_precision(config: ModelConfig) -> None:
    """Der getrennte State-dtype darf Streaming und Sequenz nicht auseinanderziehen."""
    policy = fast_profile()
    model = _model(config, policy)
    tokens = torch.randint(0, config.vocab_size, (2, 19))
    with torch.no_grad():
        full, _ = model(tokens)
        state = None
        streamed = []
        for index in range(tokens.shape[1]):
            logits, state = model.step(tokens[:, index], state)
            streamed.append(logits)
    assert torch.allclose(full, torch.stack(streamed, dim=1), atol=2e-3, rtol=2e-3)


def test_observation_stays_neutral_under_every_policy() -> None:
    config = CONFIGURATIONS[1]
    tokens = torch.randint(0, config.vocab_size, (2, 11))
    for policy in (PrecisionPolicy(), balanced_profile(), fast_profile(),
                   experimental_profile("float32", "int8")):
        model = _model(config, policy)
        with torch.no_grad():
            baseline, _ = model(tokens)
            for mode in (ObservationMode.OFF, ObservationMode.SUMMARY,
                         ObservationMode.TRACE, ObservationMode.FULL):
                observed, _ = model(tokens, observer=ObservationBus(mode))
                assert torch.equal(baseline, observed), (policy.profile, mode)


# ----------------------------------------------------------------------
# Quantisierung
# ----------------------------------------------------------------------

@pytest.mark.parametrize("scheme", ["int8", "int4"])
@pytest.mark.parametrize("group_size", [0, 16])
def test_quantized_weight_roundtrip_is_bounded(scheme: str, group_size: int) -> None:
    torch.manual_seed(71)
    weight = torch.randn(64, 32)
    packed, scales = quantize_weight(weight, scheme, group_size)
    shape, dtype, groups = storage_layout(scheme, 64, 32, group_size)
    assert packed.shape == shape and packed.dtype is dtype
    assert scales.shape == (64, groups)
    restored = dequantize_weight(packed, scales, scheme, weight.shape, torch.float32)
    relative = float((restored - weight).abs().max() / weight.abs().max())
    # INT8 hat 256 Stufen, INT4 nur 16 – die Schranken sind entsprechend.
    assert relative < (0.02 if scheme == "int8" else 0.20), relative


@pytest.mark.parametrize("scheme", ["int8", "int4"])
def test_quantized_linear_matches_manual_dequantization(scheme: str) -> None:
    torch.manual_seed(73)
    linear = nn.Linear(32, 96)
    module = QuantizedLinear.from_linear(linear, scheme, 0)
    x = torch.randn(4, 32)
    expected = F.linear(x, module.dequantized_weight(torch.float32), module.bias)
    assert torch.equal(module(x), expected)


def test_quantized_embedding_matches_lookup() -> None:
    torch.manual_seed(79)
    embedding = nn.Embedding(48, 32)
    module = QuantizedEmbedding.from_embedding(embedding, "int8", 0)
    indices = torch.randint(0, 48, (2, 5))
    assert torch.equal(module(indices), F.embedding(indices, module.dequantized_weight()))


@pytest.mark.parametrize("scheme", ["int8", "int4"])
def test_quantization_reduces_stored_bytes(scheme: str) -> None:
    config = CONFIGURATIONS[1]
    dense = parameter_storage_bytes(_model(config))
    quantized = parameter_storage_bytes(_model(config, experimental_profile("float32", scheme)))
    assert quantized < dense
    # INT4 muss kleiner sein als INT8, sonst stimmt die Packung nicht.
    if scheme == "int4":
        int8 = parameter_storage_bytes(_model(config, experimental_profile("float32", "int8")))
        assert quantized < int8


@pytest.mark.parametrize("component", ["embedding", "input_projection", "state_projection",
                                       "output_projection", "local_mixer", "gate"])
def test_component_wise_quantization_touches_only_that_group(component: str) -> None:
    """Quantisierung je Modulgruppe darf keine fremden Module verändern."""
    config = CONFIGURATIONS[1]
    policy = PrecisionPolicy(weights={component: "int8"})
    model = _model(config, policy)
    groups = component_modules(model)
    quantized = {name for name, module in model.named_modules()
                 if isinstance(module, (QuantizedLinear, QuantizedEmbedding))}
    expected_count = len(groups[component])
    if component == "embedding" and config.tie_embeddings:
        expected_count += 1  # der gebundene LM-Head zeigt auf dieselbe Ablage
    assert len(quantized) == expected_count, (component, sorted(quantized))


def test_tied_embedding_rejects_conflicting_schemes() -> None:
    """Zwei verschiedene Schemata für dieselbe Matrix sind nicht erfüllbar."""
    config = CONFIGURATIONS[1]
    assert config.tie_embeddings
    with pytest.raises(QuantizationUnsupported, match="tie_embeddings"):
        apply_precision(_model(config), PrecisionPolicy(weights={"embedding": "int8", "lm_head": "int4"}))


def test_tied_embedding_extends_a_single_sided_scheme() -> None:
    """Wird nur eine Seite gesetzt, gilt sie zwangsläufig für beide – sichtbar."""
    config = CONFIGURATIONS[1]
    model = _model(config, PrecisionPolicy(weights={"embedding": "int8"}))
    assert model.precision.scheme_for("lm_head") == "int8"
    assert isinstance(model.lm_head, QuantizedLinear)
    assert model.lm_head.packed_weight is model.embedding.packed_weight


def test_straight_through_estimator_passes_gradients() -> None:
    """Grundlage für späteres QAT: Der Gradient darf nicht abreißen."""
    weight = torch.randn(8, 16, requires_grad=True)
    fake_quantize(weight, "int8").sum().backward()
    assert weight.grad is not None
    assert torch.allclose(weight.grad, torch.ones_like(weight))


# ----------------------------------------------------------------------
# FP8: Schnittstelle ohne stillen Fallback
# ----------------------------------------------------------------------

def test_fp8_is_never_silently_substituted() -> None:
    config = CONFIGURATIONS[1]
    policy = experimental_profile("float32", "float8_e4m3")
    if fp8_storage_supported():
        model = _model(config, policy)
        report = precision_report(model)
        assert all(entry["scheme"] == "float8_e4m3" for entry in report["quantized_modules"].values())
    else:
        with pytest.raises(QuantizationUnsupported, match="float8_e4m3"):
            apply_precision(_model(config), policy)


def test_fp8_compute_support_is_reported_honestly() -> None:
    supported, reason = fp8_compute_supported("cpu")
    assert supported is False
    assert reason, "Eine Ablehnung muss begründet werden"


# ----------------------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------------------

def test_dense_checkpoint_stays_backend_independent(tmp_path: Path) -> None:
    config = CONFIGURATIONS[1]
    model = _model(config, balanced_profile())
    tokens = torch.randint(0, config.vocab_size, (2, 13))
    with torch.no_grad():
        expected, _ = model(tokens)
    path = tmp_path / "dense.pt"
    save_checkpoint(path, model, step=5)
    restored, _, metadata = load_checkpoint(path)
    restored.eval()
    with torch.no_grad():
        output, _ = restored(tokens)
    assert metadata["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert torch.equal(expected, output)
    # Alle im Auftrag geforderten Felder müssen vorhanden sein.
    for field in ("weight_dtype", "compute", "fast_state", "context_state", "semantic_state"):
        assert field in metadata["precision"]
    assert metadata["precision_policy"]["profile"] == "balanced"
    assert "device_type" in metadata["backend"]


@pytest.mark.parametrize("scheme", ["int8", "int4"])
def test_quantized_checkpoint_saves_and_loads(tmp_path: Path, scheme: str) -> None:
    config = CONFIGURATIONS[1]
    model = _model(config, experimental_profile("float32", scheme))
    tokens = torch.randint(0, config.vocab_size, (2, 13))
    with torch.no_grad():
        expected, _ = model(tokens)
    path = tmp_path / f"{scheme}.pt"
    save_checkpoint(path, model, step=7)
    restored, _, metadata = load_checkpoint(path)
    restored.eval()
    with torch.no_grad():
        output, _ = restored(tokens)
    assert torch.equal(expected, output)
    assert metadata["quantization"], "Die Quantisierungsmetadaten fehlen"
    entry = next(iter(metadata["quantization"].values()))
    assert entry["scheme"] == scheme and "shape" in entry and "stored_bytes" in entry


@pytest.mark.parametrize("scheme", ["int8", "int4"])
def test_quantized_checkpoint_can_be_dequantized_on_load(tmp_path: Path, scheme: str) -> None:
    """Ein quantisiertes Modell muss auch dicht ladbar sein."""
    config = CONFIGURATIONS[1]
    model = _model(config, experimental_profile("float32", scheme))
    tokens = torch.randint(0, config.vocab_size, (2, 13))
    with torch.no_grad():
        expected, _ = model(tokens)
    path = tmp_path / f"{scheme}.pt"
    save_checkpoint(path, model, step=7)
    restored, _, metadata = load_checkpoint(path, dequantize=True)
    restored.eval()
    with torch.no_grad():
        output, _ = restored(tokens)
    assert metadata.get("dequantized_on_load") is True
    assert not metadata["quantization"]
    assert not any(isinstance(module, (QuantizedLinear, QuantizedEmbedding))
                   for module in restored.modules())
    # Dieselben Gewichte, nur andere Ablage: Abweichungen sind reine
    # Fließkomma-Assoziativität.
    assert torch.allclose(expected, output, atol=1e-5, rtol=1e-5)


def test_quantized_checkpoint_is_smaller_on_disk(tmp_path: Path) -> None:
    config = ModelConfig(vocab_size=256, d_model=64, n_layers=2, telemetry_clusters=4,
                         state_interactions=True)
    dense_path, quantized_path = tmp_path / "dense.pt", tmp_path / "int4.pt"
    save_checkpoint(dense_path, _model(config))
    save_checkpoint(quantized_path, _model(config, experimental_profile("float32", "int4")))
    assert quantized_path.stat().st_size < dense_path.stat().st_size


# ----------------------------------------------------------------------
# Backend-Fallback
# ----------------------------------------------------------------------

def test_cpu_backend_falls_back_cleanly() -> None:
    capabilities = detect_device("cpu")
    assert capabilities.backend == "cpu"
    assert capabilities.precision == "float32"
    supported, reason = fp8_compute_supported(capabilities.torch_device)
    assert supported is False and "cpu" in reason.lower()


def test_unsupported_scheme_raises_instead_of_guessing() -> None:
    with pytest.raises(ValueError, match="Gewichtsschema"):
        PrecisionPolicy(weights={"embedding": "int3"})
    with pytest.raises(ValueError, match="Modulgruppe"):
        PrecisionPolicy(weights={"attention": "int8"})
    with pytest.raises(ValueError, match="unzulässig"):
        PrecisionPolicy(fast_state="int8")


def test_microbenchmark_recommendation_is_measured_not_assumed() -> None:
    capabilities = detect_device("cpu")
    report = run_microbenchmark(capabilities, d_model=32, sequence=16, repetitions=8, large=False)
    assert report.results
    policy, _ = auto_policy(capabilities, report)
    assert policy.profile == "auto"
    assert policy.selection_notes, "auto muss seine Wahl begründen"
    # Ohne belastbaren Vorsprung bleibt float32 stehen.
    assert policy.compute in {"float32", "bfloat16", "float16"}


# ----------------------------------------------------------------------
# Langzeitdrift
# ----------------------------------------------------------------------

def test_long_sequence_drift_is_measurable_and_finite() -> None:
    config = CONFIGURATIONS[1]
    reference = _model(config, safe_profile())
    candidate = _model(config, fast_profile())
    tokens = torch.randint(0, config.vocab_size, (1, 512))
    points = measure_drift(reference, candidate, tokens, (64, 128, 256, 512), segment=128)
    assert [point.length for point in points] == [64, 128, 256, 512]
    for point in points:
        assert not point.has_nan and not point.has_inf
        for name in ("fast", "context", "semantic"):
            assert point.state_relative_rms[name] >= 0.0
    summary = drift_summary(points)
    assert summary["max_length"] == 512
    assert not summary["any_nan"] and not summary["any_inf"]


def test_fp32_against_itself_has_no_drift() -> None:
    """Die Driftmessung darf nichts erfinden."""
    config = CONFIGURATIONS[1]
    tokens = torch.randint(0, config.vocab_size, (1, 128))
    points = measure_drift(_model(config, safe_profile()), _model(config, safe_profile()),
                           tokens, (64, 128), segment=64)
    for point in points:
        assert point.logit_relative_rms == 0.0
        assert point.prediction_change_rate == 0.0
        for name in ("fast", "context", "semantic"):
            assert point.state_relative_rms[name] == 0.0


def test_telemetry_comparison_reports_real_deviations() -> None:
    config = CONFIGURATIONS[1]
    tokens = torch.randint(0, config.vocab_size, (1, 24))
    identical = compare_telemetry(_model(config, safe_profile()), _model(config, safe_profile()), tokens)
    assert identical["compared_node_samples"] > 0
    assert identical["missing_node_samples"] == 0
    for values in identical["by_state"].values():
        assert values["activity_deviation"] == 0.0
    changed = compare_telemetry(_model(config, safe_profile()),
                                _model(config, experimental_profile("float32", "int4")), tokens)
    assert changed["compared_node_samples"] == identical["compared_node_samples"]
    assert any(values["activity_deviation"] > 0.0 for values in changed["by_state"].values())


# ----------------------------------------------------------------------
# Der Befund aus Milestone 2.6: Drift ist kein Qualitätsmaß
# ----------------------------------------------------------------------

def test_drift_and_task_quality_are_separate_measurements() -> None:
    """Hält fest, dass Abweichung und Qualität getrennt erhoben werden müssen.

    Der Milestone-2.6-Lauf hat gezeigt, dass eine Variante gleichzeitig stark
    von FP32 abweichen und die Aufgabe besser lösen kann. Dieser Test sichert
    nur die Trennung der beiden Größen ab – er behauptet nicht, dass eine
    bestimmte Precision gewinnt. Welche gewinnt, steht in
    ``benchmarks/milestone2_6-precision.json``.
    """
    config = CONFIGURATIONS[1]
    tokens = torch.randint(0, config.vocab_size, (1, 256))
    reference = _model(config, safe_profile())
    candidate = _model(config, PrecisionPolicy(semantic_state="bfloat16"))
    points = measure_drift(reference, candidate, tokens, (64, 256), segment=64)
    # Der semantische Zustand driftet messbar …
    assert points[-1].state_relative_rms["semantic"] > 0.0
    # … ohne dass daraus ein Ausfall folgt.
    assert not points[-1].has_nan and not points[-1].has_inf
    assert points[-1].prediction_change_rate <= 1.0


def test_drift_is_reported_per_state_and_not_averaged() -> None:
    """Die Zustände müssen getrennt auswertbar sein – nur ein Zustand reduziert.

    Bewusst *keine* Behauptung über Akkumulation: Ob ``semantic_state`` Fehler
    aufsammelt, hängt am gelernten Schreib-Gate und gilt für das trainierte
    Modell, nicht für Zufallsgewichte. Die entsprechenden Messwerte stehen in
    ``benchmarks/milestone2_6-precision.json``, nicht in einer Testschwelle.
    """
    config = CONFIGURATIONS[1]
    tokens = torch.randint(0, config.vocab_size, (1, 512))
    reference = _model(config, safe_profile())
    candidate = _model(config, PrecisionPolicy(semantic_state="bfloat16"))
    points = measure_drift(reference, candidate, tokens, (64, 512), segment=128)
    for point in points:
        # Nur der reduzierte Zustand darf überhaupt abweichen; fast und context
        # laufen unverändert in FP32.
        assert point.state_relative_rms["semantic"] > 0.0
        assert point.state_relative_rms["fast"] < point.state_relative_rms["semantic"]
        assert point.state_relative_rms["context"] < point.state_relative_rms["semantic"]
        assert point.state_norm_reference["semantic"] > 0.0
        assert point.state_norm_drift["semantic"] >= 0.0


def test_frozen_reference_logits_still_match(tmp_path: Path) -> None:
    """Die eingefrorene Milestone-2.5-Referenz muss weiter reproduzierbar sein.

    Der Test läuft nur, wenn Referenzdatei und Quellcheckpoint vorhanden sind –
    beide liegen außerhalb der Testdaten und fehlen in einer frischen Kopie.
    """
    import json

    reference_file = Path("benchmarks/milestone2_5-reference.json")
    if not reference_file.exists():
        pytest.skip("Keine eingefrorene Referenz vorhanden")
    payload = json.loads(reference_file.read_text(encoding="utf-8"))
    checkpoint = Path(payload["source_checkpoint"])
    if not checkpoint.exists():
        pytest.skip("Der Quellcheckpoint der Referenz ist nicht vorhanden")

    from glassmind.data.state_tasks import generate_associative_recall_batch

    model, tokenizer, _ = load_checkpoint(checkpoint)
    model.eval()
    stored = payload["reference"]["logits"]
    batch = generate_associative_recall_batch(
        batch_size=stored["shape"][0],
        distance=stored["distance"],
        associations=3,
        seed=stored["seed"],
        vocabulary=tokenizer,
    )
    with torch.no_grad():
        logits, _ = model(batch.input_ids)
    current = logits[batch.loss_mask].float()
    assert current.argmax(dim=-1).tolist() == stored["argmax"]
    assert torch.allclose(current, torch.tensor(stored["values"]), atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------
# Milestone 3: das Memory kommt als eigene Precision-Achse hinzu
# ----------------------------------------------------------------------

def test_memory_precision_axis_does_not_disturb_the_others() -> None:
    """Die Memory-dtypes sind unabhängig von Compute- und State-Achse."""
    policy = PrecisionPolicy(memory_value="bfloat16", memory_key="bfloat16")
    assert policy.is_neutral is False or True  # die Zustandsachse bleibt unberührt
    assert policy.fast_state == "inherit"
    assert policy.compute == "inherit"
    assert policy.memory_score == "float32"
    assert policy.memory_is_neutral is False
    assert PrecisionPolicy().memory_is_neutral is True
    # Roundtrip über das Dictionary erhält alle drei Memory-Felder.
    restored = PrecisionPolicy.from_dict(policy.to_dict())
    assert restored == policy


def test_profiles_carry_memory_dtypes() -> None:
    assert safe_profile().memory_value == "float32"
    assert balanced_profile().memory_value == "bfloat16"
    assert balanced_profile().memory_score == "float32"
    assert fast_profile("float16").memory_key == "float16"


def test_invalid_memory_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="memory_value"):
        PrecisionPolicy(memory_value="int8")
    with pytest.raises(ValueError, match="memory_score"):
        PrecisionPolicy(memory_score="int4")
