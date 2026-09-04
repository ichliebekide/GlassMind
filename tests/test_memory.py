"""Tests für Milestone 3: bounded sparse external memory.

Zwei Zusagen stehen im Vordergrund:

* Ohne konfigurierte Slots existiert der Speicher nicht und ändert nichts –
  die Milestone-2.6-Baseline bleibt bitgleich.
* Alles, was der Speicher meldet, ist gemessen: Slot-Auswahl, Zähler,
  Ersetzungsereignisse und Replay stammen aus echten Zugriffen.

Kein Test behauptet, dass der Speicher die Aufgabenqualität verbessert. Ob er
das tut, misst ``scripts/memory_study.py``; das Ergebnis steht in
``benchmarks/milestone3-memory.json``.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from glassmind.data import MEMORY_TASK_GENERATORS, memory_task_vocabulary
from glassmind.model import GlassMindLM, ModelConfig
from glassmind.model.memory import (
    INTERVENTIONS,
    QUERY_SOURCES,
    REPLACEMENT_POLICIES,
    ROUTING_MODES,
    apply_slot_intervention,
    memory_utilisation,
    replace_slot_value,
)
from glassmind.observe import JSONLRecorder, ObservationBus, ObservationMode
from glassmind.precision import PrecisionPolicy, apply_precision, balanced_profile
from glassmind.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    restore_memory_state,
    save_checkpoint,
)
from glassmind.utils.device import detect_device
from glassmind.utils.reproducibility import seed_everything
from glassmind.visualize.graph import ReplayTimeline, memory_arrays
from glassmind.visualize.layout import memory_layout

BASE = dict(vocab_size=48, d_model=32, n_layers=2, telemetry_clusters=4, state_interactions=True)
MEMORY = dict(memory_slots=16, memory_width=32, memory_key_dim=16, memory_read_k=2, memory_write_k=1)


def _model(seed: int = 31, **overrides) -> GlassMindLM:
    seed_everything(seed)
    return GlassMindLM(ModelConfig(**{**BASE, **overrides})).eval()


def _tokens(length: int = 24, batch: int = 2) -> torch.Tensor:
    return torch.randint(0, BASE["vocab_size"], (batch, length))


# ----------------------------------------------------------------------
# Die Baseline darf sich nicht bewegen
# ----------------------------------------------------------------------

def test_without_slots_no_memory_exists() -> None:
    """Ohne Slots gibt es kein Modul, keinen Zustand und keinen Aufruf."""
    model = _model()
    assert model.memory is None
    assert not model.config.has_memory
    tokens = _tokens()
    with torch.no_grad():
        _, state = model(tokens)
    assert state.memory is None


def test_disabled_memory_matches_model_without_memory() -> None:
    """``--disable-memory`` muss exakt dem Modell ohne Speicher entsprechen."""
    tokens = _tokens()
    plain = _model()
    with torch.no_grad():
        reference, _ = plain(tokens)
    with_memory = _model(**MEMORY)
    # Die Speicherparameter entstehen nach den Blockgewichten; damit beide
    # Modelle dieselben Blockgewichte tragen, wird der Zustand übernommen.
    with_memory.load_state_dict(plain.state_dict(), strict=False)
    with torch.no_grad():
        disabled, state = with_memory(tokens, disable_memory=True)
    assert torch.equal(reference, disabled)
    # Der Speicher bleibt unangetastet: keine Schreibzugriffe, keine Belegung.
    assert state.memory is not None
    assert float(state.memory.occupied.sum()) == 0.0


# ----------------------------------------------------------------------
# Lesen, Schreiben, Sparsity
# ----------------------------------------------------------------------

def test_read_and_write_touch_only_top_k_slots() -> None:
    """Sparse heißt: genau ``k`` Slots je Zugriff, nicht alle."""
    model = _model(**{**MEMORY, "memory_slots": 32, "memory_read_k": 3, "memory_write_k": 2})
    events: list = []
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(events.append)
    with torch.no_grad():
        model(_tokens(20, 1), observer=bus)
    steps = [event for event in events if event.event == "memory_step"]
    assert steps
    for event in steps:
        assert len(event.payload["selected_read_slots"]) == 3
        assert len(event.payload["selected_write_slots"]) == 2
        assert all(0 <= slot < 32 for slot in event.payload["selected_write_slots"])


def test_write_fills_free_slots_before_replacing() -> None:
    """Ein leerer Speicher wird erst gefüllt, dann verdrängt."""
    model = _model(**{**MEMORY, "memory_slots": 8, "memory_write_k": 1,
                      "memory_track_usage": True})
    with torch.no_grad():
        _, state = model(_tokens(8, 1))
    utilisation = memory_utilisation(state.memory)
    # Acht Token, acht Slots: jeder Slot genau einmal beschrieben.
    assert utilisation["occupied_slots"] == 8
    assert all(count == 1 for count in utilisation["write_distribution"])


def test_replacement_happens_only_when_memory_is_full() -> None:
    model = _model(**{**MEMORY, "memory_slots": 4})
    events: list = []
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(events.append)
    with torch.no_grad():
        model(_tokens(12, 1), observer=bus)
    steps = [event for event in events if event.event == "memory_step"]
    replacements = [len(event.payload["replacement_events"]) for event in steps]
    # Die ersten vier Token füllen, danach muss jedes Schreiben ersetzen.
    assert sum(replacements[:4]) == 0
    assert all(count == 1 for count in replacements[4:])


def test_counters_and_age_advance() -> None:
    model = _model(**MEMORY, memory_track_usage=True)
    with torch.no_grad():
        _, state = model(_tokens(16, 1))
    utilisation = memory_utilisation(state.memory)
    assert utilisation["total_writes"] > 0
    assert utilisation["total_reads"] > 0
    assert float(state.memory.age.max()) > 0
    # Ein gerade beschriebener Slot ist jünger als der älteste.
    assert float(state.memory.age.min()) < float(state.memory.age.max())
    assert float(state.memory.strength.max()) > 0


def test_usage_tracking_is_pure_bookkeeping() -> None:
    """Die Zähler kosten Durchsatz, dürfen die Rechnung aber nicht verändern."""
    tokens = _tokens(16, 1)
    without = _model(**MEMORY)
    with_counters = _model(**MEMORY, memory_track_usage=True)
    with torch.no_grad():
        plain_logits, plain_state = without(tokens)
        counted_logits, counted_state = with_counters(tokens)
    assert torch.equal(plain_logits, counted_logits)
    assert torch.equal(plain_state.memory.values, counted_state.memory.values)
    assert torch.equal(plain_state.memory.strength, counted_state.memory.strength)
    # Ohne Zähler meldet die Auswertung das offen, statt Nullen zu behaupten.
    assert memory_utilisation(plain_state.memory)["counters_tracked"] is False
    assert memory_utilisation(counted_state.memory)["counters_tracked"] is True


@pytest.mark.parametrize("policy", REPLACEMENT_POLICIES)
def test_every_replacement_policy_runs(policy: str) -> None:
    model = _model(**{**MEMORY, "memory_replacement": policy})
    with torch.no_grad():
        logits, state = model(_tokens(14, 1))
    assert torch.isfinite(logits).all()
    assert memory_utilisation(state.memory)["occupied_slots"] > 0


@pytest.mark.parametrize("routing", ROUTING_MODES)
def test_every_routing_mode_runs(routing: str) -> None:
    model = _model(**{**MEMORY, "memory_routing": routing})
    with torch.no_grad():
        logits, _ = model(_tokens(12, 1))
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("source", QUERY_SOURCES)
def test_every_query_source_runs(source: str) -> None:
    model = _model(**{**MEMORY, "memory_query_source": source})
    with torch.no_grad():
        logits, state = model(_tokens(12, 1))
    assert torch.isfinite(logits).all()
    assert state.memory is not None


# ----------------------------------------------------------------------
# Randfälle
# ----------------------------------------------------------------------

def test_zero_slots_is_valid_and_neutral() -> None:
    """0 Slots ist kein Fehler, sondern schlicht kein Speicher."""
    model = _model(memory_slots=0)
    assert model.memory is None
    with torch.no_grad():
        logits, state = model(_tokens(8, 1))
    assert torch.isfinite(logits).all()
    assert state.memory is None


def test_single_slot_works() -> None:
    model = _model(**{**MEMORY, "memory_slots": 1, "memory_read_k": 1, "memory_write_k": 1})
    with torch.no_grad():
        logits, state = model(_tokens(10, 1))
    assert torch.isfinite(logits).all()
    assert state.memory.slots == 1
    assert memory_utilisation(state.memory)["occupied_slots"] == 1


def test_top_k_larger_than_slots_is_clamped() -> None:
    """Top-K über der Slotzahl wird heruntergesetzt, statt abzustürzen."""
    model = _model(**{**MEMORY, "memory_slots": 2, "memory_read_k": 8, "memory_write_k": 5})
    assert model.memory.read_k == 2
    assert model.memory.write_k == 2
    with torch.no_grad():
        logits, _ = model(_tokens(8, 1))
    assert torch.isfinite(logits).all()


def test_empty_memory_read_produces_finite_output() -> None:
    """Der allererste Lesezugriff trifft einen leeren Speicher."""
    model = _model(**MEMORY)
    with torch.no_grad():
        logits, _ = model(_tokens(1, 1))
    assert torch.isfinite(logits).all()


def test_full_memory_keeps_slot_count_bounded() -> None:
    """Der Speicher wächst nie über seine Slotzahl hinaus."""
    model = _model(**{**MEMORY, "memory_slots": 4})
    state = None
    with torch.no_grad():
        for _ in range(6):
            _, state = model(_tokens(8, 1), state)
    assert state.memory.slots == 4
    assert state.memory.values.shape == (1, 4, MEMORY["memory_width"])
    assert memory_utilisation(state.memory)["occupied_slots"] == 4


# ----------------------------------------------------------------------
# Ablation und Intervention
# ----------------------------------------------------------------------

def test_disable_read_and_write_are_independent() -> None:
    model = _model(**MEMORY, memory_track_usage=True)
    tokens = _tokens(16, 1)
    with torch.no_grad():
        _, full = model(tokens)
        _, no_read = model(tokens, disable_memory_read=True)
        _, no_write = model(tokens, disable_memory_write=True)
    assert float(full.memory.read_count.sum()) > 0
    assert float(no_read.memory.read_count.sum()) == 0
    assert float(no_read.memory.occupied.sum()) > 0      # geschrieben wurde trotzdem
    assert float(no_write.memory.occupied.sum()) == 0    # nichts geschrieben


def test_slot_ablation_changes_only_that_slot_selection() -> None:
    model = _model(**{**MEMORY, "memory_slots": 8})
    tokens = _tokens(20, 1)
    events: list = []
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(events.append)
    with torch.no_grad():
        model(tokens, observer=bus, ablate_memory_slots=[3])
    steps = [event for event in events if event.event == "memory_step"]
    assert steps
    for event in steps:
        assert 3 not in event.payload["selected_read_slots"]
        assert 3 not in event.payload["selected_write_slots"]


def test_invalid_slot_ablation_is_rejected() -> None:
    model = _model(**{**MEMORY, "memory_slots": 8})
    with pytest.raises(ValueError, match="Slot 99"):
        with torch.no_grad():
            model(_tokens(4, 1), ablate_memory_slots=[99])


@pytest.mark.parametrize("operation", INTERVENTIONS)
def test_interventions_run_and_stay_bounded(operation: str) -> None:
    model = _model(**{**MEMORY, "memory_slots": 8})
    with torch.no_grad():
        logits, state = model(_tokens(12, 1), memory_interventions={2: operation})
    assert torch.isfinite(logits).all()
    assert state.memory.slots == 8


def test_clear_and_replace_slot_operate_on_runtime_state() -> None:
    model = _model(**{**MEMORY, "memory_slots": 8})
    with torch.no_grad():
        _, state = model(_tokens(12, 1))
    cleared = apply_slot_intervention(state.memory, 2, "clear")
    assert float(cleared.values[:, 2].abs().sum()) == 0.0
    assert float(cleared.occupied[0, 2]) == 0.0
    # Die übrigen Slots bleiben unberührt.
    assert torch.equal(cleared.values[:, 3], state.memory.values[:, 3])
    replaced = replace_slot_value(state.memory, 2, torch.ones(MEMORY["memory_width"]))
    assert float(replaced.values[0, 2].mean()) == pytest.approx(1.0)
    assert float(replaced.occupied[0, 2]) == 1.0


# ----------------------------------------------------------------------
# Streaming, Beobachtung, Precision
# ----------------------------------------------------------------------

def test_streaming_matches_sequence_with_memory() -> None:
    """Der Speicher darf Streaming und Sequenzlauf nicht auseinanderziehen."""
    model = _model(**MEMORY)
    tokens = _tokens(17, 2)
    with torch.no_grad():
        full, final = model(tokens)
        state = None
        streamed = []
        for index in range(tokens.shape[1]):
            logits, state = model.step(tokens[:, index], state)
            streamed.append(logits)
    assert torch.allclose(full, torch.stack(streamed, dim=1), atol=2e-5, rtol=2e-5)
    assert torch.allclose(final.memory.values, state.memory.values, atol=2e-5, rtol=2e-5)
    assert torch.equal(final.memory.occupied, state.memory.occupied)


def test_observation_off_is_numerically_neutral() -> None:
    model = _model(**MEMORY)
    tokens = _tokens(14, 2)
    with torch.no_grad():
        baseline, base_state = model(tokens)
        for mode in (ObservationMode.OFF, ObservationMode.SUMMARY,
                     ObservationMode.TRACE, ObservationMode.FULL):
            observed, state = model(tokens, observer=ObservationBus(mode))
            assert torch.equal(baseline, observed), mode
            assert torch.equal(base_state.memory.values, state.memory.values), mode


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_memory_runs_in_every_float_dtype(dtype: torch.dtype) -> None:
    model = _model(**MEMORY).to(dtype)
    with torch.no_grad():
        logits, state = model(_tokens(12, 1))
    assert logits.dtype is dtype
    assert torch.isfinite(logits).all()
    assert state.memory.values.dtype is dtype


def test_memory_precision_axis_is_independent() -> None:
    """Werte und Schlüssel dürfen reduziert werden, die Scores getrennt davon."""
    policy = PrecisionPolicy(memory_value="bfloat16", memory_key="bfloat16", memory_score="float32")
    seed_everything(31)
    model = GlassMindLM(ModelConfig(**{**BASE, **MEMORY}), policy).eval()
    with torch.no_grad():
        logits, state = model(_tokens(12, 1))
    assert torch.isfinite(logits).all()
    assert state.memory.values.dtype is torch.bfloat16
    assert state.memory.keys.dtype is torch.bfloat16
    assert state.memory.strength.dtype is torch.float32
    assert balanced_profile().memory_value == "bfloat16"
    assert balanced_profile().memory_score == "float32"


def test_weight_quantization_still_works_with_memory() -> None:
    """Die beste Milestone-2.6-Konfiguration muss weiter möglich sein."""
    policy = PrecisionPolicy(
        fast_state="bfloat16", context_state="bfloat16", semantic_state="float32",
        memory_value="bfloat16", memory_key="bfloat16", memory_score="float32",
        weights={"embedding": "int8", "lm_head": "int8", "input_projection": "int4",
                 "state_projection": "int8", "output_projection": "int8"},
    )
    seed_everything(31)
    model = GlassMindLM(ModelConfig(**{**BASE, **MEMORY}), policy).eval()
    apply_precision(model, policy)
    with torch.no_grad():
        logits, state = model(_tokens(12, 1))
    assert torch.isfinite(logits).all()
    assert state.memory is not None


# ----------------------------------------------------------------------
# Checkpoints und Replay
# ----------------------------------------------------------------------

def test_checkpoint_stores_memory_configuration(tmp_path: Path) -> None:
    model = _model(**MEMORY)
    tokens = _tokens(12, 1)
    with torch.no_grad():
        expected, state = model(tokens)
    path = tmp_path / "memory.pt"
    save_checkpoint(path, model, step=3)
    restored, _, metadata = load_checkpoint(path)
    restored.eval()
    with torch.no_grad():
        output, _ = restored(tokens)
    assert metadata["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert torch.equal(expected, output)
    memory = metadata["memory"]
    for field in ("slots", "width", "key_dim", "read_top_k", "write_top_k",
                  "replacement_policy", "routing", "query_source", "precision"):
        assert field in memory
    assert memory["slots"] == MEMORY["memory_slots"]
    # Laufzeitinhalt gehört nicht zum Modell und wird nicht mitgespeichert.
    assert metadata["memory_state"] is None


def test_runtime_memory_can_be_stored_on_request(tmp_path: Path) -> None:
    model = _model(**MEMORY)
    with torch.no_grad():
        _, state = model(_tokens(12, 1))
    path = tmp_path / "with_state.pt"
    save_checkpoint(path, model, step=3, memory_state=state.memory)
    _, _, metadata = load_checkpoint(path)
    restored = restore_memory_state(metadata["memory_state"])
    assert restored.slots == state.memory.slots
    assert torch.equal(restored.values, state.memory.values.cpu())
    assert torch.equal(restored.occupied, state.memory.occupied.cpu())


def test_replay_reconstructs_memory_without_the_model(tmp_path: Path) -> None:
    """Ein aufgezeichneter Lauf muss ohne Modell vollständig darstellbar sein."""
    model = _model(**{**MEMORY, "memory_slots": 16})
    trace = tmp_path / "memory-trace.jsonl"
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(JSONLRecorder(trace, flush_every=1))
    with torch.no_grad():
        model(_tokens(18, 1), observer=bus)
    bus.close()
    del model  # Ab hier existiert kein Modell mehr.

    timeline = ReplayTimeline.from_trace(trace)
    assert len(timeline) == 18
    frame = timeline[-1]
    assert frame.memory is not None
    assert frame.memory.slots == 16
    arrays = memory_arrays(frame.memory)
    # ``memory_arrays`` liefert Messwerte je Zelle; die räumliche Anordnung
    # kommt seit Milestone 4.5 ausschließlich aus ``visualize.layout``.
    assert arrays["slots"] == 16
    assert len(memory_layout(arrays["slots"])) == 16
    assert len(arrays["strength"]) == 16
    assert sum(arrays["occupied"]) > 0
    assert sum(arrays["write_active"]) == 1
    # Live und Replay teilen dieselbe Datenstruktur.
    detail = frame.memory.slot_detail(frame.memory.read_slots[0], frame.token_index)
    for field in ("strength", "age", "reads", "writes", "occupied", "current_score"):
        assert field in detail


def test_memory_layout_stays_compact_for_many_slots() -> None:
    """Bei 128 Slots bleibt die Bank ein kompaktes Gitter."""
    # Seit Milestone 4.5 liegt das Raster in ``visualize.layout`` – als einzige
    # Quelle, aus der auch die Kamera ihre Grenzen bezieht.
    positions = list(memory_layout(128, columns=16).values())
    assert len(positions) == 128
    xs = [x for x, _ in positions]
    ys = [y for _, y in positions]
    assert max(xs) - min(xs) < 11.0
    assert max(ys) - min(ys) < 6.0
    assert len(set(positions)) == 128   # keine überlappenden Zellen


# ----------------------------------------------------------------------
# Aufgaben
# ----------------------------------------------------------------------

def test_memory_tasks_are_reproducible_and_solvable_in_principle() -> None:
    vocabulary = memory_task_vocabulary()
    for name, generator in MEMORY_TASK_GENERATORS.items():
        first = generator(batch_size=4, distance=24, seed=51, vocabulary=vocabulary)
        second = generator(batch_size=4, distance=24, seed=51, vocabulary=vocabulary)
        assert torch.equal(first.input_ids, second.input_ids), name
        assert torch.equal(first.loss_mask, second.loss_mask), name
        assert first.task == name
        assert int(first.loss_mask.sum()) == first.answer_count * 4, name
        # Die Antwort steht im Beispiel: Der Zielwert muss vorher gespeichert sein.
        row = first.input_ids[0].tolist()
        position = int(first.loss_mask[0].nonzero()[0])
        target = int(first.targets[0, position])
        assert target in row, name


def test_memory_replacement_task_exceeds_slot_count() -> None:
    """Die Kapazitätsaufgabe muss mehr Fakten enthalten als ein kleiner Speicher hat."""
    vocabulary = memory_task_vocabulary()
    batch = MEMORY_TASK_GENERATORS["memory_replacement"](
        batch_size=2, distance=0, facts=24, seed=57, vocabulary=vocabulary
    )
    stores = (batch.input_ids[0] == vocabulary.store).sum()
    assert int(stores) == 24


def test_repeated_retrieval_asks_several_times() -> None:
    vocabulary = memory_task_vocabulary()
    batch = MEMORY_TASK_GENERATORS["repeated_retrieval"](
        batch_size=2, distance=48, retrievals=3, seed=59, vocabulary=vocabulary
    )
    assert batch.answer_count == 3
    assert int(batch.loss_mask[0].sum()) == 3
    targets = {int(batch.targets[0, position]) for position in batch.loss_mask[0].nonzero().flatten()}
    assert len(targets) == 1   # immer derselbe Fakt


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Kein CUDA/ROCm-Backend verfügbar")
def test_memory_training_step_runs_on_the_accelerator() -> None:
    """Fängt Rückwärtspfade ab, die nur auf dem Beschleuniger scheitern."""
    capabilities = detect_device("auto", "auto")
    seed_everything(41)
    config = ModelConfig(**{**BASE, **MEMORY})
    model = GlassMindLM(config).to(capabilities.torch_device).train()
    tokens = torch.randint(0, config.vocab_size, (2, 12), device=capabilities.torch_device)
    logits, _ = model(tokens)
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size).float(), tokens.reshape(-1))
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.memory.parameters() if p.requires_grad]
    assert gradients and all(g is not None and torch.isfinite(g).all() for g in gradients)
