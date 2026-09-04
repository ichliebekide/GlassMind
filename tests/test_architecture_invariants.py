"""Sichert die nicht verhandelbaren Architekturregeln aus ``AGENTS.md`` ab.

Diese Regeln sind bisher an vielen Stellen *beschrieben* worden. Hier werden
sie *geprüft* – und zwar an dem, was der Code tatsächlich ausführt, nicht an
dem, was die Dokumentation über ihn sagt.

Die zentrale Abgrenzung ist bewusst präzise formuliert: Der Memory-Read *hat*
die Form einer Attention-Operation (Query, Keys, gewichtete Wertesumme). Der
Unterschied zu Self-Attention liegt nicht in der Form, sondern darin, worüber
attendiert wird – über einen festen Slot-Satz statt über alle bisherigen
Tokens. Genau das prüfen diese Tests.
"""
from __future__ import annotations

from collections import Counter

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from glassmind.model import GlassMindLM, ModelConfig
from glassmind.utils.reproducibility import seed_everything

CONFIGURATIONS = {
    "milestone1_pfad": dict(state_interactions=False),
    "state_interactions": dict(state_interactions=True),
    "mit_memory": dict(state_interactions=True, memory_slots=64, memory_width=32,
                       memory_key_dim=16, memory_read_k=2, memory_write_k=1),
}
BASE = dict(vocab_size=48, d_model=32, n_layers=2, telemetry_clusters=4)


def _model(**overrides) -> GlassMindLM:
    seed_everything(3)
    return GlassMindLM(ModelConfig(**{**BASE, **overrides})).eval()


def _state_size(state) -> int:
    total = state.local.numel()
    for block in state.blocks:
        total += block.fast.numel() + block.context.numel() + block.semantic.numel()
    if state.memory is not None:
        memory = state.memory
        total += memory.keys.numel() + memory.values.numel()
        total += sum(
            getattr(memory, name).numel()
            for name in ("strength", "age", "usage_count", "read_count",
                         "write_count", "last_read_step", "last_write_step", "occupied")
        )
    return total


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_state_does_not_grow_with_sequence_length(name: str) -> None:
    """Der rekurrente Zustand ist begrenzt – das ist die Kernzusage."""
    model = _model(**CONFIGURATIONS[name])
    sizes = set()
    for length in (8, 32, 128, 512):
        with torch.no_grad():
            _, state = model(torch.randint(0, BASE["vocab_size"], (1, length)))
        sizes.add(_state_size(state))
    assert len(sizes) == 1, f"{name}: Der Zustand wächst mit der Sequenzlänge: {sorted(sizes)}"


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_no_tensor_carries_two_sequence_dimensions(name: str) -> None:
    """Keine Token-zu-Token-Matrix – weder im State Core noch im Speicher.

    Geprüft wird jede einzelne erzeugte Zwischengröße. Eine Aufmerksamkeits-
    matrix hätte zwangsläufig die Form ``[..., T, T]``.
    """
    model = _model(**CONFIGURATIONS[name])
    # Eine Primzahl als Länge: So kann keine andere Dimension zufällig
    # denselben Wert tragen.
    length = 37
    shapes: list[tuple[int, ...]] = []

    class Watch(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            result = func(*args, **(kwargs or {}))
            for item in (result if isinstance(result, (tuple, list)) else [result]):
                if isinstance(item, torch.Tensor):
                    shapes.append(tuple(item.shape))
            return result

    with torch.no_grad(), Watch():
        model(torch.randint(0, BASE["vocab_size"], (1, length)))
    assert shapes, "Es wurde kein einziger Tensor beobachtet"
    quadratic = [shape for shape in shapes if shape.count(length) >= 2]
    assert not quadratic, f"{name}: Token-zu-Token-Formen gefunden: {quadratic[:5]}"


@pytest.mark.parametrize("name", CONFIGURATIONS)
def test_streaming_state_stays_constant_across_calls(name: str) -> None:
    """Auch über viele Fortsetzungen hinweg wächst nichts an."""
    model = _model(**CONFIGURATIONS[name])
    state = None
    sizes = []
    with torch.no_grad():
        for _ in range(5):
            _, state = model(torch.randint(0, BASE["vocab_size"], (1, 40)), state)
            sizes.append(_state_size(state))
    assert len(set(sizes)) == 1, f"{name}: Der Zustand wächst über die Aufrufe: {sizes}"


def test_memory_read_attends_over_slots_not_over_tokens() -> None:
    """Die Abgrenzung im Detail: Die Scores haben Slot-Breite, nicht Sequenzlänge.

    Der Lesevorgang ist strukturell eine Attention-Operation. Entscheidend ist,
    dass ihre Score-Achse die feste Slotzahl ist – sie wird bei doppelter
    Sequenzlänge nicht größer.
    """
    slots = 64
    model = _model(**CONFIGURATIONS["mit_memory"])
    assert model.memory.slots == slots
    widths = set()
    for length in (16, 64):
        events: list = []
        from glassmind.observe import ObservationBus, ObservationMode

        bus = ObservationBus(ObservationMode.TRACE)
        bus.subscribe(events.append)
        with torch.no_grad():
            model(torch.randint(0, BASE["vocab_size"], (1, length)), observer=bus)
        steps = [event for event in events if event.event == "memory_step"]
        assert len(steps) == length          # ein Zugriff je Token
        widths.add(len(steps[-1].payload["slot_strength"]))
        # Gelesen wird stets nur Top-K, unabhängig von der Sequenzlänge.
        assert len(steps[-1].payload["selected_read_slots"]) == model.memory.read_k
    assert widths == {slots}, f"Die Score-Breite hängt an der Sequenzlänge: {widths}"


def test_no_module_implements_self_attention() -> None:
    """Kein Untermodul stammt aus PyTorchs Attention- oder Transformer-Familie."""
    forbidden = (
        torch.nn.MultiheadAttention,
        torch.nn.TransformerEncoder,
        torch.nn.TransformerDecoder,
        torch.nn.TransformerEncoderLayer,
        torch.nn.TransformerDecoderLayer,
        torch.nn.Transformer,
    )
    for name, overrides in CONFIGURATIONS.items():
        model = _model(**overrides)
        found = [
            type(module).__name__
            for module in model.modules()
            if isinstance(module, forbidden)
        ]
        assert not found, f"{name}: Transformer-Module gefunden: {found}"


def test_no_attention_kernels_are_dispatched() -> None:
    """Auch keine funktionale Attention über ATen-Operationen."""
    counter: Counter[str] = Counter()

    class Count(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            counter[str(func)] += 1
            return func(*args, **(kwargs or {}))

    model = _model(**CONFIGURATIONS["mit_memory"])
    with torch.no_grad(), Count():
        model(torch.randint(0, BASE["vocab_size"], (1, 24)))
    attention = [name for name in counter if any(
        marker in name.lower()
        for marker in ("attention", "scaled_dot_product", "_flash", "_efficient_attention")
    )]
    assert not attention, f"Attention-Kernel aufgerufen: {attention}"


def test_memory_cost_is_linear_in_sequence_length() -> None:
    """O(n) statt O(n²): doppelte Länge, ungefähr doppelte Operationszahl."""
    model = _model(**CONFIGURATIONS["mit_memory"])
    counts = {}
    for length in (16, 32):
        counter: Counter[str] = Counter()

        class Count(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                counter[str(func)] += 1
                return func(*args, **(kwargs or {}))

        with torch.no_grad(), Count():
            model(torch.randint(0, BASE["vocab_size"], (1, length)))
        counts[length] = sum(counter.values())
    ratio = counts[32] / counts[16]
    # Bei quadratischem Aufwand läge das Verhältnis nahe vier statt nahe zwei.
    assert 1.7 < ratio < 2.3, f"Aufwand skaliert nicht linear: Verhältnis {ratio:.2f}"
