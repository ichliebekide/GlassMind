#!/usr/bin/env python3
"""Funktionstest des bounded sparse external memory (Milestone 3).

Prüft die Zusagen des Speichers an einem frisch initialisierten Modell, ohne
Training: Sparsity, Begrenztheit, Belegung, Ersetzung, Ablation, Streaming-
Gleichheit und Replay. Ob der Speicher die Aufgabenqualität verbessert, ist
eine andere Frage – die beantwortet ``scripts/memory_study.py``.

Der Test schlägt laut fehl und liefert einen Exit-Code ungleich null.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import torch

from glassmind.model import GlassMindLM, ModelConfig
from glassmind.model.memory import memory_utilisation
from glassmind.observe import JSONLRecorder, ObservationBus, ObservationMode
from glassmind.utils.device import detect_device
from glassmind.utils.reproducibility import seed_everything
from glassmind.visualize.graph import ReplayTimeline, memory_arrays

SLOTS, READ_K, WRITE_K = 16, 2, 1


def run_memory_test() -> dict[str, object]:
    seed_everything(37)
    capabilities = detect_device("cpu")
    config = ModelConfig(
        vocab_size=48, d_model=32, n_layers=2, telemetry_clusters=4, state_interactions=True,
        memory_slots=SLOTS, memory_width=32, memory_key_dim=16,
        memory_read_k=READ_K, memory_write_k=WRITE_K, memory_track_usage=True,
    )
    model = GlassMindLM(config).to(capabilities.torch_device).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 40))

    # --- Sparsity und Begrenztheit ------------------------------------
    events: list = []
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(events.append)
    with torch.no_grad():
        logits, state = model(tokens, observer=bus)
    bus.close()
    steps = [event for event in events if event.event == "memory_step"]
    assert len(steps) == tokens.shape[1], "Es fehlt ein Speicherereignis je Token"
    for event in steps:
        assert len(event.payload["selected_read_slots"]) == READ_K, "Lesezugriff ist nicht sparse"
        assert len(event.payload["selected_write_slots"]) == WRITE_K, "Schreibzugriff ist nicht sparse"
    assert state.memory.slots == SLOTS, "Der Speicher ist über seine Grenze gewachsen"
    assert torch.isfinite(logits).all(), "Nicht-endliche Logits"

    utilisation = memory_utilisation(state.memory)
    assert utilisation["occupied_slots"] == SLOTS, "Der Speicher wurde nicht vollständig gefüllt"
    replacements = sum(len(event.payload["replacement_events"]) for event in steps)
    assert replacements == tokens.shape[1] - SLOTS, (
        "Ersetzungen dürfen erst nach dem Füllen auftreten"
    )

    # --- Ablation ------------------------------------------------------
    with torch.no_grad():
        plain, _ = model(tokens, disable_memory=True)
        no_read, read_state = model(tokens, disable_memory_read=True)
        no_write, write_state = model(tokens, disable_memory_write=True)
    assert not torch.equal(logits, plain), "Der Speicher hat keinerlei Wirkung auf die Ausgabe"
    assert float(read_state.memory.read_count.sum()) == 0.0, "Lesen wurde nicht abgeschaltet"
    assert float(write_state.memory.occupied.sum()) == 0.0, "Schreiben wurde nicht abgeschaltet"
    ablated_slot = int(steps[-1].payload["selected_read_slots"][0])
    with torch.no_grad():
        without_slot, _ = model(tokens, ablate_memory_slots=[ablated_slot])
    slot_difference = float((without_slot - logits).abs().max())

    # --- Streaming gegen Sequenz ---------------------------------------
    with torch.no_grad():
        stream_state = None
        streamed = []
        for index in range(tokens.shape[1]):
            token_logits, stream_state = model.step(tokens[:, index], stream_state)
            streamed.append(token_logits)
    stream_error = float((logits - torch.stack(streamed, dim=1)).abs().max())
    assert stream_error < 2e-5, f"Streaming weicht ab: {stream_error}"

    # --- Beobachtung bleibt neutral ------------------------------------
    with torch.no_grad():
        unobserved, _ = model(tokens)
    assert torch.equal(logits, unobserved), "Beobachtung verändert das Ergebnis"

    # --- Replay ohne Modell ---------------------------------------------
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "memory.jsonl"
        bus = ObservationBus(ObservationMode.TRACE)
        bus.subscribe(JSONLRecorder(trace, flush_every=1))
        with torch.no_grad():
            model(tokens, observer=bus)
        bus.close()
        timeline = ReplayTimeline.from_trace(trace)
        frame = timeline[-1]
        assert frame.memory is not None, "Der Trace enthält keinen Speicherzustand"
        arrays = memory_arrays(frame.memory)
        assert arrays["slots"] == SLOTS
        assert sum(arrays["occupied"]) == SLOTS
        trace_bytes = trace.stat().st_size

    return {
        "slots": SLOTS,
        "read_top_k": READ_K,
        "write_top_k": WRITE_K,
        "tokens": int(tokens.shape[1]),
        "occupied_slots": utilisation["occupied_slots"],
        "total_reads": utilisation["total_reads"],
        "total_writes": utilisation["total_writes"],
        "replacement_events": replacements,
        "slot_ablation_logit_difference": slot_difference,
        "max_stream_error": stream_error,
        "trace_bytes": trace_bytes,
        "passed": True,
    }


if __name__ == "__main__":
    result = run_memory_test()
    print("[PASS] Memory-Funktionstest bestanden")
    print(json.dumps(result, indent=2, ensure_ascii=False))
