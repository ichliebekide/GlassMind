from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from glassmind.analysis import ClusterAnalyzer, StateMetricsAnalyzer, ablation_comparison
from glassmind.data import (
    generate_associative_recall_batch,
    generate_selective_copy_batch,
)
from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import JSONLRecorder, ObservationBus, ObservationMode
from glassmind.visualize.graph import ReplayTimeline, graph_arrays


def _task_model() -> GlassMindLM:
    torch.manual_seed(91)
    return GlassMindLM(
        ModelConfig(
            vocab_size=64,
            d_model=32,
            n_layers=1,
            telemetry_clusters=4,
            state_interactions=True,
        )
    )


def test_associative_and_selective_generators_are_reproducible() -> None:
    for generator, kwargs in (
        (generate_associative_recall_batch, {"associations": 3}),
        (generate_selective_copy_batch, {"items": 2}),
    ):
        first = generator(batch_size=4, distance=64, seed=55, **kwargs)
        second = generator(batch_size=4, distance=64, seed=55, **kwargs)
        assert torch.equal(first.input_ids, second.input_ids)
        assert torch.equal(first.targets, second.targets)
        assert torch.equal(first.loss_mask, second.loss_mask)
        assert int(first.loss_mask.sum()) == first.answer_count * 4


def test_state_tasks_can_be_overfit_together() -> None:
    model = _task_model().train()
    associative = generate_associative_recall_batch(
        batch_size=8, distance=8, associations=2, seed=7
    )
    selective = generate_selective_copy_batch(
        batch_size=8, distance=8, items=2, seed=9
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for step in range(260):
        batch = associative if step % 2 == 0 else selective
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.input_ids)
        loss = F.cross_entropy(logits[batch.loss_mask], batch.targets[batch.loss_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for batch in (associative, selective):
            logits, _ = model(batch.input_ids)
            accuracy = (
                logits[batch.loss_mask].argmax(dim=-1) == batch.targets[batch.loss_mask]
            ).float().mean()
            assert float(accuracy) >= 0.95


def test_state_ablation_reports_changes_and_streams_consistently() -> None:
    model = _task_model().eval()
    batch = generate_associative_recall_batch(
        batch_size=3, distance=16, associations=2, seed=77
    )
    comparison = ablation_comparison(model, batch, ("semantic",))
    assert comparison["logit_rms_difference"] > 0.0
    with torch.no_grad():
        full, final_state = model(batch.input_ids, ablate_states=("semantic",))
        assert torch.count_nonzero(final_state.blocks[0].semantic) == 0
        state = None
        streamed = []
        for index in range(batch.input_ids.shape[1]):
            logits, state = model.step(
                batch.input_ids[:, index], state, ablate_states=("semantic",)
            )
            streamed.append(logits)
        assert torch.allclose(full, torch.stack(streamed, dim=1), atol=2e-5, rtol=2e-5)


def test_cluster_metrics_and_replay_share_enriched_nodes(tmp_path: Path) -> None:
    model = _task_model().eval()
    batch = generate_selective_copy_batch(batch_size=1, distance=12, items=2, seed=81)
    trace = tmp_path / "state-trace.jsonl"
    bus = ObservationBus(ObservationMode.TRACE)
    recorder = JSONLRecorder(trace, flush_every=1)
    clusters = ClusterAnalyzer()
    states = StateMetricsAnalyzer()
    events = []
    bus.subscribe(recorder)
    bus.subscribe(clusters)
    bus.subscribe(states)
    bus.subscribe(events.append)
    with torch.no_grad():
        unobserved, _ = model(batch.input_ids)
        observed, _ = model(batch.input_ids, observer=bus)
    bus.close()
    assert torch.equal(unobserved, observed)
    network_event = next(event for event in events if event.event == "network_step")
    cluster_node = next(
        node for node in network_event.payload["nodes"] if ".cluster." in node["id"]
    )
    components = cluster_node["components"]
    for field in (
        "state_norm",
        "delta_norm",
        "persistence_duration",
        "reactivation",
        "update_gate_activity",
        "forget_activity",
        "information_flow",
    ):
        assert field in components
    assert cluster_node["cluster_statistics"]["activation_count"] >= 0
    assert clusters.summaries()
    assert set(states.summaries()) == {"fast", "context", "semantic"}
    timeline = ReplayTimeline.from_trace(trace)
    arrays = graph_arrays(timeline[-1])
    assert len(arrays["deltas"]) == len(arrays["ids"])
    assert len(arrays["persistence"]) == len(arrays["ids"])


def test_observation_off_is_exactly_neutral() -> None:
    model = _task_model().eval()
    tokens = torch.randint(0, 64, (2, 19))
    with torch.no_grad():
        baseline, _ = model(tokens)
        disabled, _ = model(tokens, observer=ObservationBus(ObservationMode.OFF))
    assert torch.equal(baseline, disabled)
