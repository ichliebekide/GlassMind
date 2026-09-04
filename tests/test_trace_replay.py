from pathlib import Path

import torch

from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import JSONLRecorder, ObservationBus, ObservationMode
from glassmind.visualize.graph import ReplayTimeline, graph_arrays


def test_trace_is_replayable(tmp_path: Path) -> None:
    model = GlassMindLM(ModelConfig.tiny(vocab_size=32)).eval()
    trace = tmp_path / "trace.jsonl"
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(JSONLRecorder(trace, flush_every=1))
    with torch.no_grad():
        model(torch.tensor([[4, 5, 6, 7]]), observer=bus)
    bus.close()
    timeline = ReplayTimeline.from_trace(trace)
    assert len(timeline) == 4
    arrays = graph_arrays(timeline[0])
    assert arrays["positions"]
    assert arrays["segments"]
    assert all(node["id"].startswith("core.") for node in timeline[0].nodes)
