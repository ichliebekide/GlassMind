from __future__ import annotations

from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F

from glassmind.data.synthetic import make_overfit_batch
from glassmind.data.tokenizer import ByteTokenizer
from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import ObservationBus, ObservationMode
from glassmind.training.checkpoint import load_checkpoint, save_checkpoint
from glassmind.utils.reproducibility import seed_everything


def run_smoke_test() -> dict[str, float | int | bool]:
    seed_everything(123)
    config = ModelConfig.tiny(vocab_size=260)
    model = GlassMindLM(config).cpu().eval()
    inputs = torch.randint(0, config.vocab_size, (2, 13))
    targets = torch.randint(0, config.vocab_size, (2, 13))

    model.train()
    logits, state = model(inputs)
    assert logits.shape == (2, 13, config.vocab_size)
    assert state.position == 13
    loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
    loss.backward()
    missing_gradients = [name for name, parameter in model.named_parameters() if parameter.requires_grad and parameter.grad is None]
    assert not missing_gradients, f"Fehlende Gradienten: {missing_gradients}"
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)

    model.eval()
    with torch.no_grad():
        full_logits, _ = model(inputs)
        stream_state = None
        streamed = []
        for index in range(inputs.shape[1]):
            token_logits, stream_state = model.step(inputs[:, index], stream_state)
            streamed.append(token_logits)
        stream_logits = torch.stack(streamed, dim=1)
    max_stream_error = float((full_logits - stream_logits).abs().max())
    assert torch.allclose(full_logits, stream_logits, atol=2e-5, rtol=2e-5), max_stream_error

    trace_bus = ObservationBus(ObservationMode.TRACE)
    with torch.no_grad():
        observed_logits, _ = model(inputs, observer=trace_bus)
    max_observer_error = float((full_logits - observed_logits).abs().max())
    assert torch.equal(full_logits, observed_logits), max_observer_error
    assert trace_bus.events_emitted == config.n_layers * inputs.shape[1] + inputs.shape[1]

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "checkpoint.pt"
        save_checkpoint(path, model, tokenizer=ByteTokenizer(), step=7)
        restored, _, metadata = load_checkpoint(path)
        restored.eval()
        with torch.no_grad():
            restored_logits, _ = restored(inputs)
        assert torch.equal(full_logits, restored_logits)
        assert metadata["step"] == 7

    seed_everything(777)
    deterministic_a = GlassMindLM(config)
    seed_everything(777)
    deterministic_b = GlassMindLM(config)
    assert all(torch.equal(a, b) for a, b in zip(deterministic_a.state_dict().values(), deterministic_b.state_dict().values(), strict=True))
    return {
        "loss": float(loss.detach()),
        "parameters": model.parameter_count,
        "max_stream_error": max_stream_error,
        "max_observer_error": max_observer_error,
        "trace_events": trace_bus.events_emitted,
        "finite": True,
    }


def run_overfit_test(*, steps: int = 240) -> dict[str, float | int | bool]:
    seed_everything(31)
    config = ModelConfig(vocab_size=32, d_model=32, n_layers=1, telemetry_clusters=4)
    model = GlassMindLM(config).cpu().train()
    inputs, targets = make_overfit_batch(vocab_size=config.vocab_size, sequence_length=20, batch_size=6)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    with torch.no_grad():
        initial_logits, _ = model(inputs)
        initial_loss = float(F.cross_entropy(initial_logits.reshape(-1, config.vocab_size), targets.reshape(-1)))
    final_loss = initial_loss
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    with torch.no_grad():
        logits, _ = model(inputs)
        accuracy = float((logits.argmax(dim=-1) == targets).float().mean())
    passed = final_loss < initial_loss * 0.25 and accuracy >= 0.95
    assert passed, f"Tiny-Overfit fehlgeschlagen: initial={initial_loss:.4f}, final={final_loss:.4f}, Genauigkeit={accuracy:.3f}"
    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "accuracy": accuracy,
        "steps": steps,
        "passed": passed,
    }

