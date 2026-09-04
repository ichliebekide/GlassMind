#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from glassmind.data import ByteTokenizer, DEFAULT_TINY_CORPUS, TextChunkDataset
from glassmind.model import GlassMindLM, ModelConfig
from glassmind.observe import ObservationBus, ObservationMode
from glassmind.training.checkpoint import load_checkpoint, save_checkpoint
from glassmind.training.run import RunDirectory
from glassmind.training.trainer import TrainingConfig, train_steps
from glassmind.utils.device import autocast_context, detect_device
from glassmind.utils.reproducibility import environment_metadata, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Trainiert eine winzige GlassMind-Konfiguration")
    parser.add_argument("--config", type=Path, default=Path("configs/tiny.json"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--resume", type=Path, help="Training aus einem backendunabhängigen Checkpoint fortsetzen")
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    args = parser.parse_args()

    raw = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = ModelConfig.from_dict(raw["model"])
    training_values = dict(raw["training"])
    if args.steps is not None:
        training_values["steps"] = args.steps
    training_config = TrainingConfig(**training_values)
    seed_everything(training_config.seed)
    capabilities = detect_device(args.device, args.precision)
    tokenizer = ByteTokenizer()
    dataset_config = raw["dataset"]
    dataset = TextChunkDataset(
        DEFAULT_TINY_CORPUS * 8,
        tokenizer,
        sequence_length=int(dataset_config["sequence_length"]),
        stride=int(dataset_config["sequence_length"]),
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    start_step = 0
    resume_metadata = None
    if args.resume:
        model, tokenizer, resume_metadata = load_checkpoint(args.resume, device=capabilities.torch_device)
        model_config = model.config
        start_step = int(resume_metadata.get("step", 0))
    else:
        model = GlassMindLM(model_config).to(capabilities.torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_config.learning_rate, weight_decay=training_config.weight_decay)
    if resume_metadata and resume_metadata.get("optimizer_state"):
        optimizer.load_state_dict(resume_metadata["optimizer_state"])
    run = RunDirectory(args.run_root, prefix="tiny")
    run.write_json(
        "config.json",
        {
            **raw,
            "model": model_config.to_dict(),
            "training": training_config.to_dict(),
            "resume_from": str(args.resume) if args.resume else None,
            "start_step": start_step,
        },
    )
    run.write_json("environment.json", environment_metadata(capabilities, seed=training_config.seed))
    run.log(f"[Start] Lauf={run.path}  Backend={capabilities.backend}  Precision={capabilities.precision}  Parameter={model.parameter_count:,}")
    try:
        metrics = train_steps(model, loader, training_config, capabilities, logger=run, optimizer=optimizer)
        model.eval()
        summary_events = []
        summary_bus = ObservationBus(ObservationMode.SUMMARY)
        summary_bus.subscribe(summary_events.append)
        sample_inputs, _ = next(iter(loader))
        with torch.inference_mode(), autocast_context(capabilities):
            model(sample_inputs.to(capabilities.torch_device), observer=summary_bus)
        state_lines = []
        for event in summary_events:
            run.metric(event.to_dict())
            payload = event.payload
            state_lines.append(
                f"{event.layer_id}: "
                + ", ".join(
                    f"{name} norm={payload[name]['norm']:.3f} dead={payload[name]['sparsity']:.1%} gesättigt={payload[name]['saturation']:.1%}"
                    for name in ("fast", "context", "semantic")
                )
            )
        state_statistics = "  \n".join(state_lines)
        run.log(f"[state] {state_lines[-1] if state_lines else 'keine Statistik'}")
        checkpoint = run.checkpoints / "final.pt"
        save_checkpoint(
            checkpoint,
            model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            step=start_step + training_config.steps,
            extra={"dataset": dataset_config},
        )
        run.write_summary(
            {
                "result": "PASS",
                "architecture": "glassmind_selective_recurrent_v1",
                "parameter_count": model.parameter_count,
                "state_statistics": state_statistics,
                **metrics,
            }
        )
        run.log(f"[PASS] Checkpoint={checkpoint}")
    except Exception as exc:
        run.write_summary(
            {
                "result": "FAIL",
                "architecture": "glassmind_selective_recurrent_v1",
                "parameter_count": model.parameter_count,
                "warnings": f"Training abgebrochen: {type(exc).__name__}: {exc}",
            }
        )
        raise


if __name__ == "__main__":
    main()
