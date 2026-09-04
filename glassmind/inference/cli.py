from __future__ import annotations

import argparse
from pathlib import Path

import torch

from glassmind.data.tokenizer import ByteTokenizer
from glassmind.model.config import ModelConfig
from glassmind.model.lm import GlassMindLM
from glassmind.observe.bus import ObservationBus, ObservationMode
from glassmind.observe.recorder import JSONLRecorder
from glassmind.training.checkpoint import load_checkpoint
from glassmind.utils.device import autocast_context, detect_device
from glassmind.utils.reproducibility import seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Streaming-Inferenz mit GlassMind")
    parser.add_argument("prompt", help="Eingabetext")
    parser.add_argument("--checkpoint", type=Path, help="Trainierter Checkpoint; ohne Angabe wird ein deterministisches untrainiertes Mini-Modell verwendet")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--precision", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trace", action="store_true", help="Detaillierte Telemetrie aktivieren")
    parser.add_argument("--record", type=Path, help="Trace als JSONL aufzeichnen")
    parser.add_argument(
        "--ablate-state",
        action="append",
        choices=["fast", "context", "semantic"],
        default=[],
        help="Analysemodus: Zustand deaktivieren; Option kann wiederholt werden",
    )
    memory = parser.add_argument_group("Externes Memory (Milestone 3)")
    memory.add_argument("--disable-memory", action="store_true",
                        help="Analysemodus: Speicher vollständig überspringen")
    memory.add_argument("--disable-memory-read", action="store_true",
                        help="Analysemodus: nur Lesezugriffe abschalten")
    memory.add_argument("--disable-memory-write", action="store_true",
                        help="Analysemodus: nur Schreibzugriffe abschalten")
    memory.add_argument("--ablate-memory-slot", action="append", type=int, default=[],
                        metavar="SLOT",
                        help="Analysemodus: einzelne Speicherzelle abschalten; wiederholbar")
    memory.add_argument("--memory-intervention", action="append", default=[],
                        metavar="SLOT:OPERATION",
                        help="Analysemodus: clear, freeze, mute_read oder mute_write auf eine Zelle")
    return parser


def parse_interventions(entries: list[str]) -> dict[int, str]:
    """Wandelt ``17:freeze`` in ``{17: "freeze"}``."""
    from glassmind.model.memory import INTERVENTIONS

    parsed: dict[int, str] = {}
    for entry in entries:
        slot, _, operation = entry.partition(":")
        if not operation:
            raise SystemExit(
                f"--memory-intervention erwartet SLOT:OPERATION, erhalten: {entry!r}"
            )
        if operation not in INTERVENTIONS:
            raise SystemExit(
                f"Unbekannte Intervention {operation!r}; erlaubt: {', '.join(INTERVENTIONS)}"
            )
        try:
            parsed[int(slot)] = operation
        except ValueError as exc:
            raise SystemExit(f"Slotnummer {slot!r} ist keine ganze Zahl") from exc
    return parsed


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(17)
    capabilities = detect_device(args.device, args.precision)
    if args.checkpoint:
        model, tokenizer, _ = load_checkpoint(args.checkpoint, device=capabilities.torch_device)
    else:
        tokenizer = ByteTokenizer()
        model = GlassMindLM(ModelConfig.tiny(tokenizer.vocab_size)).to(capabilities.torch_device)
        print("[Hinweis] Kein Checkpoint angegeben; Ausgabe stammt aus einem untrainierten Mini-Modell.")
    model.eval()
    mode = ObservationMode.TRACE if (args.trace or args.record) else ObservationMode.OFF
    observer = ObservationBus(mode)
    recorder = JSONLRecorder(args.record) if args.record else None
    if recorder:
        observer.subscribe(recorder)
    memory_options: dict[str, object] = {}
    if model.memory is None:
        rejected = [
            name for name, active in (
                ("--disable-memory", args.disable_memory),
                ("--disable-memory-read", args.disable_memory_read),
                ("--disable-memory-write", args.disable_memory_write),
                ("--ablate-memory-slot", bool(args.ablate_memory_slot)),
                ("--memory-intervention", bool(args.memory_intervention)),
            ) if active
        ]
        if rejected:
            # Lieber ein klarer Hinweis als eine wirkungslose Option.
            print(f"[Hinweis] Dieses Modell besitzt kein externes Memory; "
                  f"{', '.join(rejected)} bleibt ohne Wirkung.")
    else:
        memory_options = {
            "disable_memory": args.disable_memory,
            "disable_memory_read": args.disable_memory_read,
            "disable_memory_write": args.disable_memory_write,
            "ablate_memory_slots": args.ablate_memory_slot or None,
            "memory_interventions": parse_interventions(args.memory_intervention) or None,
        }
    prompt = torch.tensor([tokenizer.encode(args.prompt, add_bos=True)], dtype=torch.long, device=capabilities.torch_device)
    try:
        with torch.inference_mode(), autocast_context(capabilities):
            output = model.generate(
                prompt,
                args.max_new_tokens,
                temperature=args.temperature,
                observer=observer,
                ablate_states=args.ablate_state,
                **memory_options,
            )
    finally:
        observer.close()
    print(tokenizer.decode(output[0].cpu().tolist()))
    print(f"[Gerät] {capabilities.backend}/{capabilities.precision}  Parameter={model.parameter_count:,}")
    if args.record:
        print(f"[Trace] {args.record}  Ereignisse={observer.events_emitted}")
    if model.memory is not None and any(memory_options.values()):
        active = [name.replace("_", "-") for name, value in memory_options.items() if value]
        print(f"[Memory-Analyse] aktiv: {', '.join(active)}")
    if args.ablate_state:
        print(f"[Ablation] deaktiviert={','.join(args.ablate_state)}")


if __name__ == "__main__":
    main()
