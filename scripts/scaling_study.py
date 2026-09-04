#!/usr/bin/env python3
"""Milestone 4: wie skaliert GlassMind auf echten Sprachdaten?

Das Ziel ist ausdrücklich *nicht*, eine möglichst große Parameterzahl zu
erreichen, sondern die Skalierungskurve zu vermessen.

Zwei Studien werden streng getrennt gehalten:

``fixed``     Fixed-Token Scaling – jede Größenklasse bekommt exakt dieselbe
              Zahl Trainingstoken, dieselbe Datenquelle, dieselbe effektive
              Batchgröße und dieselbe Evaluation. Nur so ist der Vergleich
              zwischen den Größen ein kontrolliertes Experiment.
``capacity``  Capacity-Aware Scaling – größere Modelle bekommen ein größeres
              Budget. Das ist die realistischere, aber nicht mehr kontrollierte
              Frage: was leistet eine Größe, wenn man sie angemessen füttert?

Weitere Abschnitte:

``benchmark``     kurzer Laufzeittest, um das Tokenbudget zu wählen
``sequence``      Streaming-Latenz in Fenstern über wachsenden Kontext
``precision``     FP32/FP16/BF16/Mixed je Größenklasse, neu gemessen
``quantization``  INT8/INT4-Ablage: Größe, Ladezeit, Qualität, Geschwindigkeit
``profile``       Übergang dispatch-bound -> compute-bound
``trace``         reproduzierbarer VisPy-Trace eines trainierten Sprachmodells
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from typing import Any, Sequence

import torch

from glassmind.data.corpus import CORPORA, TokenStream, prepare_corpus, random_batches
from glassmind.data.tokenizer import ByteTokenizer
from glassmind.model import GlassMindLM
from glassmind.model.sizes import (
    SIZE_CLASSES,
    TARGET_EFFECTIVE_BATCH,
    SizeClass,
    gradient_accumulation,
    size_class,
)
from glassmind.observe import ObservationBus, ObservationMode
from glassmind.observe.recorder import JSONLRecorder
from glassmind.precision.policy import PrecisionPolicy, experimental_profile
from glassmind.training.checkpoint import load_checkpoint, save_checkpoint
from glassmind.training.language import (
    LanguageTrainingConfig,
    ablation_logit_effect,
    corpus_vocabulary,
    evaluate_language,
    generation_quality,
    generation_samples,
    state_activity_profile,
    state_specialisation,
    state_statistics,
    steps_for_budget,
    top1_accuracy,
    train_language,
)
from glassmind.utils.device import (
    autocast_context,
    detect_device,
    peak_memory_bytes,
    reset_peak_memory,
    synchronize,
)
from glassmind.utils.reproducibility import environment_metadata, seed_everything

#: Feste Prompts je Korpus. Sie ändern sich zwischen Läufen nicht, damit
#: Generationsproben über Größenklassen hinweg vergleichbar bleiben.
PROMPTS: dict[str, tuple[str, ...]] = {
    "tinystories": (
        "Once upon a time, there was a little girl named Lily. She",
        "Tom and Sara went to the park. They saw a",
        "The dog was very hungry, so he",
        "One day, a small cat found a big red ball. The cat",
    ),
    "wikitext103": (
        " The city of Bremen is located in",
        " In 1912 , the company began producing",
        " The species was first described by",
        " The album received mixed reviews from",
    ),
}


class Console:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path

    def log(self, message: str) -> None:
        print(message, flush=True)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")

    def metric(self, data: dict[str, Any]) -> None:
        pass


def process_rss_bytes() -> int:
    """Arbeitsspeicher des Prozesses – ohne zusätzliche Abhängigkeit."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def prompts_for(corpus: str) -> tuple[str, ...]:
    return PROMPTS.get(corpus, PROMPTS["tinystories"])


# ----------------------------------------------------------------------
# Messhilfen
# ----------------------------------------------------------------------

def throughput(model: GlassMindLM, capabilities, *, length: int, batch: int = 1,
               repeats: int = 3, rounds: int = 3) -> float:
    tokens = torch.randint(0, model.config.vocab_size, (batch, length),
                           device=capabilities.torch_device)
    model.eval()
    with torch.inference_mode(), autocast_context(capabilities):
        for _ in range(2):
            model(tokens)
        synchronize(capabilities)
        samples = []
        for _ in range(rounds):
            started = time.perf_counter()
            for _ in range(repeats):
                model(tokens)
            synchronize(capabilities)
            samples.append(tokens.numel() * repeats / (time.perf_counter() - started))
    return statistics.median(samples)


def streaming_latency(model: GlassMindLM, capabilities, *, steps: int = 128,
                      warmup: int = 16, state=None, rounds: int = 3):
    """Median-Latenz je Token im Streaming – und der weitergeführte Zustand."""
    token = torch.zeros(1, dtype=torch.long, device=capabilities.torch_device)
    model.eval()
    with torch.inference_mode(), autocast_context(capabilities):
        for _ in range(warmup):
            _, state = model.step(token, state)
        synchronize(capabilities)
        samples = []
        for _ in range(rounds):
            started = time.perf_counter()
            for _ in range(steps):
                _, state = model.step(token, state)
            synchronize(capabilities)
            samples.append((time.perf_counter() - started) * 1000 / steps)
    return statistics.median(samples), state


def observation_overhead(model: GlassMindLM, capabilities, *, length: int = 256) -> dict[str, float]:
    plain = throughput(model, capabilities, length=length)
    tokens = torch.randint(0, model.config.vocab_size, (1, length),
                           device=capabilities.torch_device)
    with torch.inference_mode(), autocast_context(capabilities):
        model(tokens, observer=ObservationBus(ObservationMode.SUMMARY))
        synchronize(capabilities)
        started = time.perf_counter()
        for _ in range(3):
            model(tokens, observer=ObservationBus(ObservationMode.SUMMARY))
        synchronize(capabilities)
        summary = tokens.numel() * 3 / (time.perf_counter() - started)
        started = time.perf_counter()
        for _ in range(3):
            model(tokens, observer=ObservationBus(ObservationMode.TRACE))
        synchronize(capabilities)
        traced = tokens.numel() * 3 / (time.perf_counter() - started)
    return {
        "plain_tokens_per_second": plain,
        "summary_tokens_per_second": summary,
        "trace_tokens_per_second": traced,
        "summary_overhead_percent": 100 * (1 - summary / plain),
        "trace_overhead_percent": 100 * (1 - traced / plain),
    }


def run_documentation(
    size: SizeClass, config: LanguageTrainingConfig, corpus: TokenStream,
    validation: TokenStream, capabilities, model: GlassMindLM, *, study: str,
) -> dict[str, Any]:
    """Der vollständige Dokumentationsblock, den Milestone 4 je Lauf verlangt."""
    spec = corpus.metadata["spec"]
    return {
        "studie": study,
        "korpus": spec["name"],
        "revision": corpus.metadata["revision"],
        "lizenz": spec["license"],
        "quelle": spec["url"],
        "trainingssplit": spec["split"],
        "trainingsdateien": spec["files"],
        "korpus_tokens_gesamt": corpus.metadata["tokens"],
        "verwendete_trainingstokens": config.token_budget,
        "validierungssplit": validation.metadata["spec"]["split"],
        "validierungstokens_gesamt": validation.metadata["tokens"],
        "tokenizer": corpus.metadata["tokenizer"],
        "sequenzlaenge": config.sequence_length,
        "mikro_batchgroesse": config.batch_size,
        "gradient_accumulation": config.gradient_accumulation,
        "effektive_batchgroesse": config.effective_batch_size,
        "precision": capabilities.precision,
        "backend": capabilities.backend,
        "seed": config.seed,
        "lernrate": config.learning_rate,
        "warmup_schritte": config.warmup_steps,
        "schritte": config.steps,
        "modellkonfiguration": model.config.to_dict(),
        "parameterzahl": model.parameter_count,
    }


# ----------------------------------------------------------------------
# Vollmessung einer Größenklasse
# ----------------------------------------------------------------------

def measure_size_class(
    size: SizeClass, args, capabilities, corpora: dict[str, Any], console: Console,
    *, steps: int, study: str,
) -> dict[str, Any]:
    seed_everything(args.seed)
    checkpoint = args.checkpoint_dir / f"m4-{study}-{args.corpus}-{size.name}.pt"
    # Mit ``--reuse-checkpoints`` wird ein bereits trainiertes Modell geladen
    # statt neu trainiert. Das ist die Rettungsleine, wenn nur die Auswertung
    # verlorenging: Die Messung dauert Minuten, das Training Stunden.
    reused = args.reuse_checkpoints and checkpoint.exists()
    if reused:
        model, _, meta = load_checkpoint(checkpoint, device=capabilities.torch_device)
        console.log(f"  [wiederverwendet] {checkpoint.name} "
                    f"({meta.get('step', 0)} Schritte trainiert)")
    else:
        model = GlassMindLM(size.config()).to(capabilities.torch_device)
    parameters = model.parameter_count
    accumulation = gradient_accumulation(size)
    config = LanguageTrainingConfig(
        steps=steps,
        batch_size=size.batch_size,
        sequence_length=size.sequence_length,
        gradient_accumulation=accumulation,
        learning_rate=size.learning_rate,
        warmup_steps=max(10, steps // 10),
        log_every=max(steps // 6, 1),
        seed=args.seed,
    )
    console.log(
        f"\n=== [{study}] {size.name}  d_model={size.d_model}  Layer={size.n_layers}  "
        f"Parameter={parameters:,}  {steps} Schritte x {config.tokens_per_step:,} Token "
        f"= {config.token_budget:,} Token ==="
    )
    documentation = run_documentation(
        size, config, corpora["train"], corpora["validation"], capabilities, model, study=study
    )
    if reused:
        # Die Trainingskennzahlen dieses Laufs sind nicht rekonstruierbar; das
        # wird ausgewiesen statt mit Platzhaltern gefüllt.
        training = {
            "reused_checkpoint": str(checkpoint),
            "steps_completed": int(meta.get("step", 0)),
            "diverged": False,
            "final_loss": None, "best_loss": None, "final_bits_per_byte": None,
            "tokens_per_second": None, "seen_tokens": config.token_budget,
            "training_seconds": None, "peak_memory_bytes": 0,
            "grad_norm_mean": None, "grad_norm_median": None,
            "grad_norm_max": None, "grad_norm_final": None,
            "grad_clip_hit_rate": None, "curve": [],
            "config": config.to_dict(),
        }
    else:
        training = train_language(model, corpora["train"], config, capabilities,
                                  logger=console)
    record: dict[str, Any] = {
        "size_class": size.name,
        "study": study,
        "target": size.target,
        "d_model": size.d_model,
        "n_layers": size.n_layers,
        "parameter_count": parameters,
        "documentation": documentation,
        "training": training,
        "nan_inf": training["diverged"],
    }
    if training["diverged"]:
        console.log(f"  {size.name}: Training divergiert – gilt als Messergebnis")
        del model
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
        return record

    eval_batch = max(1, size.batch_size)
    record["validation"] = {
        "own_corpus": evaluate_language(
            model, corpora["validation"], capabilities,
            sequence_length=size.sequence_length, batch_size=eval_batch,
            batches=args.eval_batches,
        ),
        # Gegencheck auf dem jeweils anderen Korpus. Er wird getrennt
        # ausgewiesen und nicht mit der eigenen Validierung vermischt.
        "cross_corpus": evaluate_language(
            model, corpora["cross_validation"], capabilities,
            sequence_length=size.sequence_length, batch_size=eval_batch,
            batches=args.eval_batches,
        ),
        "top1": top1_accuracy(
            model, corpora["validation"], capabilities,
            sequence_length=size.sequence_length, batch_size=eval_batch,
            batches=max(1, args.eval_batches // 2),
        ),
    }
    reset_peak_memory(capabilities)
    inference = throughput(model, capabilities, length=size.sequence_length)
    latency, _ = streaming_latency(model, capabilities)
    record["inference"] = {
        "tokens_per_second": inference,
        "streaming_ms_per_token": latency,
        "peak_memory_bytes": peak_memory_bytes(capabilities) or 0,
        "process_rss_bytes": process_rss_bytes(),
        **observation_overhead(model, capabilities),
    }
    record["state"] = {
        "statistics": state_statistics(
            model, corpora["validation"], capabilities, sequence_length=size.sequence_length
        ),
        "specialisation": state_specialisation(
            model, corpora["validation"], capabilities,
            sequence_length=size.sequence_length, batch_size=eval_batch,
            batches=args.ablation_batches,
        ),
        "logit_effect": ablation_logit_effect(
            model, corpora["validation"], capabilities, sequence_length=size.sequence_length
        ),
        "activity": state_activity_profile(
            model, corpora["validation"], capabilities,
            sequence_length=min(256, size.sequence_length)
        ),
    }
    record["generation"] = generation_samples(
        model, ByteTokenizer(), capabilities, prompts_for(args.corpus),
        max_new_tokens=args.generate_tokens, seed=args.seed,
    )
    record["generation_quality"] = {
        key: generation_quality(record["generation"], corpora["vocabulary"], key=key)
        for key in ("greedy", "sampled")
    }

    if not reused:
        save_checkpoint(checkpoint, model, tokenizer=ByteTokenizer(),
                        step=training["steps_completed"],
                        extra={"milestone": "4", "study": study,
                               "size_class": size.name, "corpus": args.corpus})
    record["checkpoint"] = {
        "path": str(checkpoint),
        "bytes": checkpoint.stat().st_size,
        "bytes_per_parameter": checkpoint.stat().st_size / max(parameters, 1),
    }

    own = record["validation"]["own_corpus"]
    cross = record["validation"]["cross_corpus"]
    quality = record["generation_quality"]["sampled"]
    specialisation = record["state"]["specialisation"]
    console.log(
        f"  {size.name}: val-loss={own['loss']:.4f}  ppl={own['perplexity']:.1f}  "
        f"bpb={own['bits_per_byte']:.3f}  top1={record['validation']['top1']['top1_accuracy']:.1%}  "
        f"| Gegenkorpus bpb={cross['bits_per_byte']:.3f}"
    )
    rate = training["tokens_per_second"]
    console.log(
        f"  {size.name}: Training "
        f"{'wiederverwendet' if rate is None else f'{rate:,.0f} tok/s'}  "
        f"Inferenz {inference:,.0f} tok/s  Streaming {latency:.3f} ms/Tok  "
        f"VRAM {(training['peak_memory_bytes'])/1e6:.0f} MB  "
        f"Checkpoint {checkpoint.stat().st_size/1e6:.1f} MB"
    )
    console.log(
        f"  {size.name}: gültige Wörter={quality.get('valid_word_fraction', 0):.1%}  "
        f"distinct-2={quality.get('distinct_2', 0):.2f}  "
        f"Zustände unterscheidbar={specialisation['distinguishable']} "
        f"(Δ-Streuung {specialisation['delta_spread']:.4f})"
    )
    del model
    if capabilities.backend in {"cuda", "rocm"}:
        torch.cuda.empty_cache()
    return record


# ----------------------------------------------------------------------
# Abschnitt: Laufzeitbenchmark zur Budgetwahl
# ----------------------------------------------------------------------

def section_benchmark(args, capabilities, corpora, console) -> dict[str, Any]:
    """Wie schnell trainiert jede Stufe wirklich? Daraus folgt das Budget.

    Ohne diese Messung wäre jedes Tokenbudget geraten. Gemessen wird ein
    vollständiger Optimizer-Schritt inklusive Gradient Accumulation.
    """
    console.log("\n=== Laufzeitbenchmark: wie teuer ist ein Schritt je Größe? ===")
    records = []
    for name in args.classes:
        size = size_class(name)
        seed_everything(args.seed)
        try:
            model = GlassMindLM(size.config()).to(capabilities.torch_device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            records.append({"size_class": size.name, "out_of_memory": True})
            continue
        config = LanguageTrainingConfig(
            steps=args.benchmark_steps, batch_size=size.batch_size,
            sequence_length=size.sequence_length,
            gradient_accumulation=gradient_accumulation(size),
            learning_rate=size.learning_rate, warmup_steps=1,
            log_every=10**9, seed=args.seed,
        )
        reset_peak_memory(capabilities)
        try:
            metrics = train_language(model, corpora["train"], config, capabilities)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            records.append({"size_class": size.name, "out_of_memory": True})
            del model
            continue
        seconds_per_step = metrics["training_seconds"] / max(metrics["steps_completed"], 1)
        entry = {
            "size_class": size.name,
            "parameter_count": model.parameter_count,
            "micro_batch": size.batch_size,
            "gradient_accumulation": config.gradient_accumulation,
            "tokens_per_step": config.tokens_per_step,
            "seconds_per_step": seconds_per_step,
            "training_tokens_per_second": metrics["tokens_per_second"],
            "peak_memory_bytes": metrics["peak_memory_bytes"],
            "diverged": metrics["diverged"],
        }
        for budget in args.budget_candidates:
            entry[f"hours_for_{budget}"] = (
                budget / max(metrics["tokens_per_second"], 1e-9) / 3600
            )
        records.append(entry)
        console.log(
            f"  {size.name:8s} {model.parameter_count:>12,d} Par  "
            f"{seconds_per_step*1000:8.1f} ms/Schritt  "
            f"{metrics['tokens_per_second']:>9,.0f} tok/s  "
            f"VRAM {metrics['peak_memory_bytes']/1e6:7.0f} MB"
        )
        del model
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()

    usable = [r for r in records if "training_tokens_per_second" in r]
    recommendation = None
    if usable:
        slowest = min(usable, key=lambda r: r["training_tokens_per_second"])
        # Budget so wählen, dass auch die langsamste Stufe im vorgegebenen
        # Zeitfenster bleibt. Auf Zehntausender gerundet.
        raw = slowest["training_tokens_per_second"] * args.budget_hours * 3600
        recommendation = max(
            slowest["tokens_per_step"] * 50,
            int(round(raw / 100_000) * 100_000) or slowest["tokens_per_step"] * 50,
        )
        console.log(
            f"\n  Langsamste Stufe: {slowest['size_class']} mit "
            f"{slowest['training_tokens_per_second']:,.0f} tok/s"
        )
        console.log(
            f"  Empfohlenes Fixed-Token-Budget für {args.budget_hours} h je Stufe: "
            f"{recommendation:,} Token"
        )
    return {"models": records, "recommended_token_budget": recommendation,
            "budget_hours": args.budget_hours}


# ----------------------------------------------------------------------
# Abschnitt: Fixed-Token Scaling
# ----------------------------------------------------------------------

def section_fixed(args, capabilities, corpora, console) -> dict[str, Any]:
    """Alle Größen, dasselbe Tokenbudget – das kontrollierte Experiment."""
    budget = args.token_budget
    console.log(f"\n########## Fixed-Token Scaling: {budget:,} Token je Größenklasse "
                f"auf {args.corpus} ##########")
    records = []
    for name in args.classes:
        size = size_class(name)
        steps = steps_for_budget(
            budget, batch_size=size.batch_size, sequence_length=size.sequence_length,
            gradient_accumulation=gradient_accumulation(size),
        )
        records.append(measure_size_class(
            size, args, capabilities, corpora, console, steps=steps, study="fixed"
        ))
    return {
        "token_budget": budget,
        "effective_batch_size": TARGET_EFFECTIVE_BATCH,
        "corpus": args.corpus,
        "classes": records,
    }


def section_capacity(args, capabilities, corpora, console) -> dict[str, Any]:
    """Größere Modelle bekommen mehr Token – die realistischere Frage."""
    console.log(f"\n########## Capacity-Aware Scaling auf {args.corpus} ##########")
    records = []
    scale = {name: index for index, name in enumerate(
        [size.name for size in SIZE_CLASSES])}
    for name in args.capacity_classes:
        size = size_class(name)
        # Budget wächst mit der Wurzel der Parameterzahl. Das ist eine
        # bewusst einfache Regel und wird als solche ausgewiesen, nicht als
        # theoretisch hergeleitetes Optimum.
        factor = args.capacity_factor ** scale[name]
        budget = int(args.token_budget * factor)
        steps = steps_for_budget(
            budget, batch_size=size.batch_size, sequence_length=size.sequence_length,
            gradient_accumulation=gradient_accumulation(size),
        )
        record = measure_size_class(
            size, args, capabilities, corpora, console, steps=steps, study="capacity"
        )
        record["budget_factor"] = factor
        records.append(record)
    return {
        "base_token_budget": args.token_budget,
        "capacity_factor": args.capacity_factor,
        "corpus": args.corpus,
        "classes": records,
    }


# ----------------------------------------------------------------------
# Abschnitt: Long-Context Streaming
# ----------------------------------------------------------------------

def section_sequence(args, capabilities, corpora, console) -> dict[str, Any]:
    """Bleiben die Kosten je neuem Token bei wachsendem Kontext konstant?

    Zwei getrennte Messungen: (a) ein Durchlauf über eine lange Sequenz,
    (b) die Latenz je Token in Fenstern an bestimmten Kontextpositionen. Nur
    (b) beantwortet die eigentliche Frage, denn nur dort wächst der bereits
    verarbeitete Kontext wirklich mit.
    """
    console.log("\n=== Sequenzlängen und Streaming über wachsenden Kontext ===")
    results = []
    for name in args.sequence_classes:
        size = size_class(name)
        seed_everything(args.seed)
        model = GlassMindLM(size.config()).to(capabilities.torch_device)
        per_length = []
        for length in args.lengths:
            reset_peak_memory(capabilities)
            try:
                rate = throughput(model, capabilities, length=length, repeats=1, rounds=2)
            except torch.cuda.OutOfMemoryError:
                console.log(f"  {size.name} L={length:>7,}: Speicher reicht nicht")
                torch.cuda.empty_cache()
                per_length.append({"length": length, "out_of_memory": True})
                continue
            peak = peak_memory_bytes(capabilities) or 0
            per_length.append({
                "length": length,
                "tokens_per_second": rate,
                "seconds_per_sequence": length / rate,
                "peak_memory_bytes": peak,
            })
            console.log(f"  {size.name} L={length:>7,}: {rate:>10,.0f} tok/s  "
                        f"Peak {peak/1e6:8.1f} MB")

        # Latenzfenster an definierten Kontextpositionen. Der Zustand wird
        # fortgeführt; zwischen den Messfenstern wird der Kontext ohne Messung
        # weitergeschoben.
        windows = []
        state = None
        position = 0
        token = torch.zeros(1, dtype=torch.long, device=capabilities.torch_device)
        for start in args.latency_positions:
            if start > position:
                with torch.inference_mode(), autocast_context(capabilities):
                    for _ in range(start - position):
                        _, state = model.step(token, state)
                    synchronize(capabilities)
                position = start
            latency, state = streaming_latency(
                model, capabilities, steps=args.latency_steps, warmup=0, state=state, rounds=1
            )
            position += args.latency_steps
            windows.append({
                "window_start": start,
                "window_end": position,
                "ms_per_token": latency,
            })
            console.log(f"  {size.name} Kontext {start:>7,}–{position:>7,}: "
                        f"{latency:.4f} ms/Token")
        first, last = windows[0]["ms_per_token"], windows[-1]["ms_per_token"]
        change = 100 * (last / first - 1)
        console.log(f"  {size.name}: Änderung über den gesamten Kontext {change:+.1f} %")
        results.append({
            "size_class": size.name,
            "parameter_count": model.parameter_count,
            "lengths": per_length,
            "latency_windows": windows,
            "latency_change_percent": change,
            # Als annähernd konstant gilt eine Änderung unter 10 Prozent über
            # den gesamten geprüften Kontext.
            "constant": abs(change) < 10.0,
        })
        del model
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
    return {"models": results, "positions": list(args.latency_positions),
            "window_steps": args.latency_steps}


# ----------------------------------------------------------------------
# Abschnitt: Präzision
# ----------------------------------------------------------------------

def _training_step_seconds(model, capabilities, *, length: int, batch: int,
                           amp_dtype: torch.dtype | None, repeats: int = 3) -> float | None:
    """Zeit für einen vollständigen Trainingsschritt – ohne die Gewichte zu ruinieren.

    Die Lehre aus Milestone 2.6: Eine Messung mit echten Optimizer-Schritten
    verändert das Modell. Deshalb wird der Zustand gesichert und zurückgespielt.
    """
    import torch.nn.functional as F

    snapshot = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-9)
    tokens = torch.randint(0, model.config.vocab_size, (batch, length + 1),
                           device=capabilities.torch_device)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    model.train()
    device_type = capabilities.torch_device.type
    try:
        def one_step() -> None:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device_type,
                                dtype=amp_dtype or torch.bfloat16,
                                enabled=amp_dtype is not None):
                logits, _ = model(inputs)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                       targets.reshape(-1))
            loss.backward()
            optimizer.step()

        for _ in range(2):
            one_step()
        synchronize(capabilities)
        started = time.perf_counter()
        for _ in range(repeats):
            one_step()
        synchronize(capabilities)
        return (time.perf_counter() - started) / repeats
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None
    finally:
        model.load_state_dict({k: v.to(capabilities.torch_device) for k, v in snapshot.items()})
        model.eval()


def section_precision(args, capabilities, corpora, console) -> dict[str, Any]:
    """Wird halbe Präzision bei größeren Matrizen doch zum Gewinn?

    Milestone 2.6 hat bei d_model=64 gemessen, dass BF16 *nichts* bringt, weil
    die Cast-Kernel teurer sind als die eingesparte Rechenzeit. Diese
    Erkenntnis wird hier nicht übertragen, sondern je Stufe neu geprüft.
    """
    console.log("\n=== Präzision je Größenklasse, neu gemessen ===")
    variants: list[tuple[str, torch.dtype | None, torch.dtype | None]] = [
        ("fp32", None, None),
        ("fp32 + AMP bf16", None, torch.bfloat16),
        ("fp32 + AMP fp16", None, torch.float16),
        ("Gewichte bf16", torch.bfloat16, None),
        ("Gewichte fp16", torch.float16, None),
    ]
    results = []
    for name in args.precision_classes:
        size = size_class(name)
        per_variant = []
        for label, weight_dtype, amp_dtype in variants:
            seed_everything(args.seed)
            try:
                model = GlassMindLM(size.config()).to(capabilities.torch_device)
                if weight_dtype is not None:
                    model = model.to(weight_dtype)
                model.eval()
                device_type = capabilities.torch_device.type
                tokens = torch.randint(0, model.config.vocab_size,
                                       (1, size.sequence_length),
                                       device=capabilities.torch_device)
                with torch.inference_mode(), torch.autocast(
                    device_type=device_type, dtype=amp_dtype or torch.bfloat16,
                    enabled=amp_dtype is not None
                ):
                    for _ in range(2):
                        model(tokens)
                    synchronize(capabilities)
                    samples = []
                    for _ in range(3):
                        started = time.perf_counter()
                        for _ in range(3):
                            model(tokens)
                        synchronize(capabilities)
                        samples.append(tokens.numel() * 3 / (time.perf_counter() - started))
                    finite = bool(torch.isfinite(model(tokens)[0]).all())
                inference = statistics.median(samples)
                step_seconds = _training_step_seconds(
                    model, capabilities, length=size.sequence_length,
                    batch=size.batch_size, amp_dtype=amp_dtype,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                per_variant.append({"variant": label, "out_of_memory": True})
                continue
            per_variant.append({
                "variant": label,
                "inference_tokens_per_second": inference,
                "training_step_seconds": step_seconds,
                "training_tokens_per_second": (
                    None if step_seconds is None
                    else size.batch_size * size.sequence_length / step_seconds
                ),
                "logits_finite": finite,
                "parameter_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            })
            console.log(
                f"  {size.name:8s} {label:18s} Inferenz {inference:>9,.0f} tok/s  "
                f"Training {'n/a' if step_seconds is None else f'{step_seconds*1000:8.1f} ms'}"
                f"  endlich={finite}"
            )
            del model
            if capabilities.backend in {"cuda", "rocm"}:
                torch.cuda.empty_cache()
        baseline = next((v for v in per_variant if v.get("variant") == "fp32"), None)
        if baseline and "inference_tokens_per_second" in baseline:
            for entry in per_variant:
                if "inference_tokens_per_second" in entry:
                    entry["inference_speedup"] = (
                        entry["inference_tokens_per_second"]
                        / baseline["inference_tokens_per_second"]
                    )
                if entry.get("training_step_seconds") and baseline.get("training_step_seconds"):
                    entry["training_speedup"] = (
                        baseline["training_step_seconds"] / entry["training_step_seconds"]
                    )
        results.append({"size_class": size.name, "variants": per_variant})
    return {"models": results}


# ----------------------------------------------------------------------
# Abschnitt: Quantisierte Ablage
# ----------------------------------------------------------------------

def section_quantization(args, capabilities, corpora, console) -> dict[str, Any]:
    """Was bringt Weight-Quantisierung bei einem größeren Modell wirklich?

    Ausdrücklich getrennt ausgewiesen: Speicherersparnis ist nicht
    Geschwindigkeit. Wenn nur die Datei kleiner wird, steht genau das da.
    """
    console.log("\n=== Quantisierte Ablage ===")
    name = args.quantization_class
    source = args.checkpoint_dir / f"m4-fixed-{args.corpus}-{name}.pt"
    if not source.exists():
        console.log(f"  Kein trainierter Checkpoint unter {source} – Abschnitt übersprungen")
        return {"skipped": f"{source} fehlt"}

    reference_model, tokenizer, _ = load_checkpoint(source, device=capabilities.torch_device)
    reference_model.eval()
    size = size_class(name)
    tokens = next(random_batches(
        corpora["validation"], batch_size=1, sequence_length=size.sequence_length, seed=4711
    ))[0].to(capabilities.torch_device)
    with torch.inference_mode(), autocast_context(capabilities):
        reference_logits = reference_model(tokens)[0].float()
    reference_prediction = reference_logits.argmax(-1)
    reference_eval = evaluate_language(
        reference_model, corpora["validation"], capabilities,
        sequence_length=size.sequence_length, batch_size=1, batches=args.eval_batches,
    )

    # Die Gruppengröße muss *jede* quantisierte Eingangsbreite teilen. Sie wird
    # deshalb aus dem Modell abgeleitet statt aus einer Kandidatenliste geraten:
    # der größte Teiler des ggT aller Breiten, der die gewünschte Größe nicht
    # überschreitet.
    #
    # Das ist keine Formsache. GlassMinds fusionierter ``integrator`` liest
    # ``2*d_model + semantic_width + binding_rank`` Werte – bei d_model=640 sind
    # das 1930 = 2*5*193. Keine Zweierpotenz teilt das. Die Folge steht im
    # Ergebnis: Wo nur winzige Gruppen möglich sind, kosten die Skalen mehr, als
    # die schmaleren Gewichte sparen.
    def usable_group_size() -> tuple[int | None, int]:
        from functools import reduce
        from math import gcd

        from glassmind.precision.apply import iter_linear_like

        probe, _, _ = load_checkpoint(source, device="cpu")
        widths = {
            module.in_features for _, module in iter_linear_like(probe)
            if getattr(module, "in_features", None)
        }
        del probe
        if not widths:
            return None, 0
        common = reduce(gcd, widths)
        candidates = [d for d in range(1, min(common, args.group_size) + 1)
                      if common % d == 0]
        return (max(candidates) if candidates else None), common

    group, common = usable_group_size()
    console.log(f"  größter gemeinsamer Teiler aller Eingangsbreiten: {common} "
                f"-> Gruppengröße {group}")
    variants: list[tuple[str, PrecisionPolicy | None]] = [("unquantisiert (fp32)", None)]
    if group is None:
        console.log("  keine gültige Gruppengröße – Quantisierung nicht anwendbar")
    else:
        for scheme in ("int8", "int4"):
            variants.append((f"{scheme.upper()} weight-only (Gruppe {group})",
                             experimental_profile("float32", scheme, group)))
    records = []
    for label, policy in variants:
        slug = label.split()[0].lower().replace("(", "").replace(")", "")
        target = args.checkpoint_dir / f"m4-quant-{name}-{slug}.pt"
        if policy is None:
            model = reference_model
            save_checkpoint(target, model, tokenizer=tokenizer)
        else:
            model, _, _ = load_checkpoint(source, device="cpu")
            from glassmind.precision.apply import apply_precision

            apply_precision(model, policy, device=capabilities.torch_device)
            model = model.to(capabilities.torch_device)
            model.precision = policy
            save_checkpoint(target, model, tokenizer=tokenizer)
        model.eval()

        started = time.perf_counter()
        loaded, _, _ = load_checkpoint(target, device=capabilities.torch_device)
        load_seconds = time.perf_counter() - started
        loaded.eval()

        reset_peak_memory(capabilities)
        with torch.inference_mode(), autocast_context(capabilities):
            logits = loaded(tokens)[0].float()
        rate = throughput(loaded, capabilities, length=size.sequence_length)
        metrics = evaluate_language(
            loaded, corpora["validation"], capabilities,
            sequence_length=size.sequence_length, batch_size=1, batches=args.eval_batches,
        )
        entry = {
            "variant": label,
            "checkpoint_bytes": target.stat().st_size,
            "checkpoint_ratio": target.stat().st_size / max(source.stat().st_size, 1),
            "load_seconds": load_seconds,
            "inference_tokens_per_second": rate,
            "peak_memory_bytes": peak_memory_bytes(capabilities) or 0,
            "process_rss_bytes": process_rss_bytes(),
            "validation_loss": metrics["loss"],
            "validation_bits_per_byte": metrics["bits_per_byte"],
            "bits_per_byte_delta": metrics["bits_per_byte"] - reference_eval["bits_per_byte"],
            "logit_rms_difference": float((logits - reference_logits).square().mean().sqrt()),
            "logit_max_difference": float((logits - reference_logits).abs().max()),
            "prediction_change_rate": float(
                (logits.argmax(-1) != reference_prediction).float().mean()
            ),
        }
        records.append(entry)
        console.log(
            f"  {label:22s} {entry['checkpoint_bytes']/1e6:7.1f} MB "
            f"({entry['checkpoint_ratio']:.2f}x)  Laden {load_seconds:5.2f} s  "
            f"{rate:>9,.0f} tok/s  Δbpb={entry['bits_per_byte_delta']:+.4f}  "
            f"geänderte Vorhersagen={entry['prediction_change_rate']:.2%}"
        )
        if policy is not None:
            del model
        del loaded
        if capabilities.backend in {"cuda", "rocm"}:
            torch.cuda.empty_cache()
    del reference_model
    return {"size_class": name, "source_checkpoint": str(source),
            "reference": reference_eval, "variants": records}


# ----------------------------------------------------------------------
# Abschnitt: Dispatch gegen Compute
# ----------------------------------------------------------------------

def section_profile(args, capabilities, corpora, console) -> dict[str, Any]:
    """Ab welcher Größe rechnet die GPU mehr, als der Host sie beschäftigt?

    Der Quotient GPU-Zeit zu Wanduhrzeit ist der Kern: nahe 0 heißt, der
    Python-/Dispatch-Pfad bestimmt das Tempo; nahe 1 heißt, die Matrizen sind
    groß genug, dass die Rechenleistung limitiert.
    """
    if capabilities.backend not in {"cuda", "rocm"}:
        console.log("\n[Profil] Nur auf CUDA/ROCm sinnvoll – übersprungen")
        return {"skipped": "kein CUDA/ROCm-Backend"}
    from torch.profiler import ProfilerActivity, profile

    console.log("\n=== Dispatch gegen Compute ===")
    results = []
    for name in args.profile_classes:
        size = size_class(name)
        seed_everything(args.seed)
        try:
            model = GlassMindLM(size.config()).to(capabilities.torch_device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            results.append({"size_class": size.name, "out_of_memory": True})
            continue
        model.eval()
        tokens = torch.randint(0, model.config.vocab_size, (1, args.profile_length),
                               device=capabilities.torch_device)
        with torch.inference_mode():
            for _ in range(3):
                model(tokens)
            synchronize(capabilities)
            started = time.perf_counter()
            model(tokens)
            synchronize(capabilities)
            wall = time.perf_counter() - started
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                model(tokens)
                synchronize(capabilities)
        events = prof.key_averages()
        cpu_us = sum(e.self_cpu_time_total for e in events)
        gpu_us = sum(getattr(e, "self_device_time_total", 0) or 0 for e in events)
        calls = sum(e.count for e in events)
        entry = {
            "size_class": size.name,
            "parameter_count": model.parameter_count,
            "d_model": size.d_model,
            "n_layers": size.n_layers,
            "profile_length": args.profile_length,
            "wall_seconds": wall,
            "cpu_time_ms": cpu_us / 1000,
            "gpu_time_ms": gpu_us / 1000,
            "aten_calls": calls,
            "calls_per_token": calls / args.profile_length,
            "gpu_utilisation_ratio": (gpu_us / 1e6) / max(wall, 1e-9),
            "gpu_over_cpu": gpu_us / max(cpu_us, 1),
            "top_operations": [
                {"name": e.key, "calls": e.count,
                 "cpu_ms": e.self_cpu_time_total / 1000,
                 "gpu_ms": (getattr(e, "self_device_time_total", 0) or 0) / 1000}
                for e in sorted(events,
                                key=lambda x: -(getattr(x, "self_device_time_total", 0) or 0))[:10]
            ],
        }
        results.append(entry)
        console.log(
            f"  {size.name:8s} d={size.d_model:4d}  ATen={calls:6,d}  "
            f"CPU={cpu_us/1000:8.1f} ms  GPU={gpu_us/1000:8.1f} ms  "
            f"GPU/Wanduhr={entry['gpu_utilisation_ratio']:.2f}  "
            f"GPU/CPU={entry['gpu_over_cpu']:.2f}"
        )
        del model
        torch.cuda.empty_cache()

    crossing = None
    usable = [r for r in results if "gpu_utilisation_ratio" in r]
    for previous, current in zip(usable, usable[1:]):
        if previous["gpu_utilisation_ratio"] < 0.5 <= current["gpu_utilisation_ratio"]:
            crossing = {"from": previous["size_class"], "to": current["size_class"],
                        "from_ratio": previous["gpu_utilisation_ratio"],
                        "to_ratio": current["gpu_utilisation_ratio"]}
            break
    if crossing:
        console.log(f"\n  Übergang dispatch-bound -> compute-bound zwischen "
                    f"{crossing['from']} und {crossing['to']}")
    else:
        console.log("\n  Kein Übergang im geprüften Bereich: alle Stufen bleiben "
                    "auf derselben Seite der Grenze")
    return {"models": results, "transition": crossing}


# ----------------------------------------------------------------------
# Abschnitt: VisPy-Trace
# ----------------------------------------------------------------------

def section_trace(args, capabilities, corpora, console) -> dict[str, Any]:
    """Ein reproduzierbarer Trace eines trainierten Sprachmodells.

    Das Netz zeigt weiterhin ausschließlich gemessene Aktivität. Es wird
    keinem Cluster eine Bedeutung zugewiesen – die Frage lautet nur, ob bei
    natürlicher Sprache stabilere oder wiederkehrende Cluster entstehen.
    """
    name = args.trace_class
    source = args.checkpoint_dir / f"m4-fixed-{args.corpus}-{name}.pt"
    if not source.exists():
        console.log(f"\n[Trace] Kein Checkpoint unter {source} – übersprungen")
        return {"skipped": f"{source} fehlt"}
    console.log(f"\n=== VisPy-Trace aus {source.name} ===")
    model, tokenizer, _ = load_checkpoint(source, device=capabilities.torch_device)
    model.eval()
    prompt = prompts_for(args.corpus)[0]
    encoded = torch.tensor([tokenizer.encode(prompt, add_bos=True)], dtype=torch.long,
                           device=capabilities.torch_device)
    destination = args.trace_dir
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"m4-{args.corpus}-{name}.jsonl"

    from glassmind.analysis.clusters import ClusterAnalyzer

    analyzer = ClusterAnalyzer()
    recorder = JSONLRecorder(path)
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(recorder)
    bus.subscribe(analyzer)
    with torch.inference_mode(), autocast_context(capabilities):
        model.generate(encoded, args.trace_tokens, temperature=0.0, observer=bus)
    bus.close()

    summaries = analyzer.summaries()
    per_kind: dict[str, list[float]] = {}
    for node_id, summary in summaries.items():
        parts = node_id.split(".")
        kind = parts[2] if len(parts) > 2 else "other"
        per_kind.setdefault(kind, []).append(float(summary.get("mean_persistence", 0.0)))
    stability = {
        kind: {"clusters": len(values),
               "mean_persistence": statistics.fmean(values) if values else 0.0}
        for kind, values in per_kind.items()
    }
    console.log(f"  Trace: {path}  Ereignisse={bus.events_emitted:,}  "
                f"Knoten={len(summaries)}")
    for kind, entry in sorted(stability.items()):
        console.log(f"    {kind:10s} {entry['clusters']:3d} Cluster  "
                    f"mittlere Persistenz {entry['mean_persistence']:.2f} Token")
    del model
    if capabilities.backend in {"cuda", "rocm"}:
        torch.cuda.empty_cache()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "events": bus.events_emitted,
        "prompt": prompt,
        "tokens": args.trace_tokens,
        "cluster_stability": stability,
        "node_summaries": summaries,
    }


# ----------------------------------------------------------------------

SECTIONS = {
    "benchmark": section_benchmark,
    "fixed": section_fixed,
    "capacity": section_capacity,
    "sequence": section_sequence,
    "precision": section_precision,
    "quantization": section_quantization,
    "profile": section_profile,
    "trace": section_trace,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 4: Skalierung auf echten Sprachdaten")
    parser.add_argument("--device", default="auto",
                        choices=["auto", "cpu", "cuda", "rocm", "mps", "xpu"])
    parser.add_argument("--seed", type=int, default=4)
    parser.add_argument("--corpus", default="tinystories", choices=["tinystories", "wikitext103"])
    parser.add_argument("--token-budget", type=int, default=4_000_000)
    parser.add_argument("--capacity-factor", type=float, default=2.0)
    parser.add_argument("--classes", nargs="+", default=[s.name for s in SIZE_CLASSES])
    parser.add_argument("--capacity-classes", nargs="+", default=[s.name for s in SIZE_CLASSES])
    parser.add_argument("--sequence-classes", nargs="+", default=["tiny", "small"])
    parser.add_argument("--precision-classes", nargs="+", default=["tiny", "small", "medium", "large"])
    parser.add_argument("--profile-classes", nargs="+", default=[s.name for s in SIZE_CLASSES])
    parser.add_argument("--quantization-class", default="medium")
    parser.add_argument("--trace-class", default="small")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--profile-length", type=int, default=128)
    parser.add_argument("--benchmark-steps", type=int, default=8)
    parser.add_argument("--budget-hours", type=float, default=0.5)
    parser.add_argument("--budget-candidates", type=int, nargs="+",
                        default=[2_000_000, 4_000_000, 8_000_000, 16_000_000])
    parser.add_argument("--lengths", type=int, nargs="+",
                        default=[1024, 4096, 8192, 16384, 32768])
    parser.add_argument("--latency-positions", type=int, nargs="+",
                        default=[0, 4096, 16384, 65536])
    parser.add_argument("--latency-steps", type=int, default=1024)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--ablation-batches", type=int, default=8)
    parser.add_argument("--generate-tokens", type=int, default=300)
    parser.add_argument("--trace-tokens", type=int, default=64)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs/milestone4"))
    parser.add_argument("--trace-dir", type=Path, default=Path("runs/milestone4/traces"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--reuse-checkpoints", action="store_true",
                        help="Vorhandene Checkpoints laden statt neu zu trainieren")
    parser.add_argument("--sections", nargs="+",
                        default=["benchmark", "fixed", "sequence", "precision", "profile"],
                        choices=sorted(SECTIONS))
    args = parser.parse_args()

    output = args.output or Path(f"benchmarks/milestone4-{args.corpus}.json")
    seed_everything(args.seed)
    capabilities = detect_device(args.device, "auto")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    console = Console(args.checkpoint_dir / f"scaling-{args.corpus}.log")

    console.log(f"[Skalierung] Backend={capabilities.backend}  Gerät={capabilities.name}  "
                f"Precision={capabilities.precision}")
    cross = "wikitext103" if args.corpus == "tinystories" else "tinystories"
    corpora: dict[str, Any] = {
        "train": prepare_corpus(args.corpus, "train"),
        "validation": prepare_corpus(args.corpus, "validation"),
        "cross_validation": prepare_corpus(cross, "validation"),
    }
    console.log("[Korpus] baue Wortliste für die Qualitätsmessung ...")
    corpora["vocabulary"] = corpus_vocabulary(corpora["train"])
    for key in ("train", "validation", "cross_validation"):
        summary = corpora[key].summary()
        console.log(f"[Korpus] {key:16s} {summary['name']}/{summary['split']}  "
                    f"{summary['token']:,} Token  {summary['dokumente']:,} Dokumente  "
                    f"Lizenz {summary['lizenz']}  Revision {summary['revision']}")

    payload: dict[str, Any] = {
        "created": datetime.now(timezone.utc).isoformat(),
        "milestone": "4",
        "corpus": args.corpus,
        "environment": environment_metadata(capabilities, seed=args.seed),
        "corpora": {key: corpora[key].metadata
                    for key in ("train", "validation", "cross_validation")},
        "vocabulary_words": len(corpora["vocabulary"]),
        "size_classes": [
            {"name": s.name, "target": s.target, "d_model": s.d_model, "n_layers": s.n_layers,
             "learning_rate": s.learning_rate, "sequence_length": s.sequence_length,
             "micro_batch_size": s.batch_size,
             "gradient_accumulation": gradient_accumulation(s),
             "effective_batch_size": s.batch_size * gradient_accumulation(s)}
            for s in SIZE_CLASSES
        ],
        "settings": {k: (str(v) if isinstance(v, Path) else v)
                     for k, v in vars(args).items() if k != "output"},
        "sections": {},
    }
    # Vorhandene Abschnitte **vor** dem ersten Schreiben übernehmen.
    #
    # Vorher stand der Merge am Ende – nach dem Zwischenspeichern. Das
    # Zwischenspeichern hatte die Datei da längst mit den neuen Abschnitten
    # überschrieben, und der Merge las genau diese überschriebene Datei zurück.
    # Er war damit wirkungslos und die früheren Abschnitte waren verloren.
    # Einmal vorne einlesen behebt beides zugleich.
    if args.merge and output.exists():
        try:
            previous = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            console.log(f"[Warnung] {output} nicht lesbar ({exc}); nichts übernommen")
        else:
            for key, value in previous.get("sections", {}).items():
                payload["sections"][key] = value
            if previous.get("sections"):
                console.log("[Merge] übernommen: "
                            + ", ".join(sorted(previous["sections"])))

    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")

    for name in args.sections:
        payload["sections"][name] = SECTIONS[name](args, capabilities, corpora, console)
        # Zwischenstand nach jedem Abschnitt sichern: Ein Abbruch im letzten
        # Abschnitt darf nicht die Ergebnisse der ersten kosten.
        save()
    save()
    console.log(f"\n[Ergebnis] {output}  Abschnitte: "
                + ", ".join(sorted(payload["sections"])))


if __name__ == "__main__":
    main()
