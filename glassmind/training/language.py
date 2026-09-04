"""Milestone 4: Training und Messung auf echten Sprachdaten.

Der Kern ist bewusst dünn: ``train_language`` ist eine gewöhnliche
Trainingsschleife über zufällige Fenster eines Tokenstroms. Der Wert liegt in
dem, was daneben gemessen wird – Gradientennormen, Zustandsnormen,
Zustandsspezialisierung, Durchsatz, Speicher. Genau diese Größen entscheiden,
ob GlassMind sinnvoll skaliert.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
import statistics
import time
from typing import Any, Iterator

import torch
from torch import Tensor
import torch.nn.functional as F

from glassmind.data.corpus import TokenStream, bits_per_byte, random_batches
from glassmind.data.tokenizer import ByteTokenizer
from glassmind.model.lm import GlassMindLM
from glassmind.observe.bus import ObservationBus, ObservationMode
from glassmind.utils.device import (
    DeviceCapabilities,
    autocast_context,
    peak_memory_bytes,
    reset_peak_memory,
    synchronize,
)

#: Die drei Zeitskalen des Zustandskerns. Sie werden in Milestone 4 genauso
#: geprüft wie in Milestone 2 – echtes Sprachtraining darf sie nicht
#: stillschweigend verschwinden lassen.
STATE_NAMES = ("fast", "context", "semantic")


@dataclass(frozen=True)
class LanguageTrainingConfig:
    steps: int = 400
    batch_size: int = 8
    sequence_length: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    warmup_steps: int = 40
    log_every: int = 50
    seed: int = 4
    #: Wie viele Mikrobatches ein Optimizer-Schritt zusammenfasst. Damit lässt
    #: sich die *effektive* Batchgröße über Größenklassen hinweg konstant
    #: halten, obwohl große Modelle nur kleinere Mikrobatches in den VRAM
    #: bekommen. Ohne das wäre ein Fixed-Token-Vergleich unfair.
    gradient_accumulation: int = 1

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation

    @property
    def tokens_per_step(self) -> int:
        return self.effective_batch_size * self.sequence_length

    @property
    def token_budget(self) -> int:
        return self.tokens_per_step * self.steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {
            "effective_batch_size": self.effective_batch_size,
            "tokens_per_step": self.tokens_per_step,
            "token_budget": self.token_budget,
        }


def steps_for_budget(
    token_budget: int, *, batch_size: int, sequence_length: int, gradient_accumulation: int = 1
) -> int:
    """Schrittzahl, die ein gegebenes Tokenbudget genau ausschöpft."""
    per_step = batch_size * gradient_accumulation * sequence_length
    return max(1, round(token_budget / per_step))


def _scaler(capabilities: DeviceCapabilities) -> torch.amp.GradScaler | None:
    if capabilities.precision != "float16" or capabilities.backend not in {"cuda", "rocm", "xpu"}:
        return None
    return torch.amp.GradScaler("cuda" if capabilities.backend in {"cuda", "rocm"} else "xpu")


def _learning_rate(step: int, config: LanguageTrainingConfig) -> float:
    """Linearer Warmup, danach Cosine-Abfall auf ein Zehntel.

    Ohne Warmup divergiert der gebundene Zustandspfad bei größeren Modellen in
    den ersten Schritten – das war messbar, nicht angenommen.
    """
    if step <= config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


def train_language(
    model: GlassMindLM,
    stream: TokenStream,
    config: LanguageTrainingConfig,
    capabilities: DeviceCapabilities,
    *,
    logger: Any = None,
    validation: TokenStream | None = None,
    validate_every: int = 0,
) -> dict[str, Any]:
    """Trainiert auf zufälligen Fenstern und berichtet, was dabei passiert."""
    device = capabilities.torch_device
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )
    scaler = _scaler(capabilities)
    batches = random_batches(
        stream, batch_size=config.batch_size, sequence_length=config.sequence_length,
        seed=config.seed,
    )
    model.train()
    reset_peak_memory(capabilities)
    grad_norms: list[float] = []
    losses: list[float] = []
    curve: list[dict[str, float]] = []
    best_loss = math.inf
    seen_tokens = 0
    diverged = False
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        inputs, targets = next(batches)
        inputs, targets = inputs.to(device), targets.to(device)
        learning_rate = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated = 0.0
        for micro in range(config.gradient_accumulation):
            if micro > 0:
                inputs, targets = next(batches)
                inputs, targets = inputs.to(device), targets.to(device)
            with autocast_context(capabilities):
                logits, _ = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1)
                )
            if not torch.isfinite(loss):
                # Kein Abbruch mit Exception: Divergenz ist bei einer
                # Skalierungsstudie ein *Messergebnis* und gehört in den Bericht.
                diverged = True
                break
            # Durch die Zahl der Mikrobatches teilen, damit der Gradient
            # derselbe ist wie bei einem einzigen großen Batch.
            scaled = loss / config.gradient_accumulation
            (scaled if scaler is None else scaler.scale(scaled)).backward()
            accumulated += float(loss.detach())
            seen_tokens += targets.numel()
        if diverged:
            if logger is not None:
                logger.log(f"[sprache] Schritt {step}: nicht-endlicher Loss, Abbruch")
            break
        if scaler is None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
        else:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        value = accumulated / config.gradient_accumulation
        losses.append(value)
        grad_norms.append(float(grad_norm))
        best_loss = min(best_loss, value)
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            elapsed = max(time.perf_counter() - started, 1e-9)
            entry = {
                "step": step,
                "loss": value,
                "bits_per_byte": bits_per_byte(value),
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "tokens_per_second": seen_tokens / elapsed,
            }
            if validate_every and validation is not None and step % validate_every == 0:
                entry.update(
                    {f"val_{k}": v for k, v in
                     evaluate_language(model, validation, capabilities,
                                       sequence_length=config.sequence_length,
                                       batch_size=max(1, config.batch_size // 2),
                                       batches=8).items()}
                )
                model.train()
            curve.append(entry)
            if logger is not None:
                logger.log(
                    f"[sprache] Schritt {step}/{config.steps}  loss={value:.4f}  "
                    f"bpb={bits_per_byte(value):.3f}  lr={learning_rate:.2e}  "
                    f"grad={float(grad_norm):.3f}  tok/s={seen_tokens / elapsed:,.0f}"
                )
                logger.metric({"event": "language_step", **entry})
    synchronize(capabilities)
    elapsed = max(time.perf_counter() - started, 1e-9)
    tail = losses[-20:] or [math.inf]
    return {
        "steps_completed": len(losses),
        "diverged": diverged,
        "final_loss": tail[-1],
        "final_loss_mean_tail": statistics.fmean(tail),
        "best_loss": best_loss,
        "final_bits_per_byte": bits_per_byte(tail[-1]) if math.isfinite(tail[-1]) else None,
        "tokens_per_second": seen_tokens / elapsed,
        "seen_tokens": seen_tokens,
        "training_seconds": elapsed,
        "peak_memory_bytes": peak_memory_bytes(capabilities) or 0,
        "grad_norm_mean": statistics.fmean(grad_norms) if grad_norms else None,
        "grad_norm_median": statistics.median(grad_norms) if grad_norms else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "grad_norm_final": grad_norms[-1] if grad_norms else None,
        # Wie oft das Clipping überhaupt gegriffen hat – ein Maß dafür, ob die
        # gewählte Lernrate zur Größe passt.
        "grad_clip_hit_rate": (
            sum(1 for value in grad_norms if value > config.grad_clip) / len(grad_norms)
            if grad_norms else None
        ),
        "curve": curve,
        "config": config.to_dict(),
    }


@torch.inference_mode()
def evaluate_language(
    model: GlassMindLM,
    stream: TokenStream,
    capabilities: DeviceCapabilities,
    *,
    sequence_length: int = 512,
    batch_size: int = 4,
    batches: int = 32,
    seed: int = 777,
    **forward_options: Any,
) -> dict[str, float]:
    """Validierungs-Loss, Perplexity und bits/byte auf festen Fenstern."""
    model.eval()
    generator = random_batches(
        stream, batch_size=batch_size, sequence_length=sequence_length, seed=seed,
        steps=batches,
    )
    total_loss = 0.0
    total_tokens = 0
    for inputs, targets in generator:
        inputs = inputs.to(capabilities.torch_device)
        targets = targets.to(capabilities.torch_device)
        with autocast_context(capabilities):
            logits, _ = model(inputs, **forward_options)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss)
        total_tokens += targets.numel()
    mean = total_loss / max(total_tokens, 1)
    return {
        "loss": mean,
        # Perplexity ist hier *pro Byte*. Sie ist damit nicht mit
        # wortbasierten Literaturwerten vergleichbar; bits/byte ist es.
        "perplexity": math.exp(min(mean, 60.0)),
        "bits_per_byte": bits_per_byte(mean),
        "tokens": total_tokens,
    }


@torch.inference_mode()
def state_statistics(
    model: GlassMindLM,
    stream: TokenStream,
    capabilities: DeviceCapabilities,
    *,
    sequence_length: int = 512,
    seed: int = 909,
) -> dict[str, Any]:
    """RMS-Normen der drei Zeitskalen nach einem echten Sprachfenster.

    Gemessen wird der Zustand am Sequenzende, je Block getrennt. Es wird keine
    Rolle unterstellt – die Zahlen stehen für sich.
    """
    model.eval()
    inputs, _ = next(random_batches(stream, batch_size=2, sequence_length=sequence_length, seed=seed))
    with autocast_context(capabilities):
        _, state = model(inputs.to(capabilities.torch_device))
    per_block: list[dict[str, float]] = []
    for block in state.blocks:
        entry: dict[str, float] = {}
        for name in STATE_NAMES:
            value = getattr(block, name, None)
            if value is None:
                continue
            tensor = value.float()
            entry[f"{name}_rms"] = float(tensor.square().mean().sqrt())
            entry[f"{name}_absmax"] = float(tensor.abs().max())
            # Anteil praktisch inaktiver Einheiten: sagt, ob eine Zeitskala
            # überhaupt benutzt wird.
            entry[f"{name}_dead_fraction"] = float(
                (tensor.abs() < 1e-3).float().mean()
            )
        per_block.append(entry)
    aggregate = {
        key: statistics.fmean([block[key] for block in per_block if key in block])
        for key in {k for block in per_block for k in block}
    }
    return {"per_block": per_block, "mean": aggregate}


@torch.inference_mode()
def state_specialisation(
    model: GlassMindLM,
    stream: TokenStream,
    capabilities: DeviceCapabilities,
    *,
    sequence_length: int = 512,
    batch_size: int = 4,
    batches: int = 8,
    seed: int = 4242,
) -> dict[str, Any]:
    """Ablation der drei Zeitskalen auf echten Sprachdaten.

    Eine Zeitskala gilt als funktional unterscheidbar, wenn ihr Abschalten den
    Loss *anders* verändert als das Abschalten der beiden anderen. Es wird
    keine bestimmte Rangfolge erwartet.
    """
    reference = evaluate_language(
        model, stream, capabilities, sequence_length=sequence_length,
        batch_size=batch_size, batches=batches, seed=seed,
    )
    records = []
    for name in STATE_NAMES:
        ablated = evaluate_language(
            model, stream, capabilities, sequence_length=sequence_length,
            batch_size=batch_size, batches=batches, seed=seed,
            ablate_states=[name],
        )
        records.append({
            "state": name,
            "loss": ablated["loss"],
            "delta_loss": ablated["loss"] - reference["loss"],
            "delta_bits_per_byte": ablated["bits_per_byte"] - reference["bits_per_byte"],
            "relative_increase": (ablated["loss"] - reference["loss"]) / max(reference["loss"], 1e-9),
        })
    deltas = [record["delta_loss"] for record in records]
    spread = max(deltas) - min(deltas)
    return {
        "reference": reference,
        "ablations": records,
        # Streuung der drei Effekte: nahe null hieße, die Zeitskalen sind
        # austauschbar geworden.
        "delta_spread": spread,
        "distinguishable": spread > 0.02 and max(deltas) > 0.02,
    }


@torch.inference_mode()
def ablation_logit_effect(
    model: GlassMindLM,
    stream: TokenStream,
    capabilities: DeviceCapabilities,
    *,
    sequence_length: int = 512,
    batch_size: int = 2,
    seed: int = 8080,
) -> list[dict[str, Any]]:
    """Wirkung jeder Zeitskala direkt auf den Logits, nicht nur auf dem Loss.

    Der Loss mittelt über das Vokabular; die Logit-Differenz und die Zahl
    tatsächlich geänderter Vorhersagen zeigen genauer, ob eine Zeitskala die
    Ausgabe formt oder nur ihre Kalibrierung verschiebt.
    """
    model.eval()
    inputs, targets = next(
        random_batches(stream, batch_size=batch_size, sequence_length=sequence_length, seed=seed)
    )
    inputs = inputs.to(capabilities.torch_device)
    targets = targets.to(capabilities.torch_device)
    with autocast_context(capabilities):
        reference, _ = model(inputs)
    reference = reference.float()
    reference_prediction = reference.argmax(-1)
    records = []
    for name in STATE_NAMES:
        with autocast_context(capabilities):
            altered, _ = model(inputs, ablate_states=[name])
        altered = altered.float()
        prediction = altered.argmax(-1)
        records.append({
            "state": name,
            "logit_rms_difference": float((altered - reference).square().mean().sqrt()),
            "logit_max_difference": float((altered - reference).abs().max()),
            "prediction_change_rate": float((prediction != reference_prediction).float().mean()),
            "accuracy_reference": float((reference_prediction == targets).float().mean()),
            "accuracy_ablated": float((prediction == targets).float().mean()),
        })
    return records


def state_activity_profile(
    model: GlassMindLM,
    stream: TokenStream,
    capabilities: DeviceCapabilities,
    *,
    sequence_length: int = 256,
    seed: int = 6161,
) -> dict[str, Any]:
    """Aktivität, Persistenz und Reaktivierung je Zeitskala auf echtem Text.

    Die Werte stammen aus derselben Telemetrie, die auch das VisPy-Netz
    speist. Es wird ausdrücklich keine Bedeutung zugewiesen – nur gemessen,
    wie lange Aktivität anhält und wie oft sie zurückkehrt.
    """
    from glassmind.analysis.clusters import ClusterAnalyzer

    inputs, _ = next(random_batches(stream, batch_size=1, sequence_length=sequence_length, seed=seed))
    analyzer = ClusterAnalyzer()
    bus = ObservationBus(ObservationMode.TRACE)
    bus.subscribe(analyzer)
    model.eval()
    with torch.inference_mode(), autocast_context(capabilities):
        model(inputs.to(capabilities.torch_device), observer=bus)
    bus.close()
    summaries = analyzer.summaries()
    per_kind: dict[str, dict[str, float]] = {}
    for node_id, summary in summaries.items():
        parts = node_id.split(".")
        kind = parts[2] if len(parts) > 2 else "other"
        if kind not in STATE_NAMES:
            continue
        bucket = per_kind.setdefault(
            kind, {"clusters": 0, "mean_activity": 0.0, "mean_delta": 0.0,
                   "mean_persistence": 0.0, "reactivations": 0.0, "updates": 0.0}
        )
        bucket["clusters"] += 1
        bucket["mean_activity"] += float(summary.get("mean_activity", 0.0))
        bucket["mean_delta"] += float(summary.get("mean_delta", 0.0))
        bucket["mean_persistence"] += float(summary.get("mean_persistence", 0.0))
        bucket["reactivations"] += float(summary.get("reactivations", 0))
        bucket["updates"] += float(summary.get("updates", 0))
    for bucket in per_kind.values():
        count = max(bucket["clusters"], 1)
        for key in ("mean_activity", "mean_delta", "mean_persistence"):
            bucket[key] /= count
    return {"per_state": per_kind, "sequence_length": sequence_length,
            "observed_nodes": len(summaries)}


@torch.inference_mode()
def generation_samples(
    model: GlassMindLM,
    tokenizer: ByteTokenizer,
    capabilities: DeviceCapabilities,
    prompts: tuple[str, ...],
    *,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    seed: int = 31,
) -> list[dict[str, Any]]:
    """Qualitative Proben – greedy und gesampelt, damit beides sichtbar ist."""
    model.eval()
    torch.manual_seed(seed)
    samples = []
    for prompt in prompts:
        encoded = torch.tensor(
            [tokenizer.encode(prompt, add_bos=True)], dtype=torch.long,
            device=capabilities.torch_device,
        )
        with autocast_context(capabilities):
            greedy = model.generate(encoded, max_new_tokens, temperature=0.0)
            sampled = model.generate(encoded, max_new_tokens, temperature=temperature)
        samples.append({
            "prompt": prompt,
            "greedy": tokenizer.decode(greedy[0].cpu().tolist()),
            "sampled": tokenizer.decode(sampled[0].cpu().tolist()),
            "temperature": temperature,
        })
    return samples


# ----------------------------------------------------------------------
# Objektive Sprachqualität jenseits der Perplexity
# ----------------------------------------------------------------------
# Perplexity misst, wie überrascht das Modell von echtem Text ist. Sie sagt
# nichts darüber, ob das Modell *selbst* brauchbaren Text erzeugt. Die
# folgenden Größen sind bewusst einfach, objektiv und ohne zweites Modell
# berechenbar.

#: Ein Wort ist hier eine Folge aus Buchstaben und Apostrophen. Bewusst
#: schlicht: Es geht darum, ob das Modell buchstabieren gelernt hat, nicht um
#: linguistisch saubere Tokenisierung.
_WORD_PATTERN = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")


def _words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _WORD_PATTERN.finditer(text)]


def corpus_vocabulary(stream: TokenStream, *, sample_tokens: int = 4_000_000) -> set[str]:
    """Wortmenge aus einem Ausschnitt des Trainingskorpus.

    Dient als Referenz für ``valid_word_fraction``. Ein Wort gilt als gültig,
    wenn es im Trainingskorpus vorkommt – das ist eine Aussage über
    Buchstabierfähigkeit, nicht über Sprachrichtigkeit.
    """
    import numpy as np

    window = np.asarray(stream.tokens[: min(sample_tokens, len(stream))], dtype=np.int64)
    payload = bytes(
        int(value - ByteTokenizer.BYTE_OFFSET) & 0xFF
        for value in window
        if ByteTokenizer.BYTE_OFFSET <= value < 260
    )
    return set(_words(payload.decode("utf-8", errors="ignore")))


def distinct_ratio(tokens: list[str], n: int) -> float:
    """Anteil einzigartiger n-Gramme – ein direktes Maß gegen Endlosschleifen."""
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams)


def generation_quality(
    samples: list[dict[str, Any]], vocabulary: set[str], *, key: str = "sampled"
) -> dict[str, float]:
    """Objektive Kennzahlen über erzeugten Text – ohne Geschmacksurteil."""
    continuations = [
        sample[key][len(sample["prompt"]) :] for sample in samples if key in sample
    ]
    words = [word for text in continuations for word in _words(text)]
    if not words:
        return {"words": 0}
    valid = sum(1 for word in words if word in vocabulary)
    characters = "".join(continuations)
    printable = sum(1 for character in characters if character.isprintable() or character in "\n\t")
    return {
        "words": len(words),
        # Anteil erzeugter Wörter, die im Trainingskorpus vorkommen. Bei einem
        # Byte-Tokenizer ist das die harte Prüfung, ob Buchstabieren gelernt wurde.
        "valid_word_fraction": valid / len(words),
        "mean_word_length": sum(len(word) for word in words) / len(words),
        "distinct_1": distinct_ratio(words, 1),
        "distinct_2": distinct_ratio(words, 2),
        "distinct_3": distinct_ratio(words, 3),
        # Ersatzzeichen entstehen, wenn erzeugte Bytes keine gültige UTF-8-Folge
        # bilden – ein direktes Maß für kaputte Ausgabe.
        "replacement_char_rate": characters.count("�") / max(len(characters), 1),
        "printable_fraction": printable / max(len(characters), 1),
    }


@torch.inference_mode()
def top1_accuracy(
    model: GlassMindLM,
    stream: TokenStream,
    capabilities: DeviceCapabilities,
    *,
    sequence_length: int = 512,
    batch_size: int = 4,
    batches: int = 8,
    seed: int = 555,
) -> dict[str, float]:
    """Wie oft trifft das Modell das nächste Byte exakt?

    Eine grobe, aber vollständig objektive Größe: Sie hängt nicht von der
    Kalibrierung der Wahrscheinlichkeiten ab, anders als die Perplexity.
    """
    model.eval()
    correct = total = 0
    for inputs, targets in random_batches(
        stream, batch_size=batch_size, sequence_length=sequence_length, seed=seed, steps=batches
    ):
        inputs = inputs.to(capabilities.torch_device)
        targets = targets.to(capabilities.torch_device)
        with autocast_context(capabilities):
            logits, _ = model(inputs)
        correct += int((logits.argmax(-1) == targets).sum())
        total += targets.numel()
    return {"top1_accuracy": correct / max(total, 1), "tokens": total}
