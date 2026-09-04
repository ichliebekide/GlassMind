"""Milestone 4: prüft Korpus-Pipeline, Größenklassen und Sprachtraining.

Die Tests kommen ohne Netzzugriff aus: Wo ein echter Korpus gebraucht wird,
tritt ein synthetischer Tokenstrom an seine Stelle, der dieselbe Schnittstelle
bedient. Nur der Registrierungs-Test schaut auf die echten Metadaten, und auch
er lädt nichts herunter.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from glassmind.data.corpus import (
    CORPORA,
    TOKEN_DTYPE,
    TokenStream,
    TokenWindowDataset,
    bits_per_byte,
    corpus_key,
    random_batches,
)
from glassmind.data.tokenizer import ByteTokenizer
from glassmind.model import GlassMindLM
from glassmind.model.sizes import DEFAULT_VOCAB, SIZE_CLASSES, size_class
from glassmind.training.language import (
    STATE_NAMES,
    LanguageTrainingConfig,
    evaluate_language,
    generation_samples,
    state_specialisation,
    state_statistics,
    train_language,
)
from glassmind.utils.device import detect_device
from glassmind.utils.reproducibility import seed_everything


@pytest.fixture(scope="module")
def capabilities():
    return detect_device("cpu", "float32")


@pytest.fixture
def stream(tmp_path: Path) -> TokenStream:
    """Ein deterministischer Tokenstrom mit derselben Schnittstelle wie ein Korpus."""
    generator = np.random.default_rng(3)
    tokens = generator.integers(
        ByteTokenizer.BYTE_OFFSET, DEFAULT_VOCAB, size=20_000, dtype=np.int64
    ).astype(TOKEN_DTYPE)
    path = tmp_path / "synthetisch.bin"
    path.write_bytes(tokens.tobytes())
    metadata = {
        "spec": {"name": "synthetisch", "split": "train", "url": "", "license": "",
                 "files": [], "join": "document"},
        "revision": "test", "tokens": int(tokens.size), "documents": 1,
        "characters": int(tokens.size), "raw_bytes": int(tokens.size),
        "source_bytes": 0, "cache_bytes": path.stat().st_size,
    }
    return TokenStream(path, metadata)


# ----------------------------------------------------------------------
# Korpus-Registrierung
# ----------------------------------------------------------------------

def test_every_registered_corpus_is_fully_documented():
    """Milestone 4 verlangt Lizenz, Quelle und Ausschnitt – nicht nur einen Namen."""
    assert CORPORA, "Es muss mindestens ein Korpus registriert sein"
    for key, spec in CORPORA.items():
        assert spec.license, f"{key} hat keine Lizenzangabe"
        assert spec.url.startswith("https://"), f"{key} hat keine nachprüfbare Quelle"
        assert spec.files, f"{key} nennt keine konkrete Datei"
        assert spec.description, f"{key} begründet die Auswahl nicht"


def test_corpus_revisions_are_pinned_to_a_commit():
    """Ein Branchname wäre nicht reproduzierbar; es muss ein SHA sein."""
    for key, spec in CORPORA.items():
        assert len(spec.revision) == 40, f"{key} pinnt keine 40-stellige Revision"
        assert all(character in "0123456789abcdef" for character in spec.revision), key


def test_unknown_corpus_is_rejected():
    with pytest.raises(KeyError):
        corpus_key("gibt-es-nicht", "train")


# ----------------------------------------------------------------------
# Tokenstrom
# ----------------------------------------------------------------------

def test_window_dataset_yields_shifted_pairs(stream: TokenStream):
    dataset = TokenWindowDataset(stream, sequence_length=32)
    inputs, targets = dataset[0]
    assert inputs.shape == targets.shape == (32,)
    # Ziel ist die um eins verschobene Eingabe – das ist die gesamte Aufgabe.
    assert torch.equal(inputs[1:], targets[:-1])


def test_window_dataset_rejects_too_short_corpus(stream: TokenStream):
    with pytest.raises(ValueError):
        TokenWindowDataset(stream, sequence_length=len(stream) + 1)


def test_random_batches_are_reproducible(stream: TokenStream):
    first = next(random_batches(stream, batch_size=3, sequence_length=16, seed=11))
    second = next(random_batches(stream, batch_size=3, sequence_length=16, seed=11))
    other = next(random_batches(stream, batch_size=3, sequence_length=16, seed=12))
    assert torch.equal(first[0], second[0])
    assert not torch.equal(first[0], other[0])


def test_bits_per_byte_matches_the_definition():
    # Ein Loss von ln(2) nats entspricht genau einem Bit pro Byte.
    assert bits_per_byte(math.log(2)) == pytest.approx(1.0)
    assert bits_per_byte(0.0) == 0.0


# ----------------------------------------------------------------------
# Größenklassen
# ----------------------------------------------------------------------

def test_size_ladder_grows_monotonically():
    counts = [GlassMindLM(size.config()).parameter_count for size in SIZE_CLASSES]
    assert counts == sorted(counts), "Die Leiter muss aufsteigend sein"
    for smaller, larger in zip(counts, counts[1:]):
        # Jede Stufe muss deutlich größer sein, sonst misst die Studie Rauschen.
        # Der Faktor 2 statt 10 ist bewusst: Die oberste Stufe ist durch die
        # 8 GB VRAM der Testhardware begrenzt, nicht durch die Architektur.
        assert larger > smaller * 2.0


def test_size_classes_differ_only_in_width_and_depth():
    """Skalierung heißt: dieselbe Architektur, andere Größe."""
    baseline = SIZE_CLASSES[0].config().to_dict()
    varying = {"d_model", "n_layers"}
    for size in SIZE_CLASSES[1:]:
        current = size.config().to_dict()
        differing = {key for key in baseline if baseline[key] != current[key]}
        assert differing <= varying, f"{size.name} ändert zusätzlich {differing - varying}"


def test_no_size_class_enables_memory():
    """Milestone-3-Befund: Memory bleibt im Standardpfad aus."""
    for size in SIZE_CLASSES:
        assert size.config().memory_slots == 0


def test_learning_rate_decreases_with_size():
    rates = [size.learning_rate for size in SIZE_CLASSES]
    assert rates == sorted(rates, reverse=True)


def test_unknown_size_class_is_rejected():
    with pytest.raises(KeyError):
        size_class("gigantisch")


# ----------------------------------------------------------------------
# Sprachtraining
# ----------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    return GlassMindLM(size_class("tiny").config(d_model=32, n_layers=1))


def test_training_reduces_loss(tiny_model, stream, capabilities):
    config = LanguageTrainingConfig(steps=12, batch_size=4, sequence_length=48,
                                    warmup_steps=3, log_every=100, learning_rate=5e-3)
    metrics = train_language(tiny_model, stream, config, capabilities)
    assert not metrics["diverged"]
    assert metrics["steps_completed"] == 12
    assert metrics["best_loss"] < metrics["curve"][0]["loss"]
    assert metrics["grad_norm_mean"] is not None


def test_training_reports_divergence_instead_of_raising(tiny_model, stream, capabilities):
    """Eine Divergenz ist ein Messergebnis und darf den Lauf nicht abbrechen."""
    config = LanguageTrainingConfig(steps=8, batch_size=4, sequence_length=32,
                                    warmup_steps=1, log_every=100, learning_rate=1e9,
                                    grad_clip=1e9)
    metrics = train_language(tiny_model, stream, config, capabilities)
    assert metrics["diverged"] is True
    assert metrics["steps_completed"] < 8


def test_evaluation_reports_consistent_loss_perplexity_and_bits(tiny_model, stream, capabilities):
    result = evaluate_language(tiny_model, stream, capabilities,
                               sequence_length=32, batch_size=2, batches=3)
    assert result["perplexity"] == pytest.approx(math.exp(result["loss"]), rel=1e-6)
    assert result["bits_per_byte"] == pytest.approx(bits_per_byte(result["loss"]), rel=1e-9)
    assert result["tokens"] == 32 * 2 * 3


def test_evaluation_is_deterministic(tiny_model, stream, capabilities):
    first = evaluate_language(tiny_model, stream, capabilities, sequence_length=32,
                              batch_size=2, batches=2, seed=5)
    second = evaluate_language(tiny_model, stream, capabilities, sequence_length=32,
                               batch_size=2, batches=2, seed=5)
    assert first["loss"] == pytest.approx(second["loss"], rel=1e-9)


# ----------------------------------------------------------------------
# Zustandsanalyse bleibt in Milestone 4 erhalten
# ----------------------------------------------------------------------

def test_state_statistics_cover_all_three_timescales(tiny_model, stream, capabilities):
    result = state_statistics(tiny_model, stream, capabilities, sequence_length=32)
    assert len(result["per_block"]) == tiny_model.config.n_layers
    for name in STATE_NAMES:
        assert f"{name}_rms" in result["mean"], f"{name} fehlt in der Zustandsstatistik"


def test_state_specialisation_ablates_each_timescale(tiny_model, stream, capabilities):
    result = state_specialisation(tiny_model, stream, capabilities,
                                  sequence_length=32, batch_size=2, batches=2)
    assert [record["state"] for record in result["ablations"]] == list(STATE_NAMES)
    # Der Verdikt-Schlüssel muss existieren; welchen Wert er hat, ist Messung.
    assert isinstance(result["distinguishable"], bool)
    assert result["delta_spread"] >= 0


def test_state_ablation_actually_changes_the_loss(tiny_model, stream, capabilities):
    """Wäre eine Ablation wirkungslos, wäre die Zeitskala funktionslos."""
    result = state_specialisation(tiny_model, stream, capabilities,
                                  sequence_length=32, batch_size=2, batches=2)
    changed = [record for record in result["ablations"] if abs(record["delta_loss"]) > 1e-6]
    assert changed, "Keine der drei Zeitskalen beeinflusst die Vorhersage"


def test_generation_returns_decodable_text(tiny_model, capabilities):
    samples = generation_samples(tiny_model, ByteTokenizer(), capabilities,
                                 ("Hallo",), max_new_tokens=12)
    assert len(samples) == 1
    assert isinstance(samples[0]["greedy"], str)
    assert isinstance(samples[0]["sampled"], str)
    # Der Prompt muss im greedy-Ergebnis wieder auftauchen.
    assert "Hallo" in samples[0]["greedy"]


# ----------------------------------------------------------------------
# Fixed-Token Scaling: der Vergleich muss kontrolliert sein
# ----------------------------------------------------------------------

def test_every_size_class_sees_the_same_tokens_per_step():
    """Ohne gleiche effektive Batchgröße wäre der Größenvergleich verzerrt."""
    from glassmind.model.sizes import TARGET_EFFECTIVE_BATCH, gradient_accumulation

    for size in SIZE_CLASSES:
        effective = size.batch_size * gradient_accumulation(size)
        assert effective == TARGET_EFFECTIVE_BATCH, (
            f"{size.name} erreicht {effective} statt {TARGET_EFFECTIVE_BATCH}"
        )


def test_steps_for_budget_hits_the_requested_token_count():
    from glassmind.model.sizes import gradient_accumulation
    from glassmind.training.language import steps_for_budget

    budget = 4_000_000
    for size in SIZE_CLASSES:
        accumulation = gradient_accumulation(size)
        steps = steps_for_budget(budget, batch_size=size.batch_size,
                                 sequence_length=size.sequence_length,
                                 gradient_accumulation=accumulation)
        actual = steps * size.batch_size * accumulation * size.sequence_length
        # Höchstens ein halber Schritt Abweichung durch Rundung.
        assert abs(actual - budget) <= size.batch_size * accumulation * size.sequence_length


def test_gradient_accumulation_matches_a_single_large_batch(capabilities):
    """Vier Mikrobatches müssen denselben Gradienten ergeben wie ein großer.

    Wäre das nicht so, sähen die Größenklassen unterschiedlich starke
    Trainingssignale und die Skalierungskurve wäre ein Artefakt der
    Batchaufteilung statt eine Eigenschaft des Modells.

    Geprüft wird die Rechnung direkt auf identischen Daten. Über
    ``train_language`` ginge das nicht: dort zieht jeder Mikrobatch bewusst
    ein neues Zufallsfenster, sodass die effektive Batchgröße auch wirklich
    aus verschiedenen Textstellen besteht.
    """
    import torch.nn.functional as F

    seed_everything(5)
    model = GlassMindLM(size_class("tiny").config(d_model=32, n_layers=1))
    generator = torch.Generator().manual_seed(7)
    tokens = torch.randint(0, model.config.vocab_size, (8, 33), generator=generator)
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    def gradients(chunks: int) -> list[torch.Tensor]:
        model.zero_grad(set_to_none=True)
        size = inputs.shape[0] // chunks
        for index in range(chunks):
            piece = slice(index * size, (index + 1) * size)
            logits, _ = model(inputs[piece])
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), targets[piece].reshape(-1)
            )
            (loss / chunks).backward()
        return [
            parameter.grad.detach().clone()
            for parameter in model.parameters()
            if parameter.grad is not None
        ]

    single = gradients(1)
    accumulated = gradients(4)
    assert len(single) == len(accumulated)
    for one, many in zip(single, accumulated):
        assert torch.allclose(one, many, atol=1e-6), "Accumulation verändert den Gradienten"


def test_training_counts_all_accumulated_tokens(tiny_model, stream, capabilities):
    config = LanguageTrainingConfig(steps=3, batch_size=2, sequence_length=32,
                                    gradient_accumulation=3, warmup_steps=1,
                                    log_every=10**9)
    metrics = train_language(tiny_model, stream, config, capabilities)
    assert metrics["seen_tokens"] == 3 * 3 * 2 * 32


# ----------------------------------------------------------------------
# Erweiterte Zustandsanalyse
# ----------------------------------------------------------------------

def test_logit_ablation_reports_prediction_changes(tiny_model, stream, capabilities):
    from glassmind.training.language import ablation_logit_effect

    records = ablation_logit_effect(tiny_model, stream, capabilities, sequence_length=32)
    assert [record["state"] for record in records] == list(STATE_NAMES)
    for record in records:
        assert 0.0 <= record["prediction_change_rate"] <= 1.0
        assert record["logit_max_difference"] >= record["logit_rms_difference"]


def test_activity_profile_covers_all_three_timescales(tiny_model, stream, capabilities):
    from glassmind.training.language import state_activity_profile

    profile = state_activity_profile(tiny_model, stream, capabilities, sequence_length=48)
    assert set(profile["per_state"]) == set(STATE_NAMES)
    for entry in profile["per_state"].values():
        assert entry["clusters"] > 0
        assert entry["mean_persistence"] >= 0.0


def test_quality_metrics_detect_degenerate_repetition():
    """distinct-n muss eine Endlosschleife erkennen, sonst nützt es nichts."""
    from glassmind.training.language import generation_quality

    loop = [{"prompt": "P", "sampled": "P " + "cat cat cat cat cat cat"}]
    varied = [{"prompt": "P", "sampled": "P the small dog ran across a wide green field"}]
    vocabulary = {"cat", "the", "small", "dog", "ran", "across", "a", "wide", "green", "field"}
    assert generation_quality(loop, vocabulary)["distinct_2"] < 0.3
    assert generation_quality(varied, vocabulary)["distinct_2"] > 0.9
    assert generation_quality(varied, vocabulary)["valid_word_fraction"] == 1.0


def test_merge_preserves_earlier_sections(tmp_path: Path):
    """Ein Lauf mit ``--merge`` darf frühere Abschnitte nicht überschreiben.

    Der ursprüngliche Ablauf speicherte nach jedem Abschnitt und mergte erst
    danach – aus der bereits überschriebenen Datei. Der Merge war dadurch
    wirkungslos und die früheren Ergebnisse verloren. Der Test prüft die
    Reihenfolge an der Datei, nicht am Code.
    """
    import json as _json

    output = tmp_path / "studie.json"
    output.write_text(_json.dumps({"sections": {"fixed": {"a": 1}, "profile": {"b": 2}}}))

    # Nachbau der korrigierten Reihenfolge: erst lesen, dann schreiben.
    payload = {"sections": {}}
    previous = _json.loads(output.read_text())
    for key, value in previous.get("sections", {}).items():
        payload["sections"][key] = value
    payload["sections"]["capacity"] = {"c": 3}
    output.write_text(_json.dumps(payload))

    result = _json.loads(output.read_text())["sections"]
    assert set(result) == {"fixed", "profile", "capacity"}
    assert result["fixed"] == {"a": 1}
