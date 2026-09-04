"""Tests für die Aufgaben mit mittlerer zeitlicher Struktur und für ``context_state``.

Wichtig für die Ehrlichkeit dieser Suite: Kein Test verlangt, dass
``context_state`` eine bestimmte kausale Bedeutung *hat*. Geprüft wird, dass

* die Aufgaben korrekt, reproduzierbar und lösbar sind,
* ``context_state`` überhaupt an den Ausgang angeschlossen ist,
* seine Zeitskala sich messbar von ``fast_state`` unterscheidet,
* und dass die Ablationsmaschinerie für ``context`` genauso funktioniert wie
  für die anderen Zustände.

Wie groß der kausale Beitrag ausfällt, ist ein Messergebnis und steht in
``runs/context-specialization-*/summary.md``, nicht in einer Testschwelle.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from glassmind.analysis import ablation_comparison
from glassmind.data import (
    CONTEXT_TASK_GENERATORS,
    StateTaskVocabulary,
    generate_hierarchical_scope_batch,
    generate_sectioned_recall_batch,
    generate_topic_resumption_batch,
)
from glassmind.model import GlassMindLM, ModelConfig

VOCABULARY = StateTaskVocabulary(section_count=4)


def _context_model() -> GlassMindLM:
    torch.manual_seed(17)
    return GlassMindLM(
        ModelConfig(
            vocab_size=VOCABULARY.vocab_size,
            d_model=32,
            n_layers=1,
            telemetry_clusters=4,
            state_interactions=True,
        )
    )


def _scope_key_value(batch, row: int) -> tuple[int, int, int]:
    """Liest QUERY <scope> <key> ANSWER <value> aus einem erzeugten Beispiel."""
    tokens = batch.input_ids[row].tolist()
    query_position = len(tokens) - 1 - tokens[::-1].index(VOCABULARY.query)
    scope, key = tokens[query_position + 1], tokens[query_position + 2]
    answer_position = int(batch.loss_mask[row].nonzero()[0])
    return scope, key, int(batch.targets[row, answer_position])


def test_default_vocabulary_stays_compatible_with_milestone_two() -> None:
    plain = StateTaskVocabulary()
    assert plain.vocab_size == 64
    assert not plain.has_sections
    assert VOCABULARY.vocab_size == 69
    assert VOCABULARY.token_name(VOCABULARY.section(1)) == "SECTION_1"
    assert VOCABULARY.token_name(VOCABULARY.document) == "DOCUMENT"
    assert StateTaskVocabulary.from_dict(VOCABULARY.to_dict()) == VOCABULARY


def test_context_generators_are_reproducible_and_masked_once() -> None:
    for name, generator in CONTEXT_TASK_GENERATORS.items():
        first = generator(batch_size=5, distance=24, seed=61, vocabulary=VOCABULARY)
        second = generator(batch_size=5, distance=24, seed=61, vocabulary=VOCABULARY)
        assert first.task == name
        assert torch.equal(first.input_ids, second.input_ids)
        assert torch.equal(first.targets, second.targets)
        assert torch.equal(first.loss_mask, second.loss_mask)
        assert int(first.loss_mask.sum()) == 5, name
        different = generator(batch_size=5, distance=24, seed=62, vocabulary=VOCABULARY)
        assert not torch.equal(first.input_ids, different.input_ids)


def test_query_target_matches_the_scoped_binding() -> None:
    """Der Zielwert muss genau der im gefragten Abschnitt gespeicherte sein."""
    for generator in CONTEXT_TASK_GENERATORS.values():
        batch = generator(batch_size=8, distance=0, seed=71, vocabulary=VOCABULARY)
        for row in range(batch.input_ids.shape[0]):
            scope, key, target = _scope_key_value(batch, row)
            tokens = batch.input_ids[row].tolist()
            store_positions = [
                index
                for index, token in enumerate(tokens)
                if token == VOCABULARY.store and tokens[index + 1] == key
            ]
            # Der letzte STORE vor der Frage, dessen Abschnittsmarke der
            # gefragte Scope ist, definiert die Antwort.
            resolved = None
            for position in store_positions:
                markers = [
                    token
                    for token in tokens[:position]
                    if token >= VOCABULARY.section_start
                ]
                if markers and markers[-1] == scope:
                    resolved = tokens[position + 2]
            assert resolved == target, (batch.task, row, scope, key)


def test_same_key_carries_different_values_across_sections() -> None:
    """Ohne kollidierende Schlüssel wäre die Abschnittsstruktur bedeutungslos."""
    batch = generate_sectioned_recall_batch(
        batch_size=6, distance=0, sections=3, facts_per_section=2, seed=83, vocabulary=VOCABULARY
    )
    for row in range(batch.input_ids.shape[0]):
        tokens = batch.input_ids[row].tolist()
        per_key: dict[int, set[int]] = {}
        for index, token in enumerate(tokens):
            if token == VOCABULARY.store:
                per_key.setdefault(tokens[index + 1], set()).add(tokens[index + 2])
        assert per_key, row
        assert all(len(values) > 1 for values in per_key.values()), (row, per_key)


def test_topic_resumption_returns_to_an_earlier_section() -> None:
    batch = generate_topic_resumption_batch(
        batch_size=6, distance=8, sections=2, seed=89, vocabulary=VOCABULARY
    )
    for row in range(batch.input_ids.shape[0]):
        markers = [
            token for token in batch.input_ids[row].tolist() if token >= VOCABULARY.section_start
        ]
        # Zuletzt genannte Abschnittsmarke vor der Frage ist die Wiederaufnahme
        # des ersten Themas; die Frage nennt denselben Abschnitt.
        assert markers[0] == markers[-2] == markers[-1], (row, markers)


def test_hierarchical_scope_mixes_document_and_section_facts() -> None:
    batch = generate_hierarchical_scope_batch(
        batch_size=16, distance=8, sections=3, seed=97, vocabulary=VOCABULARY
    )
    scopes = {_scope_key_value(batch, row)[0] for row in range(batch.input_ids.shape[0])}
    assert VOCABULARY.document in scopes
    assert any(scope != VOCABULARY.document for scope in scopes)


def test_context_tasks_can_be_overfit() -> None:
    """Die Aufgaben müssen prinzipiell lösbar sein, sonst misst die Ablation nichts."""
    model = _context_model().train()
    batches = [
        generator(batch_size=8, distance=4, seed=101, vocabulary=VOCABULARY)
        for generator in CONTEXT_TASK_GENERATORS.values()
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    for step in range(320):
        batch = batches[step % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(batch.input_ids)
        loss = F.cross_entropy(logits[batch.loss_mask], batch.targets[batch.loss_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for batch in batches:
            logits, _ = model(batch.input_ids)
            accuracy = (
                logits[batch.loss_mask].argmax(dim=-1) == batch.targets[batch.loss_mask]
            ).float().mean()
            assert float(accuracy) >= 0.95, batch.task


def test_context_state_is_connected_and_ablatable() -> None:
    """``context`` muss den Ausgang erreichen und sich sauber ablatieren lassen.

    Der Test fordert keine bestimmte Effektgröße. Er stellt nur sicher, dass der
    Zustand nicht versehentlich vom Ausgang abgekoppelt ist – sonst wäre jede
    Ablationsmessung trivial null und damit nichtssagend.
    """
    model = _context_model().eval()
    batch = generate_sectioned_recall_batch(
        batch_size=4, distance=12, seed=113, vocabulary=VOCABULARY
    )
    comparison = ablation_comparison(model, batch, ("context",))
    assert comparison["logit_rms_difference"] > 0.0
    assert comparison["ablated_states"] == ["context"]
    with torch.no_grad():
        full, final_state = model(batch.input_ids, ablate_states=("context",))
        assert torch.count_nonzero(final_state.blocks[0].context) == 0
        state = None
        streamed = []
        for index in range(batch.input_ids.shape[1]):
            logits, state = model.step(
                batch.input_ids[:, index], state, ablate_states=("context",)
            )
            streamed.append(logits)
    assert torch.allclose(full, torch.stack(streamed, dim=1), atol=2e-5, rtol=2e-5)


def test_context_state_moves_on_a_slower_timescale_than_fast_state() -> None:
    """Architektureigenschaft, keine Bedeutungsbehauptung.

    ``fast_bias`` und ``context_bias`` setzen unterschiedliche Startzeitskalen.
    Der Test prüft, dass sich das in der realen Zustandsdynamik zeigt.
    """
    model = _context_model().eval()
    batch = generate_topic_resumption_batch(
        batch_size=4, distance=32, seed=127, vocabulary=VOCABULARY
    )
    block = model.blocks[0]
    with torch.no_grad():
        embedded = model.embedding(batch.input_ids)
        mixed, _ = model.local_mixer(embedded)
        _, _, metrics = block(mixed, collect_metrics=True)
    assert metrics is not None and len(metrics) == batch.input_ids.shape[1]
    fast_change = torch.stack([metric.fast_delta.abs().mean() for metric in metrics]).mean()
    context_change = torch.stack([metric.context_delta.abs().mean() for metric in metrics]).mean()
    assert float(context_change) < float(fast_change)
