from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor


@dataclass(frozen=True)
class StateTaskVocabulary:
    """Kleine, disjunkte Symbolräume für reproduzierbare Zustandsaufgaben."""

    pad: int = 0
    bos: int = 1
    eos: int = 2
    store: int = 3
    query: int = 4
    answer: int = 5
    keep: int = 6
    recall: int = 7
    key_start: int = 8
    key_count: int = 8
    value_count: int = 16
    noise_count: int = 32
    # 0 hält das Milestone-2-Vokabular unverändert. Ein positiver Wert ergänzt
    # Abschnittsmarken und eine Dokumentebene für die Kontextaufgaben.
    section_count: int = 0

    # Die Bereichsgrenzen ergeben sich aus den Zählern statt fest zu stehen.
    # Bei den bisherigen Standardwerten liefern sie exakt dieselben IDs
    # (value_start=16, noise_start=32, section_start=64) – größere
    # Schlüsselräume für die Memory-Aufgaben werden damit erst möglich.
    @property
    def value_start(self) -> int:
        return self.key_start + self.key_count

    @property
    def noise_start(self) -> int:
        return self.value_start + self.value_count

    @property
    def section_start(self) -> int:
        return self.noise_start + self.noise_count

    @property
    def document(self) -> int:
        if self.section_count <= 0:
            raise ValueError("Ohne section_count gibt es keine Dokumentebene")
        return self.section_start + self.section_count

    @property
    def has_sections(self) -> bool:
        return self.section_count > 0

    @property
    def vocab_size(self) -> int:
        if self.section_count <= 0:
            return self.section_start
        return self.section_start + self.section_count + 1

    def section(self, index: int) -> int:
        if not 0 <= index < self.section_count:
            raise ValueError(f"Abschnitt {index} liegt außerhalb von 0..{self.section_count - 1}")
        return self.section_start + index

    def token_name(self, token_id: int) -> str:
        specials = {
            self.pad: "PAD",
            self.bos: "BOS",
            self.eos: "EOS",
            self.store: "STORE",
            self.query: "QUERY",
            self.answer: "ANSWER",
            self.keep: "KEEP",
            self.recall: "RECALL",
        }
        if token_id in specials:
            return specials[token_id]
        if self.key_start <= token_id < self.key_start + self.key_count:
            return f"KEY_{token_id - self.key_start}"
        if self.value_start <= token_id < self.value_start + self.value_count:
            return f"VALUE_{token_id - self.value_start}"
        if self.noise_start <= token_id < self.noise_start + self.noise_count:
            return f"NOISE_{token_id - self.noise_start}"
        if self.has_sections:
            if self.section_start <= token_id < self.section_start + self.section_count:
                return f"SECTION_{token_id - self.section_start}"
            if token_id == self.document:
                return "DOCUMENT"
        return f"TOKEN_{token_id}"

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        reverse = {self.token_name(token_id): token_id for token_id in range(self.vocab_size)}
        try:
            tokens = [reverse[part.upper()] for part in text.split() if part]
        except KeyError as exc:
            raise ValueError(
                "State-Task-Checkpoints erwarten symbolische Tokens wie 'STORE KEY_0 VALUE_1'"
            ) from exc
        if add_bos:
            tokens.insert(0, self.bos)
        if add_eos:
            tokens.append(self.eos)
        return tokens

    def decode(self, tokens: Iterable[int], *, skip_special: bool = True) -> str:
        skipped = {self.pad, self.bos, self.eos} if skip_special else set()
        return " ".join(self.token_name(int(token)) for token in tokens if int(token) not in skipped)

    def to_dict(self) -> dict[str, int | str]:
        return {
            "kind": "state_task",
            "version": 1,
            "vocab_size": self.vocab_size,
            "key_start": self.key_start,
            "key_count": self.key_count,
            "value_count": self.value_count,
            "noise_count": self.noise_count,
            "section_count": self.section_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "StateTaskVocabulary":
        if data.get("kind") != "state_task" or int(data.get("version", -1)) != 1:
            raise ValueError("Nicht unterstützte State-Task-Tokenizer-Metadaten")
        vocabulary = cls(
            key_start=int(data.get("key_start", 8)),
            key_count=int(data.get("key_count", 8)),
            value_count=int(data.get("value_count", 16)),
            noise_count=int(data.get("noise_count", 32)),
            section_count=int(data.get("section_count", 0)),
        )
        if int(data.get("vocab_size", vocabulary.vocab_size)) != vocabulary.vocab_size:
            raise ValueError("Inkonsistente State-Task-Vokabulargröße")
        return vocabulary


@dataclass(frozen=True)
class StateTaskBatch:
    input_ids: Tensor
    targets: Tensor
    loss_mask: Tensor
    task: str
    distance: int
    answer_count: int

    def to(self, device: torch.device | str) -> "StateTaskBatch":
        return StateTaskBatch(
            self.input_ids.to(device),
            self.targets.to(device),
            self.loss_mask.to(device),
            self.task,
            self.distance,
            self.answer_count,
        )


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _noise(vocabulary: StateTaskVocabulary, count: int, generator: torch.Generator) -> list[int]:
    if count <= 0:
        return []
    return (
        torch.randint(
            vocabulary.noise_start,
            vocabulary.noise_start + vocabulary.noise_count,
            (count,),
            generator=generator,
        )
        .tolist()
    )


def generate_associative_recall_batch(
    *,
    batch_size: int,
    distance: int,
    associations: int = 3,
    seed: int = 17,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Erzeugt pro Beispiel neue, nicht aus dem Schlüssel erratbare Zuordnungen."""
    vocabulary = vocabulary or StateTaskVocabulary()
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    if not 1 <= associations <= min(vocabulary.key_count, vocabulary.value_count):
        raise ValueError("associations liegt außerhalb des verfügbaren Symbolraums")
    generator = _generator(seed)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(batch_size):
        keys = torch.randperm(vocabulary.key_count, generator=generator)[:associations] + vocabulary.key_start
        values = torch.randperm(vocabulary.value_count, generator=generator)[:associations] + vocabulary.value_start
        query_index = int(torch.randint(0, associations, (1,), generator=generator).item())
        tokens = [vocabulary.bos]
        for key, value in zip(keys.tolist(), values.tolist(), strict=True):
            tokens.extend((vocabulary.store, key, value))
        tokens.extend(_noise(vocabulary, distance, generator))
        tokens.extend(
            (
                vocabulary.query,
                int(keys[query_index]),
                vocabulary.answer,
                int(values[query_index]),
                vocabulary.eos,
            )
        )
        full = torch.tensor(tokens, dtype=torch.long)
        mask = torch.zeros(full.numel() - 1, dtype=torch.bool)
        answer_value_position = full.numel() - 2
        mask[answer_value_position - 1] = True
        sequences.append(full)
        masks.append(mask)
    stacked = torch.stack(sequences)
    return StateTaskBatch(
        input_ids=stacked[:, :-1],
        targets=stacked[:, 1:],
        loss_mask=torch.stack(masks),
        task="associative_recall",
        distance=distance,
        answer_count=1,
    )


def generate_selective_copy_batch(
    *,
    batch_size: int,
    distance: int,
    items: int = 2,
    seed: int = 23,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Verteilt KEEP-Werte deterministisch zwischen insgesamt ``distance`` Noise-Tokens."""
    vocabulary = vocabulary or StateTaskVocabulary()
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    if not 1 <= items <= vocabulary.value_count:
        raise ValueError("items liegt außerhalb des verfügbaren Werteraums")
    generator = _generator(seed)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    base_noise, remainder = divmod(distance, items + 1)
    segment_lengths = [base_noise + (1 if index < remainder else 0) for index in range(items + 1)]
    for _ in range(batch_size):
        values = torch.randperm(vocabulary.value_count, generator=generator)[:items] + vocabulary.value_start
        tokens = [vocabulary.bos]
        for index, value in enumerate(values.tolist()):
            tokens.extend(_noise(vocabulary, segment_lengths[index], generator))
            tokens.extend((vocabulary.keep, value))
        tokens.extend(_noise(vocabulary, segment_lengths[-1], generator))
        tokens.append(vocabulary.recall)
        first_answer_position = len(tokens)
        tokens.extend(values.tolist())
        tokens.append(vocabulary.eos)
        full = torch.tensor(tokens, dtype=torch.long)
        mask = torch.zeros(full.numel() - 1, dtype=torch.bool)
        for answer_position in range(first_answer_position, first_answer_position + items):
            mask[answer_position - 1] = True
        sequences.append(full)
        masks.append(mask)
    stacked = torch.stack(sequences)
    return StateTaskBatch(
        input_ids=stacked[:, :-1],
        targets=stacked[:, 1:],
        loss_mask=torch.stack(masks),
        task="selective_copy",
        distance=distance,
        answer_count=items,
    )


# ---------------------------------------------------------------------------
# Aufgaben mit mittlerer zeitlicher Struktur
#
# Die folgenden Generatoren erzeugen Abschnitte, lokale Fakten pro Abschnitt und
# Fragen, die den Abschnitt mitnennen. Entscheidend ist, dass derselbe Schlüssel
# in mehreren Abschnitten mit unterschiedlichen Werten vorkommt. Eine rein
# globale Schlüssel-Wert-Bindung kollidiert dabei zwangsläufig.
#
# Die Aufgaben sind so gebaut, dass sie *Gelegenheit* für eine mittlere
# Zeitskala schaffen. Sie erzwingen nicht, dass ``context_state`` diese Rolle
# übernimmt; welcher Zustand tatsächlich kausal beiträgt, wird gemessen.
# ---------------------------------------------------------------------------


def _require_sections(vocabulary: StateTaskVocabulary, sections: int) -> None:
    if not vocabulary.has_sections:
        raise ValueError(
            "Kontextaufgaben benötigen ein Vokabular mit section_count > 0"
        )
    if not 2 <= sections <= vocabulary.section_count:
        raise ValueError("sections liegt außerhalb des verfügbaren Abschnittsraums")


def _segment_lengths(distance: int, segments: int) -> list[int]:
    base, remainder = divmod(max(distance, 0), max(segments, 1))
    return [base + (1 if index < remainder else 0) for index in range(segments)]


def _finish(
    sequences: list[Tensor],
    masks: list[Tensor],
    *,
    task: str,
    distance: int,
    answer_count: int,
) -> StateTaskBatch:
    stacked = torch.stack(sequences)
    return StateTaskBatch(
        input_ids=stacked[:, :-1],
        targets=stacked[:, 1:],
        loss_mask=torch.stack(masks),
        task=task,
        distance=distance,
        answer_count=answer_count,
    )


def _query_block(
    vocabulary: StateTaskVocabulary,
    tokens: list[int],
    scope: int,
    key: int,
    value: int,
) -> tuple[Tensor, Tensor]:
    """Hängt ``QUERY <scope> <key> ANSWER <value> EOS`` an und liefert die Maske."""
    tokens.extend((vocabulary.query, scope, key, vocabulary.answer, value, vocabulary.eos))
    full = torch.tensor(tokens, dtype=torch.long)
    mask = torch.zeros(full.numel() - 1, dtype=torch.bool)
    # Das Zieltoken ist der Wert; vorhergesagt wird er an der ANSWER-Position.
    mask[full.numel() - 3] = True
    return full, mask


def generate_sectioned_recall_batch(
    *,
    batch_size: int,
    distance: int,
    sections: int = 3,
    facts_per_section: int = 2,
    seed: int = 31,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Abschnittslokale Fakten mit bewusst kollidierenden Schlüsseln.

    Jeder Abschnitt speichert denselben geteilten Schlüssel mit einem eigenen
    Wert. Die Frage nennt Abschnitt und Schlüssel; nur die Kombination bestimmt
    den Zielwert.
    """
    vocabulary = vocabulary or StateTaskVocabulary()
    _require_sections(vocabulary, sections)
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    if not 1 <= facts_per_section <= vocabulary.key_count:
        raise ValueError("facts_per_section liegt außerhalb des Schlüsselraums")
    if sections * facts_per_section > vocabulary.value_count:
        raise ValueError("Nicht genügend Werte für alle Abschnittsfakten")
    generator = _generator(seed)
    lengths = _segment_lengths(distance, sections)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(batch_size):
        section_ids = (torch.randperm(vocabulary.section_count, generator=generator)[:sections]).tolist()
        keys = (
            torch.randperm(vocabulary.key_count, generator=generator)[:facts_per_section]
            + vocabulary.key_start
        ).tolist()
        values = (
            torch.randperm(vocabulary.value_count, generator=generator)[
                : sections * facts_per_section
            ]
            + vocabulary.value_start
        ).tolist()
        tokens = [vocabulary.bos]
        table: dict[tuple[int, int], int] = {}
        for order, section_index in enumerate(section_ids):
            marker = vocabulary.section(section_index)
            tokens.append(marker)
            for fact, key in enumerate(keys):
                value = values[order * facts_per_section + fact]
                tokens.extend((vocabulary.store, key, value))
                table[(marker, key)] = value
            tokens.extend(_noise(vocabulary, lengths[order], generator))
        pairs = list(table)
        chosen = pairs[int(torch.randint(0, len(pairs), (1,), generator=generator).item())]
        full, mask = _query_block(vocabulary, tokens, chosen[0], chosen[1], table[chosen])
        sequences.append(full)
        masks.append(mask)
    return _finish(
        sequences, masks, task="sectioned_recall", distance=distance, answer_count=1
    )


def generate_topic_resumption_batch(
    *,
    batch_size: int,
    distance: int,
    sections: int = 2,
    seed: int = 37,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Ein früher Abschnitt wird nach einer Unterbrechung wieder aufgenommen.

    Der geteilte Schlüssel erhält in jedem Abschnittsauftritt einen anderen
    Wert. Gefragt wird nach einem der beiden Auftritte des ersten Themas, also
    über eine dazwischenliegende Unterbrechung hinweg.
    """
    vocabulary = vocabulary or StateTaskVocabulary()
    _require_sections(vocabulary, sections)
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    generator = _generator(seed)
    # Abschnittsfolge A, B, ..., A – die Wiederaufnahme ist der letzte Block.
    lengths = _segment_lengths(distance, sections + 1)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(batch_size):
        section_ids = (torch.randperm(vocabulary.section_count, generator=generator)[:sections]).tolist()
        keys = (
            torch.randperm(vocabulary.key_count, generator=generator)[:2] + vocabulary.key_start
        ).tolist()
        values = (
            torch.randperm(vocabulary.value_count, generator=generator)[: sections + 1]
            + vocabulary.value_start
        ).tolist()
        tokens = [vocabulary.bos]
        table: dict[tuple[int, int], int] = {}
        order = list(section_ids) + [section_ids[0]]
        for position, section_index in enumerate(order):
            marker = vocabulary.section(section_index)
            # Beim zweiten Auftritt des ersten Themas kommt ein neuer Schlüssel
            # hinzu; der geteilte Schlüssel bleibt abschnittsweise verschieden.
            key = keys[1] if position == len(order) - 1 else keys[0]
            value = values[position]
            tokens.extend((marker, vocabulary.store, key, value))
            table[(marker, key)] = value
            tokens.extend(_noise(vocabulary, lengths[position], generator))
        first_marker = vocabulary.section(section_ids[0])
        candidates = [pair for pair in table if pair[0] == first_marker]
        chosen = candidates[int(torch.randint(0, len(candidates), (1,), generator=generator).item())]
        full, mask = _query_block(vocabulary, tokens, chosen[0], chosen[1], table[chosen])
        sequences.append(full)
        masks.append(mask)
    return _finish(
        sequences, masks, task="topic_resumption", distance=distance, answer_count=1
    )


def generate_hierarchical_scope_batch(
    *,
    batch_size: int,
    distance: int,
    sections: int = 3,
    seed: int = 43,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Eine dokumentweite Konstante neben abschnittslokalen Fakten.

    Die Frage bezieht sich zufällig auf die globale Ebene (``DOCUMENT``) oder
    auf einen einzelnen Abschnitt. Derselbe Schlüssel trägt auf beiden Ebenen
    unterschiedliche Werte.
    """
    vocabulary = vocabulary or StateTaskVocabulary()
    _require_sections(vocabulary, sections)
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    if sections + 1 > vocabulary.value_count:
        raise ValueError("Nicht genügend Werte für globale und lokale Ebene")
    generator = _generator(seed)
    lengths = _segment_lengths(distance, sections)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(batch_size):
        section_ids = (torch.randperm(vocabulary.section_count, generator=generator)[:sections]).tolist()
        key = int(torch.randint(0, vocabulary.key_count, (1,), generator=generator).item()) + vocabulary.key_start
        values = (
            torch.randperm(vocabulary.value_count, generator=generator)[: sections + 1]
            + vocabulary.value_start
        ).tolist()
        tokens = [vocabulary.bos, vocabulary.document, vocabulary.store, key, values[0]]
        table: dict[tuple[int, int], int] = {(vocabulary.document, key): values[0]}
        for order, section_index in enumerate(section_ids):
            marker = vocabulary.section(section_index)
            tokens.extend((marker, vocabulary.store, key, values[order + 1]))
            table[(marker, key)] = values[order + 1]
            tokens.extend(_noise(vocabulary, lengths[order], generator))
        pairs = list(table)
        chosen = pairs[int(torch.randint(0, len(pairs), (1,), generator=generator).item())]
        full, mask = _query_block(vocabulary, tokens, chosen[0], chosen[1], table[chosen])
        sequences.append(full)
        masks.append(mask)
    return _finish(
        sequences, masks, task="hierarchical_scope", distance=distance, answer_count=1
    )


CONTEXT_TASK_GENERATORS = {
    "sectioned_recall": generate_sectioned_recall_batch,
    "topic_resumption": generate_topic_resumption_batch,
    "hierarchical_scope": generate_hierarchical_scope_batch,
}


# ---------------------------------------------------------------------------
# Milestone 3: Aufgaben für bounded sparse external memory
#
# Alle fünf Aufgaben teilen dieselbe Grundform: Fakten werden gespeichert, dann
# folgt eine lange Strecke irrelevanter Token, dann wird abgefragt. Sie
# unterscheiden sich darin, *welche* Anforderung sie an den Speicher stellen –
# Distanz, Anzahl, Verwechslungsgefahr, Kapazität oder Wiederverwendung.
#
# Keine der Aufgaben setzt voraus, dass ein externes Memory hilft. Ob es hilft,
# misst ``scripts/memory_study.py`` gegen dieselbe Architektur ohne Speicher.
# ---------------------------------------------------------------------------


def _store_block(vocabulary: StateTaskVocabulary, key: int, value: int) -> list[int]:
    return [vocabulary.store, key, value]


def _query_answer(
    vocabulary: StateTaskVocabulary, tokens: list[int], key: int, value: int
) -> tuple[Tensor, Tensor]:
    """Hängt ``QUERY <key> ANSWER <value> EOS`` an und markiert die Antwort."""
    tokens.extend((vocabulary.query, key, vocabulary.answer, value, vocabulary.eos))
    full = torch.tensor(tokens, dtype=torch.long)
    mask = torch.zeros(full.numel() - 1, dtype=torch.bool)
    mask[full.numel() - 3] = True
    return full, mask


def generate_delayed_binding_batch(
    *,
    batch_size: int,
    distance: int,
    seed: int = 101,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Ein einziger Fakt, dann eine sehr lange Strecke, dann die Frage.

    Die reine Distanzprobe: Es gibt nichts zu verwechseln und nichts zu
    verdrängen. Wenn ein Speicher hier nicht hilft, hilft er nirgends.
    """
    vocabulary = vocabulary or StateTaskVocabulary()
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    generator = _generator(seed)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(batch_size):
        key = int(torch.randint(0, vocabulary.key_count, (1,), generator=generator)) + vocabulary.key_start
        value = int(torch.randint(0, vocabulary.value_count, (1,), generator=generator)) + vocabulary.value_start
        tokens = [vocabulary.bos, *_store_block(vocabulary, key, value)]
        tokens.extend(_noise(vocabulary, distance, generator))
        full, mask = _query_answer(vocabulary, tokens, key, value)
        sequences.append(full)
        masks.append(mask)
    return _finish(sequences, masks, task="delayed_binding", distance=distance, answer_count=1)


def generate_multiple_bindings_batch(
    *,
    batch_size: int,
    distance: int,
    bindings: int = 4,
    seed: int = 103,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Mehrere Fakten am Anfang, eine davon wird abgefragt."""
    vocabulary = vocabulary or StateTaskVocabulary()
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    if not 1 <= bindings <= min(vocabulary.key_count, vocabulary.value_count):
        raise ValueError("bindings liegt außerhalb des verfügbaren Symbolraums")
    generator = _generator(seed)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(batch_size):
        keys = (torch.randperm(vocabulary.key_count, generator=generator)[:bindings] + vocabulary.key_start).tolist()
        values = (
            torch.randperm(vocabulary.value_count, generator=generator)[:bindings] + vocabulary.value_start
        ).tolist()
        tokens = [vocabulary.bos]
        for key, value in zip(keys, values, strict=True):
            tokens.extend(_store_block(vocabulary, key, value))
        tokens.extend(_noise(vocabulary, distance, generator))
        chosen = int(torch.randint(0, bindings, (1,), generator=generator))
        full, mask = _query_answer(vocabulary, tokens, keys[chosen], values[chosen])
        sequences.append(full)
        masks.append(mask)
    return _finish(sequences, masks, task="multiple_bindings", distance=distance, answer_count=1)


def generate_distractor_recall_batch(
    *,
    batch_size: int,
    distance: int,
    distractors: int = 6,
    seed: int = 107,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Der gesuchte Fakt steht zwischen vielen ähnlich gebauten Ablenkern.

    Alle Ablenker haben dieselbe Form ``STORE <key> <value>`` und verwenden
    denselben Wertebereich. Nur der Schlüssel unterscheidet sie – wer die
    Bindung nur grob speichert, greift daneben.
    """
    vocabulary = vocabulary or StateTaskVocabulary()
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    total = distractors + 1
    if not 1 <= total <= min(vocabulary.key_count, vocabulary.value_count):
        raise ValueError("distractors liegt außerhalb des verfügbaren Symbolraums")
    generator = _generator(seed)
    # Die Ablenker verteilen sich über die gesamte Noise-Strecke, statt sich am
    # Anfang zu sammeln.
    segments = _segment_lengths(distance, total + 1)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(batch_size):
        keys = (torch.randperm(vocabulary.key_count, generator=generator)[:total] + vocabulary.key_start).tolist()
        values = (
            torch.randperm(vocabulary.value_count, generator=generator)[:total] + vocabulary.value_start
        ).tolist()
        target = int(torch.randint(0, total, (1,), generator=generator))
        tokens = [vocabulary.bos]
        for index, (key, value) in enumerate(zip(keys, values, strict=True)):
            tokens.extend(_store_block(vocabulary, key, value))
            tokens.extend(_noise(vocabulary, segments[index], generator))
        tokens.extend(_noise(vocabulary, segments[-1], generator))
        full, mask = _query_answer(vocabulary, tokens, keys[target], values[target])
        sequences.append(full)
        masks.append(mask)
    return _finish(sequences, masks, task="distractor_recall", distance=distance, answer_count=1)


def generate_memory_replacement_batch(
    *,
    batch_size: int,
    distance: int,
    facts: int = 24,
    seed: int = 109,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Mehr Fakten als Speicherplätze; gefragt wird nach dem *letzten*.

    Bei begrenzter Kapazität muss etwas verdrängt werden. Die Frage zielt auf
    den zuletzt gespeicherten Fakt – wer sinnvoll verdrängt, behält ihn.
    """
    vocabulary = vocabulary or StateTaskVocabulary()
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    if not 2 <= facts <= min(vocabulary.key_count, vocabulary.value_count):
        raise ValueError("facts liegt außerhalb des verfügbaren Symbolraums")
    generator = _generator(seed)
    segments = _segment_lengths(distance, facts)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(batch_size):
        keys = (torch.randperm(vocabulary.key_count, generator=generator)[:facts] + vocabulary.key_start).tolist()
        values = (
            torch.randperm(vocabulary.value_count, generator=generator)[:facts] + vocabulary.value_start
        ).tolist()
        tokens = [vocabulary.bos]
        for index, (key, value) in enumerate(zip(keys, values, strict=True)):
            tokens.extend(_store_block(vocabulary, key, value))
            tokens.extend(_noise(vocabulary, segments[index], generator))
        full, mask = _query_answer(vocabulary, tokens, keys[-1], values[-1])
        sequences.append(full)
        masks.append(mask)
    return _finish(sequences, masks, task="memory_replacement", distance=distance, answer_count=1)


def generate_repeated_retrieval_batch(
    *,
    batch_size: int,
    distance: int,
    retrievals: int = 3,
    seed: int = 113,
    vocabulary: StateTaskVocabulary | None = None,
) -> StateTaskBatch:
    """Derselbe Fakt wird mehrfach mit großem Abstand abgefragt.

    Damit lässt sich prüfen, ob Lesezugriffe oder Stärke etwas bewirken: Ein
    mehrfach gelesener Eintrag sollte nicht verdrängt werden. Alle Abfragen
    zählen zur Bewertung, nicht nur die letzte.
    """
    vocabulary = vocabulary or StateTaskVocabulary()
    if batch_size < 1 or distance < 0:
        raise ValueError("batch_size muss positiv und distance nichtnegativ sein")
    if retrievals < 1:
        raise ValueError("retrievals muss mindestens 1 sein")
    generator = _generator(seed)
    segments = _segment_lengths(distance, retrievals)
    sequences: list[Tensor] = []
    masks: list[Tensor] = []
    for _ in range(batch_size):
        key = int(torch.randint(0, vocabulary.key_count, (1,), generator=generator)) + vocabulary.key_start
        value = int(torch.randint(0, vocabulary.value_count, (1,), generator=generator)) + vocabulary.value_start
        tokens = [vocabulary.bos, *_store_block(vocabulary, key, value)]
        answer_positions: list[int] = []
        for index in range(retrievals):
            tokens.extend(_noise(vocabulary, segments[index], generator))
            tokens.extend((vocabulary.query, key, vocabulary.answer))
            answer_positions.append(len(tokens))
            tokens.append(value)
        tokens.append(vocabulary.eos)
        full = torch.tensor(tokens, dtype=torch.long)
        mask = torch.zeros(full.numel() - 1, dtype=torch.bool)
        for position in answer_positions:
            mask[position - 1] = True
        sequences.append(full)
        masks.append(mask)
    return _finish(
        sequences, masks, task="repeated_retrieval", distance=distance, answer_count=retrievals
    )


MEMORY_TASK_GENERATORS = {
    "delayed_binding": generate_delayed_binding_batch,
    "multiple_bindings": generate_multiple_bindings_batch,
    "distractor_recall": generate_distractor_recall_batch,
    "memory_replacement": generate_memory_replacement_batch,
    "repeated_retrieval": generate_repeated_retrieval_batch,
}


def memory_task_vocabulary() -> StateTaskVocabulary:
    """Größerer Schlüsselraum, damit „mehr Fakten als Slots" überhaupt möglich ist."""
    return StateTaskVocabulary(key_count=48, value_count=32, noise_count=32)
