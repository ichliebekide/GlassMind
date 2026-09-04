"""Milestone 4: echte Sprachkorpora vorbereiten, cachen und lesen.

Der Weg ist bewusst schlicht und reproduzierbar: Jede Quelle wird über eine
*festgepinnte* Revision und namentlich genannte Dateien geladen, einmal in
einen flachen Tokenstrom übersetzt und als ``uint16``-Memmap abgelegt. Danach
ist jede Sequenzlänge nur noch eine Sicht auf denselben Puffer – neu
tokenisiert wird nie.

Warum Byte-Tokenizer statt BPE
------------------------------
GlassMind trägt seit Milestone 1 einen deterministischen UTF-8-Byte-Tokenizer.
Er bleibt auch hier die Grundlage, aus drei nachprüfbaren Gründen:

* Keine zusätzliche Abhängigkeit und kein zweites Vokabularartefakt, das in
  Checkpoints, Precision-Policy und Replay mitgeführt werden müsste.
* Das Vokabular ist mit 260 Einträgen so klein, dass die Einbettung selbst bei
  100M Parametern praktisch nichts kostet. Der Skalierungsbefund misst damit
  den *Kern*, nicht die Größe einer Vokabularmatrix.
* Ein Token entspricht einem Byte. Eine Sequenz von 32768 Token ist damit
  wörtlich 32 KB Text – die Aussage über Streaming bei langen Kontexten ist so
  nicht von einer Tokenisierungskonvention abhängig.

Der Preis steht ausdrücklich dabei: Byteweise Modellierung braucht mehr
Schritte pro Wort, und die Perplexity ist **nicht** mit wortbasierten
Literaturwerten vergleichbar. Deshalb wird zusätzlich ``bits_per_byte``
berichtet, das über Tokenisierungen hinweg vergleichbar ist.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from glassmind.data.tokenizer import ByteTokenizer

#: Der Tokenstrom liegt als ``uint16`` vor: Das Byte-Vokabular umfasst 260
#: Einträge und passt damit nicht mehr in ``uint8``.
TOKEN_DTYPE = np.uint16

DEFAULT_CACHE = Path("data/cache")


@dataclass(frozen=True)
class CorpusSpec:
    """Eine benannte, festgepinnte Datenquelle."""

    name: str
    split: str
    repo_id: str
    revision: str
    files: tuple[str, ...]
    text_column: str
    license: str
    url: str
    #: Trennzeichen zwischen zwei Datensätzen. ``document`` setzt ein
    #: EOS-Token, ``line`` hängt nur einen Zeilenumbruch an.
    join: str = "document"
    description: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "split": self.split,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "files": list(self.files),
            "text_column": self.text_column,
            "license": self.license,
            "url": self.url,
            "join": self.join,
            "description": self.description,
        }


# ----------------------------------------------------------------------
# Registrierte Korpora
# ----------------------------------------------------------------------
# Die Revisionen sind bewusst als Commit-SHA eingetragen. Ein späterer Lauf
# lädt damit exakt dieselben Bytes, auch wenn der Datensatz auf dem Hub
# weiterentwickelt wird.

_TINYSTORIES_REV = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
_WIKITEXT_REV = "b08601e04326c79dfdd32d625aee71d232d685c3"

CORPORA: dict[str, CorpusSpec] = {
    spec.name + "/" + spec.split: spec
    for spec in (
        CorpusSpec(
            name="tinystories",
            split="train",
            repo_id="roneneldan/TinyStories",
            revision=_TINYSTORIES_REV,
            files=("data/train-00000-of-00004-2d5a1467fff1081b.parquet",),
            text_column="text",
            license="CDLA-Sharing-1.0",
            url="https://huggingface.co/datasets/roneneldan/TinyStories",
            join="document",
            description=(
                "Kurze, synthetisch erzeugte Kindergeschichten mit stark "
                "begrenztem Wortschatz. Gewählt, weil kleine Modelle darauf "
                "nachweisbar zusammenhängende Sprache lernen – die Frage "
                "'lernt GlassMind stabile Sprache?' wird damit überhaupt "
                "beantwortbar, statt an der Schwierigkeit des Korpus zu "
                "scheitern. Es wird nur der erste von vier Trainings-Shards "
                "verwendet; mehr Text als die Skalierungsleiter verbraucht "
                "bringt keinen zusätzlichen Erkenntniswert."
            ),
        ),
        CorpusSpec(
            name="tinystories",
            split="validation",
            repo_id="roneneldan/TinyStories",
            revision=_TINYSTORIES_REV,
            files=("data/validation-00000-of-00001-869c898b519ad725.parquet",),
            text_column="text",
            license="CDLA-Sharing-1.0",
            url="https://huggingface.co/datasets/roneneldan/TinyStories",
            join="document",
            description="Offizieller Validierungssplit von TinyStories.",
        ),
        CorpusSpec(
            name="wikitext103",
            split="train",
            repo_id="Salesforce/wikitext",
            revision=_WIKITEXT_REV,
            files=(
                "wikitext-103-raw-v1/train-00000-of-00002.parquet",
                "wikitext-103-raw-v1/train-00001-of-00002.parquet",
            ),
            text_column="text",
            license="CC BY-SA 3.0 / GFDL",
            url="https://huggingface.co/datasets/Salesforce/wikitext",
            join="line",
            description=(
                "Bereinigte Wikipedia-Artikel, der klassische Maßstab für "
                "Sprachmodell-Perplexity. Gewählt als *harter* Gegenpol zu "
                "TinyStories: echte Enzyklopädiesprache mit Eigennamen, "
                "Zahlen und Markup-Resten. Die Rohvariante (``raw``) wird "
                "verwendet, weil sie nicht vorab durch ein Wortvokabular "
                "gefiltert ist und damit zu einem Byte-Tokenizer passt."
            ),
        ),
        CorpusSpec(
            name="wikitext103",
            split="validation",
            repo_id="Salesforce/wikitext",
            revision=_WIKITEXT_REV,
            files=("wikitext-103-raw-v1/validation-00000-of-00001.parquet",),
            text_column="text",
            license="CC BY-SA 3.0 / GFDL",
            url="https://huggingface.co/datasets/Salesforce/wikitext",
            join="line",
            description="Offizieller Validierungssplit von WikiText-103.",
        ),
        CorpusSpec(
            name="wikitext103",
            split="test",
            repo_id="Salesforce/wikitext",
            revision=_WIKITEXT_REV,
            files=("wikitext-103-raw-v1/test-00000-of-00001.parquet",),
            text_column="text",
            license="CC BY-SA 3.0 / GFDL",
            url="https://huggingface.co/datasets/Salesforce/wikitext",
            join="line",
            description="Offizieller Testsplit von WikiText-103.",
        ),
    )
}


def corpus_key(name: str, split: str) -> str:
    key = f"{name}/{split}"
    if key not in CORPORA:
        raise KeyError(f"Unbekannter Korpus {key}; bekannt: {', '.join(sorted(CORPORA))}")
    return key


# ----------------------------------------------------------------------
# Vorbereitung
# ----------------------------------------------------------------------

def _iter_text(paths: Sequence[Path], column: str) -> Iterator[str]:
    """Liest die Textspalte batchweise, ohne die Datei ganz zu materialisieren."""
    import pyarrow.parquet as pq

    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=8192, columns=[column]):
            for value in batch.column(0):
                text = value.as_py()
                if text:
                    yield text


def prepare_corpus(
    name: str,
    split: str,
    *,
    cache_dir: str | Path = DEFAULT_CACHE,
    max_tokens: int | None = None,
    force: bool = False,
    progress: object | None = None,
) -> "TokenStream":
    """Lädt, tokenisiert und cached einen Korpus; gibt den fertigen Strom zurück.

    Ein bereits vorhandener Cache mit passenden Metadaten wird wiederverwendet.
    Der Download selbst übernimmt ``huggingface_hub`` und landet in dessen
    eigenem Cache – die Parquet-Dateien werden hier nicht kopiert.
    """
    spec = CORPORA[corpus_key(name, split)]
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    suffix = "" if max_tokens is None else f"-{max_tokens}"
    binary = cache / f"{name}-{split}{suffix}.bin"
    sidecar = binary.with_suffix(".json")

    if binary.exists() and sidecar.exists() and not force:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if metadata.get("revision") == spec.revision:
            return TokenStream(binary, metadata)

    from huggingface_hub import hf_hub_download

    paths = [
        Path(
            hf_hub_download(
                repo_id=spec.repo_id,
                filename=filename,
                revision=spec.revision,
                repo_type="dataset",
            )
        )
        for filename in spec.files
    ]
    source_bytes = sum(path.stat().st_size for path in paths)

    tokenizer = ByteTokenizer()
    written = 0
    documents = 0
    characters = 0
    raw_bytes = 0
    buffer = bytearray()
    with binary.open("wb") as handle:
        for text in _iter_text(paths, spec.text_column):
            documents += 1
            characters += len(text)
            payload = text.encode("utf-8")
            raw_bytes += len(payload)
            chunk = np.frombuffer(payload, dtype=np.uint8).astype(TOKEN_DTYPE)
            chunk = chunk + ByteTokenizer.BYTE_OFFSET
            if spec.join == "document":
                chunk = np.append(chunk, TOKEN_DTYPE(ByteTokenizer.EOS))
            if max_tokens is not None and written + chunk.size > max_tokens:
                chunk = chunk[: max_tokens - written]
            buffer.extend(chunk.tobytes())
            written += chunk.size
            if len(buffer) >= 32 << 20:
                handle.write(buffer)
                buffer.clear()
                if progress is not None:
                    progress.log(f"[korpus] {name}/{split}: {written:,} Token geschrieben")
            if max_tokens is not None and written >= max_tokens:
                break
        if buffer:
            handle.write(buffer)

    metadata = {
        "spec": spec.to_dict(),
        "revision": spec.revision,
        "tokenizer": tokenizer.to_dict(),
        "tokens": written,
        "documents": documents,
        "characters": characters,
        "raw_bytes": raw_bytes,
        "source_bytes": source_bytes,
        "cache_bytes": binary.stat().st_size,
        "max_tokens": max_tokens,
        "token_dtype": "uint16",
    }
    sidecar.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return TokenStream(binary, metadata)


@dataclass
class TokenStream:
    """Ein fertig tokenisierter Korpus als Memmap."""

    path: Path
    metadata: dict[str, object]
    _tokens: np.ndarray | None = field(default=None, repr=False)

    @property
    def tokens(self) -> np.ndarray:
        if self._tokens is None:
            self._tokens = np.memmap(self.path, dtype=TOKEN_DTYPE, mode="r")
        return self._tokens

    def __len__(self) -> int:
        return int(self.metadata["tokens"])

    def summary(self) -> dict[str, object]:
        spec = self.metadata["spec"]
        return {
            "name": spec["name"],
            "split": spec["split"],
            "quelle": spec["url"],
            "revision": self.metadata["revision"][:12],
            "lizenz": spec["license"],
            "dateien": spec["files"],
            "quellgroesse_bytes": self.metadata["source_bytes"],
            "dokumente": self.metadata["documents"],
            "zeichen": self.metadata["characters"],
            "token": self.metadata["tokens"],
            "cache_bytes": self.metadata["cache_bytes"],
        }


class TokenWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Sicht auf einen Tokenstrom als überlappungsfreie Fenster fester Länge."""

    def __init__(
        self,
        stream: TokenStream,
        *,
        sequence_length: int,
        stride: int | None = None,
        limit: int | None = None,
    ) -> None:
        if len(stream) <= sequence_length:
            raise ValueError(
                f"Korpus hat nur {len(stream)} Token, gebraucht werden mehr als {sequence_length}"
            )
        self.stream = stream
        self.sequence_length = sequence_length
        self.stride = stride or sequence_length
        count = (len(stream) - sequence_length - 1) // self.stride + 1
        self.count = count if limit is None else min(count, limit)

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        start = index * self.stride
        window = np.asarray(
            self.stream.tokens[start : start + self.sequence_length + 1], dtype=np.int64
        )
        chunk = torch.from_numpy(window)
        return chunk[:-1], chunk[1:]


def random_batches(
    stream: TokenStream,
    *,
    batch_size: int,
    sequence_length: int,
    seed: int = 17,
    steps: int | None = None,
) -> Iterator[tuple[Tensor, Tensor]]:
    """Zufällige Fenster – der übliche Weg, einen großen Korpus zu lesen.

    Zufällige Startpunkte statt einer festen Reihenfolge, damit ein Lauf nicht
    an der Reihenfolge der Quelldateien hängt. Der Generator ist über ``seed``
    reproduzierbar.
    """
    tokens = stream.tokens
    high = len(stream) - sequence_length - 1
    if high <= 0:
        raise ValueError("Korpus ist für diese Sequenzlänge zu kurz")
    generator = np.random.default_rng(seed)
    produced = 0
    while steps is None or produced < steps:
        starts = generator.integers(0, high, size=batch_size)
        window = np.stack(
            [np.asarray(tokens[start : start + sequence_length + 1], dtype=np.int64) for start in starts]
        )
        chunk = torch.from_numpy(window)
        yield chunk[:, :-1], chunk[:, 1:]
        produced += 1


def bits_per_byte(cross_entropy_nats: float) -> float:
    """Wandelt den Byte-Level-Loss in bits/byte um.

    Anders als Perplexity ist bits/byte über Tokenisierungen hinweg
    vergleichbar und damit die ehrlichere Zahl für einen Byte-Tokenizer.
    """
    return cross_entropy_nats / math.log(2)
