"""Milestone 4: reproduzierbare Größenklassen für die Skalierungsstudie.

Die Leiter ist so gewählt, dass jede Stufe die vorige um etwa eine
Zehnerpotenz überschreitet und dabei *dieselbe* Architektur behält – nur
``d_model`` und ``n_layers`` ändern sich. Damit misst die Studie Skalierung
und nicht den Effekt wechselnder Bauteile.

Was bewusst **nicht** mitskaliert:

* Das Vokabular bleibt der Byte-Tokenizer mit 260 Einträgen. Die Einbettung
  ist damit auf jeder Stufe vernachlässigbar; die Parameter sitzen im Kern.
* ``state_interactions`` bleibt auf jeder Stufe aktiv, damit der Milestone-2-
  Pfad durchgehend gemessen wird.
* Der Sparse Memory bleibt auf jeder Stufe aus – siehe Milestone-3-Ergebnis.
"""
from __future__ import annotations

from dataclasses import dataclass

from glassmind.model.config import ModelConfig

#: Byte-Tokenizer-Vokabular.
DEFAULT_VOCAB = 260


@dataclass(frozen=True)
class SizeClass:
    """Eine Stufe der Skalierungsleiter."""

    name: str
    d_model: int
    n_layers: int
    #: Grobe Zielgröße aus der Milestone-4-Vorgabe, nur zur Einordnung.
    target: str
    #: Startwert für die Lernrate. Größere Modelle brauchen kleinere Schritte;
    #: die Staffelung folgt der üblichen ``1/sqrt(d_model)``-Regel, verankert
    #: an der in Milestone 1–3 bewährten Rate für d_model=64.
    learning_rate: float
    #: Ab welcher Sequenzlänge und Batchgröße die Stufe standardmäßig trainiert.
    sequence_length: int
    batch_size: int

    def config(self, *, vocab_size: int = DEFAULT_VOCAB, **overrides) -> ModelConfig:
        values = dict(
            vocab_size=vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            telemetry_clusters=4,
            state_interactions=True,
            memory_slots=0,
        )
        values.update(overrides)
        return ModelConfig(**values)


#: Effektive Batchgröße, die für *alle* Größenklassen gilt. Große Modelle
#: bekommen nur kleinere Mikrobatches in den VRAM; die Differenz gleicht
#: Gradient Accumulation aus. Ohne das wäre ein Fixed-Token-Vergleich nicht
#: kontrolliert, weil größere Modelle systematisch verrauschtere Gradienten
#: sähen – ein Effekt, der leicht als schlechtere Skalierung fehlgedeutet wird.
TARGET_EFFECTIVE_BATCH = 16

# Warum die Mikrobatches so groß wie möglich sind
# -----------------------------------------------
# Der rekurrente Kern läuft als sequentielle Python-Schleife über die Token.
# Die Zahl der Schleifendurchläufe je Optimizer-Schritt ist
#
#     seq * accum  =  seq * TOKEN/(seq * batch)  =  TOKEN / batch
#
# Die Sequenzlänge kürzt sich heraus: Die Durchläufe hängen **allein am
# Mikrobatch**. Gemessen bei Sequenz 512 und 8192 Token je Schritt (RTX 3070):
#
#     Klasse   batch 1   batch 2   batch 4   batch 8   batch 16
#     small     144       286       582      1 208      2 294  Tok/s
#     medium    107       217       442        858      1 698  Tok/s
#     large      90       171       357        725        OOM
#
# Der Zusammenhang ist linear. Die Kosten je Schleifendurchlauf und Layer sind
# mit 1,16 ms (small) und 1,11 ms (large) praktisch gleich, obwohl large 8,8-mal
# mehr Parameter hat: Die Zeit steckt im Dispatch, nicht in der Rechnung.
# Deshalb wird der Mikrobatch bis an die VRAM-Grenze gehoben, statt ihn zu
# verkleinern und mit Accumulation auszugleichen. Das kostet nichts an
# Vergleichbarkeit – die effektive Batchgröße bleibt überall 16 – und bringt je
# nach Stufe Faktor 4 bis 8.

SIZE_CLASSES: tuple[SizeClass, ...] = (
    SizeClass("tiny",   d_model=64,  n_layers=2,  target="~120K", learning_rate=3.0e-3, sequence_length=512, batch_size=16),
    SizeClass("xs",     d_model=160, n_layers=4,  target="~1,24M", learning_rate=2.0e-3, sequence_length=512, batch_size=16),
    SizeClass("small",  d_model=384, n_layers=6,  target="~10,2M", learning_rate=1.2e-3, sequence_length=512, batch_size=16),
    SizeClass("medium", d_model=640, n_layers=8,  target="~37,4M", learning_rate=8.0e-4, sequence_length=512, batch_size=16),
    # Nur diese Stufe braucht Gradient Accumulation: Batch 16 bei Sequenz 512
    # sprengt die 8 GB der RTX 3070, Batch 8 belegt gemessen 4 039 MB.
    SizeClass("large",  d_model=896, n_layers=10, target="~90,5M", learning_rate=6.0e-4, sequence_length=512, batch_size=8),
)

SIZE_BY_NAME = {size.name: size for size in SIZE_CLASSES}


def gradient_accumulation(size: SizeClass, *, target: int = TARGET_EFFECTIVE_BATCH) -> int:
    """Wie viele Mikrobatches die Stufe braucht, um ``target`` zu erreichen."""
    return max(1, target // size.batch_size)


def size_class(name: str) -> SizeClass:
    if name not in SIZE_BY_NAME:
        raise KeyError(f"Unbekannte Größenklasse {name}; bekannt: {', '.join(SIZE_BY_NAME)}")
    return SIZE_BY_NAME[name]
