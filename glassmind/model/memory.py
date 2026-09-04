"""Bounded Sparse External Memory für GlassMind (Milestone 3).

Motivation aus der Messung, nicht aus Architekturbegeisterung: Milestone 2.6
zeigte, dass der ``semantic_state`` – der Bindungsakkumulator des State Core –
jenseits der trainierten Distanz sättigt. Er addiert jeden Schreibvorgang auf
denselben Zustand, und über tausende Token überlagern sich die Beiträge. Ein
Speicher mit diskreten Slots hat dieses Problem nicht: Er überschreibt gezielt,
statt zu akkumulieren. Ob das den Langzeitabruf tatsächlich verbessert, ist die
Frage, die Milestone 3 beantwortet – nicht voraussetzt.

Aufbau:

* Der Speicher ist **begrenzt**. Es gibt genau ``slots`` Plätze, nie mehr.
* Lesen und Schreiben sind **sparse**: Nur ``read_k`` beziehungsweise
  ``write_k`` Slots werden angefasst.
* Das Ähnlichkeitsscoring läuft über alle Slots. Bei 64–128 Slots ist das eine
  einzige kleine Reduktion und schneller als jede echte Suchstruktur; die
  Messung dazu steht in ``benchmarks/milestone3-memory.json``.
Abgrenzung zur Self-Attention – bewusst genau formuliert, weil eine bequeme
Formulierung hier irreführend wäre:

Der Lesevorgang **hat** die Form einer Attention-Operation: eine Query, ein
Satz Schlüssel, eine gewichtete Wertesumme. Das zu verschweigen wäre unehrlich.
Der Unterschied liegt nicht in der Form, sondern darin, *worüber* attendiert
wird:

* Self-Attention bildet Token gegen Token ab. Ihre Score-Matrix hat die Form
  ``[batch, sequence, sequence]`` und wächst quadratisch; ihr Cache wächst mit
  jedem Token.
* Dieser Speicher bildet eine Query gegen ``slots`` feste Plätze ab. Die
  Score-Achse hat die Form ``[batch, slots]`` und ist von der Sequenzlänge
  unabhängig; der Speicher verdrängt, statt zu wachsen.

Damit entsteht weder eine Token-zu-Token-Matrix noch ein mit der Sequenz
wachsender Cache. ``tests/test_architecture_invariants.py`` prüft beides an
jeder einzelnen erzeugten Zwischengröße, statt es zu behaupten.

Der Speicherinhalt ist **Laufzeitzustand**, kein Modellgewicht – genauso wie
``fast_state``. Gewichte sind nur die Projektionen.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from glassmind.precision.policy import PrecisionPolicy
from glassmind.precision.quantization import linear_weight

#: Woraus die Query gebildet wird. Welche Quelle gewinnt, entscheidet die
#: Messung in ``scripts/memory_study.py``.
QUERY_SOURCES = ("output", "fast", "context", "semantic", "fast_context", "context_semantic")

#: Wie die Lesescores gebildet werden. Beide Varianten sind reine Vergleiche
#: einer Query mit gespeicherten Schlüsseln – keine Token-zu-Token-Matrix.
ROUTING_MODES = ("cosine", "cosine_strength")

#: Auswahlkriterien für den zu überschreibenden Slot bei vollem Speicher.
REPLACEMENT_POLICIES = ("age", "strength", "usage", "lru_strength", "learned")

#: Analyseoperationen auf einzelnen Slots. Sie wirken nur auf den Laufzeit-
#: zustand und verändern keine Gewichte.
INTERVENTIONS = ("clear", "freeze", "mute_read", "mute_write")


@dataclass
class MemoryState:
    """Laufzeitinhalt des Speichers – pro Batch-Eintrag eine eigene Bank."""

    keys: Tensor          # [batch, slots, key_dim]
    values: Tensor        # [batch, slots, width]
    strength: Tensor      # [batch, slots]
    age: Tensor           # [batch, slots] – Token seit dem letzten Schreiben
    usage_count: Tensor   # [batch, slots] – Lese- plus Schreibzugriffe
    read_count: Tensor    # [batch, slots]
    write_count: Tensor   # [batch, slots]
    last_read_step: Tensor   # [batch, slots] – -1 heißt: noch nie gelesen
    last_write_step: Tensor  # [batch, slots]
    occupied: Tensor      # [batch, slots] – 1.0 sobald ein Slot beschrieben wurde
    step: int = 0

    @property
    def slots(self) -> int:
        return self.keys.shape[1]

    @property
    def batch(self) -> int:
        return self.keys.shape[0]

    def detach(self) -> "MemoryState":
        return MemoryState(
            *(
                getattr(self, name).detach()
                for name in (
                    "keys", "values", "strength", "age", "usage_count",
                    "read_count", "write_count", "last_read_step",
                    "last_write_step", "occupied",
                )
            ),
            step=self.step,
        )

    def to(self, device: torch.device | str) -> "MemoryState":
        return MemoryState(
            *(
                getattr(self, name).to(device)
                for name in (
                    "keys", "values", "strength", "age", "usage_count",
                    "read_count", "write_count", "last_read_step",
                    "last_write_step", "occupied",
                )
            ),
            step=self.step,
        )

    def slot_summary(self, slot: int, batch_index: int = 0) -> dict[str, float]:
        """Einzelwerte eines Slots – für Analyse und Anzeige."""
        return {
            "slot": float(slot),
            "strength": float(self.strength[batch_index, slot]),
            "age": float(self.age[batch_index, slot]),
            "usage_count": float(self.usage_count[batch_index, slot]),
            "read_count": float(self.read_count[batch_index, slot]),
            "write_count": float(self.write_count[batch_index, slot]),
            "last_read_step": float(self.last_read_step[batch_index, slot]),
            "last_write_step": float(self.last_write_step[batch_index, slot]),
            "occupied": float(self.occupied[batch_index, slot]),
            "value_norm": float(torch.linalg.vector_norm(self.values[batch_index, slot].float())),
        }


@dataclass
class MemoryStepMetrics:
    """Echte Messwerte eines einzelnen Speicherzugriffs."""

    read_slots: Tensor       # [batch, read_k]
    read_scores: Tensor      # [batch, read_k]
    read_weights: Tensor     # [batch, read_k]
    read_output: Tensor      # [batch, d_model]
    read_gate: Tensor        # [batch, 1]
    write_slots: Tensor      # [batch, write_k]
    write_scores: Tensor     # [batch, write_k] – Ersetzungsscore des gewählten Slots
    write_strength: Tensor   # [batch, write_k]
    write_gate: Tensor       # [batch, 1]
    replaced: Tensor         # [batch, write_k] – 1.0, wenn ein belegter Slot überschrieben wurde
    slot_strength: Tensor    # [batch, slots]
    slot_age: Tensor         # [batch, slots]
    slot_usage: Tensor       # [batch, slots]
    slot_read_count: Tensor  # [batch, slots]
    slot_write_count: Tensor # [batch, slots]
    slot_occupied: Tensor    # [batch, slots]
    query: Tensor            # [batch, key_dim]
    all_scores: Tensor       # [batch, slots]


class SparseMemory(nn.Module):
    """Begrenzter Speicher mit sparsamem Lesen und Schreiben.

    Der Speicher wird als eigene Stufe zwischen zwei Blöcken ausgeführt. Ist er
    abgeschaltet, ruft ``GlassMindLM`` ihn gar nicht erst auf – der bestehende
    Pfad bleibt damit bitgleich und exakt gleich schnell.
    """

    def __init__(
        self,
        d_model: int,
        *,
        slots: int = 64,
        width: int = 64,
        key_dim: int = 32,
        read_k: int = 2,
        write_k: int = 1,
        replacement: str = "lru_strength",
        query_source: str = "output",
        routing: str = "cosine_strength",
        decay: float = 0.999,
        source_width: int | None = None,
        track_usage: bool = False,
        precision: PrecisionPolicy | None = None,
    ) -> None:
        super().__init__()
        if slots < 0:
            raise ValueError("slots darf nicht negativ sein")
        if width < 1 or key_dim < 1:
            raise ValueError("width und key_dim müssen mindestens 1 sein")
        if replacement not in REPLACEMENT_POLICIES:
            raise ValueError(
                f"Unbekannte Replacement-Policy {replacement!r}; "
                f"erlaubt: {', '.join(REPLACEMENT_POLICIES)}"
            )
        if query_source not in QUERY_SOURCES:
            raise ValueError(
                f"Unbekannte Query-Quelle {query_source!r}; erlaubt: {', '.join(QUERY_SOURCES)}"
            )
        if routing not in ROUTING_MODES:
            raise ValueError(
                f"Unbekanntes Routing {routing!r}; erlaubt: {', '.join(ROUTING_MODES)}"
            )
        if not 0.0 < decay <= 1.0:
            raise ValueError("decay muss in (0, 1] liegen")
        self.d_model = d_model
        self.slots = slots
        self.width = width
        self.key_dim = key_dim
        # Top-K darf die Slotzahl nie überschreiten; bei winzigen Bänken wird
        # sauber heruntergesetzt statt abzustürzen.
        self.read_k = max(0, min(read_k, slots))
        self.write_k = max(0, min(write_k, slots))
        self.replacement = replacement
        self.query_source = query_source
        self.routing = routing
        self.decay = decay
        # ``usage`` und ``learned`` brauchen die Zugriffszähler zwingend. Sonst
        # entscheidet ``track_usage``. Gemessen kosten die Zähler rund ein
        # Viertel des Durchsatzes (620 gegen 835 Token/s bei 64 Slots) und
        # beeinflussen die Rechnung nicht – sie sind reine Buchhaltung.
        # Standard ist deshalb aus; Analysen und der Trace-Modus schalten sie
        # ein.
        self.track_usage = track_usage
        self.needs_counters = track_usage or replacement in ("usage", "learned")
        self.precision = precision or PrecisionPolicy()

        # Die Quellenbreite kommt vom Aufrufer: Der semantische Zustand ist im
        # gebundenen Pfad schmaler als ``d_model``.
        source_width = source_width if source_width is not None else self._source_width(d_model)
        # Eine gemeinsame Projektion für alles, was aus derselben Eingabe
        # gelesen wird – dieselbe Lehre wie im State Core aus Milestone 2.5.
        # Reihenfolge: Query | Schreibschlüssel | Schreibwert | Lese-Gate | Schreib-Gate
        self.access_proj = nn.Linear(source_width, 2 * key_dim + width + 2)
        self.read_out = nn.Linear(width, d_model, bias=False)
        self.write_gate_bias = nn.Parameter(torch.full((1,), -2.0))
        self.read_gate_bias = nn.Parameter(torch.zeros(1))
        # Temperatur der Lesegewichtung; gelernt, damit das Modell zwischen
        # scharfer Auswahl und Mischung wählen kann.
        self.read_temperature = nn.Parameter(torch.zeros(1))
        if replacement == "learned":
            # Gelernter Ersetzungsscore aus den Slot-Statistiken.
            self.replacement_score = nn.Linear(5, 1)
        self.reset_parameters()

    def _source_width(self, d_model: int) -> int:
        if self.query_source in ("fast_context", "context_semantic"):
            return 2 * d_model
        return d_model

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.xavier_uniform_(self.access_proj.weight)
            nn.init.zeros_(self.access_proj.bias)
            nn.init.xavier_uniform_(self.read_out.weight)
            self.write_gate_bias.fill_(-2.0)
            self.read_gate_bias.zero_()
            self.read_temperature.zero_()
            if self.replacement == "learned":
                nn.init.zeros_(self.replacement_score.weight)
                nn.init.zeros_(self.replacement_score.bias)

    # ------------------------------------------------------------------
    # Zustand
    # ------------------------------------------------------------------

    def memory_dtypes(self, fallback: torch.dtype) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        return self.precision.memory_dtypes(fallback)

    def initial_state(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> MemoryState:
        value_dtype, key_dtype, score_dtype = self.memory_dtypes(dtype)
        zeros = lambda width, dt: torch.zeros(batch_size, self.slots, width, device=device, dtype=dt)
        counters = lambda fill: torch.full(
            (batch_size, self.slots), fill, device=device, dtype=score_dtype
        )
        return MemoryState(
            keys=zeros(self.key_dim, key_dtype),
            values=zeros(self.width, value_dtype),
            strength=counters(0.0),
            age=counters(0.0),
            usage_count=counters(0.0),
            read_count=counters(0.0),
            write_count=counters(0.0),
            last_read_step=counters(-1.0),
            last_write_step=counters(-1.0),
            occupied=counters(0.0),
            step=0,
        )

    # ------------------------------------------------------------------
    # Auswahl des zu überschreibenden Slots
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Sequenzlauf
    # ------------------------------------------------------------------

    def forward(
        self,
        source: Tensor,
        state: MemoryState | None = None,
        *,
        collect_metrics: bool = False,
        disable_read: bool = False,
        disable_write: bool = False,
        ablate_slots: frozenset[int] | None = None,
        interventions: dict[int, str] | None = None,
    ) -> tuple[Tensor, MemoryState, list[MemoryStepMetrics] | None]:
        """Liest und schreibt entlang der Sequenz.

        Gibt den Speicherbeitrag ``[batch, sequence, d_model]`` zurück; der
        Aufrufer addiert ihn residual auf seinen Strom.
        """
        if source.ndim != 3:
            raise ValueError("source muss die Form [batch, sequence, breite] besitzen")
        batch, length, _ = source.shape
        if state is None:
            state = self.initial_state(batch, device=source.device, dtype=source.dtype)
        if self.slots == 0 or (disable_read and disable_write):
            # Ein Speicher ohne Slots ist gültig und liefert einen neutralen
            # Beitrag, statt zu scheitern.
            zero = torch.zeros(batch, length, self.d_model, device=source.device, dtype=source.dtype)
            return zero, state, ([] if collect_metrics else None)

        compute_dtype = source.dtype
        # Die sequenzweite Projektion darf unter Autocast laufen – sie ist die
        # einzige größere Matrix hier. Der Tokenpfad danach nicht: Dort würde
        # Autocast je Zugriff eigene Cast-Kernel einfügen, genau wie im State
        # Core vor Milestone 2.5.
        device_type = source.device.type
        disable_autocast = torch.is_autocast_enabled(device_type)
        if disable_autocast:
            with torch.autocast(device_type=device_type, enabled=False):
                return self._scan(
                    source, state, compute_dtype,
                    collect_metrics=collect_metrics,
                    disable_read=disable_read,
                    disable_write=disable_write,
                    ablate_slots=ablate_slots,
                    interventions=interventions,
                )
        return self._scan(
            source, state, compute_dtype,
            collect_metrics=collect_metrics,
            disable_read=disable_read,
            disable_write=disable_write,
            ablate_slots=ablate_slots,
            interventions=interventions,
        )

    def _scan(
        self,
        source: Tensor,
        state: MemoryState,
        compute_dtype: torch.dtype,
        *,
        collect_metrics: bool,
        disable_read: bool,
        disable_write: bool,
        ablate_slots: frozenset[int] | None,
        interventions: dict[int, str] | None,
    ) -> tuple[Tensor, MemoryState, list[MemoryStepMetrics] | None]:
        batch, length, _ = source.shape
        access = linear_weight(self.access_proj, compute_dtype)
        access_bias = self.access_proj.bias
        if access_bias.dtype is not compute_dtype:
            access_bias = access_bias.to(compute_dtype)
        read_weight = linear_weight(self.read_out, compute_dtype)
        key_dtype, value_dtype = state.keys.dtype, state.values.dtype
        score_dtype = state.strength.dtype
        # Slot-Statistiken und Scores laufen in der Score-Präzision. Das ist
        # kein Luxus: Die Sperr- und Bonuskonstanten liegen außerhalb des
        # FP16-Wertebereichs, und Top-K entscheidet über kleine Differenzen.
        score_compute = score_dtype if score_dtype is not torch.float16 else torch.float32
        sizes = (self.key_dim, self.key_dim, self.width, 1, 1)
        projected = F.linear(source.to(compute_dtype), access, access_bias)
        if projected.dtype is not compute_dtype:
            projected = projected.to(compute_dtype)
        query_raw, write_key_raw, write_value_raw, read_gate_raw, write_gate_raw = projected.split(
            sizes, dim=-1
        )
        # Alles Eingabeabhängige einmal je Sequenz, nicht je Token.
        # Bereits in der Form, die der Tokenpfad braucht: [batch, 1, dim].
        # Das erspart je Token zwei ``unsqueeze``-Aufrufe.
        queries = F.normalize(torch.tanh(query_raw), dim=-1, eps=1e-6).unsqueeze(2).unbind(1)
        write_keys = F.normalize(torch.tanh(write_key_raw), dim=-1, eps=1e-6).unsqueeze(2).unbind(1)
        write_values = torch.tanh(write_value_raw).unsqueeze(2).unbind(1)
        read_bias, write_bias = self.read_gate_bias, self.write_gate_bias
        if read_bias.dtype is not compute_dtype:
            read_bias = read_bias.to(compute_dtype)
            write_bias = write_bias.to(compute_dtype)
        read_gates = torch.sigmoid(read_gate_raw + read_bias).unbind(1)
        write_gates = torch.sigmoid(write_gate_raw + write_bias).unbind(1)
        temperature = (F.softplus(self.read_temperature) + 0.5).to(score_compute)

        # Im Tokenpfad wird durchgehend im Rechendatentyp gearbeitet; die
        # Rückwandlung in die Speicherpräzision geschieht einmal am Ende.
        keys_c = state.keys.to(compute_dtype)
        values_c = state.values.to(compute_dtype)
        strength = state.strength.to(score_compute)
        age = state.age.to(score_compute)
        occupied = state.occupied.to(score_compute)
        last_read, last_write = state.last_read_step, state.last_write_step
        step = state.step

        read_mask, write_mask, frozen = self._intervention_masks(
            state, ablate_slots, interventions, score_compute
        )
        if frozen is not None and write_mask is not None:
            write_mask = write_mask + frozen

        outputs: list[Tensor] = []
        metrics: list[MemoryStepMetrics] | None = [] if collect_metrics else None
        read_k, write_k = self.read_k, self.write_k
        strength_routing = self.routing == "cosine_strength"
        zero_read = torch.zeros(batch, self.d_model, device=source.device, dtype=compute_dtype)
        # Zugriffszähler nur führen, wenn eine Policy oder die Telemetrie sie
        # liest. Ohne sie entfallen je Token zwei Streuoperationen.
        track_counters = self.needs_counters or collect_metrics
        track_timestamps = collect_metrics
        counters = torch.stack(
            (state.usage_count, state.read_count, state.write_count), dim=-1
        ).to(score_compute) if track_counters else None
        # Konstante Deltas einmal vor der Schleife: usage/reads bei jedem Lesen,
        # usage/writes bei jedem Schreiben.
        read_delta = (
            torch.tensor([1.0, 1.0, 0.0], device=source.device, dtype=score_compute)
            .expand(batch, max(read_k, 1), 3).contiguous()
            if track_counters else None
        )
        write_delta = (
            torch.tensor([1.0, 0.0, 1.0], device=source.device, dtype=score_compute)
            .expand(batch, max(write_k, 1), 3).contiguous()
            if track_counters else None
        )
        single_write = write_k == 1
        ones_k = torch.ones(batch, max(write_k, 1), device=source.device, dtype=score_compute)
        zeros_k = torch.zeros_like(ones_k)
        # Ein freier Slot muss jeden belegten schlagen, eine Ablation jeden
        # freien. Beide Konstanten bleiben im FP32-Bereich.
        free_bonus_scale = 1e6

        for index in range(length):
            # Belegungssperre und optionaler Stärkebonus in einem Term.
            # Unbeschriebene Slots werden hart ausgeschlossen.
            # Ein einziger Term trägt beide Rollen: Freie Slots bekommen beim
            # Schreiben einen großen Bonus, beim Lesen eine harte Sperre.
            free_bonus = (1.0 - occupied) * free_bonus_scale
            availability = free_bonus * -1e-2
            if strength_routing:
                # Ein stark beschriebener Slot gewinnt bei gleicher Ähnlichkeit.
                availability = availability + torch.log1p(strength)
            if read_mask is not None:
                availability = availability + read_mask

            # --- Lesen ------------------------------------------------
            if disable_read or read_k == 0:
                read_output = zero_read
                read_slots = torch.zeros(batch, 0, dtype=torch.long, device=source.device)
                read_scores = read_weights = read_slots.to(score_dtype)
                all_scores = torch.zeros(batch, self.slots, device=source.device, dtype=score_dtype)
            else:
                query = queries[index]  # [batch, 1, key_dim]
                # Ähnlichkeit ohne Token-zu-Token-Matrix: Query gegen Slots.
                # Broadcast-Multiplikation mit Reduktion statt bmm – dieselbe
                # Portabilitätslehre wie im State Core.
                # ``availability`` fasst Belegungssperre und Stärkebonus in einem
                # Term zusammen, der einmal je Token entsteht statt in drei
                # getrennten Additionen.
                all_scores = (keys_c * query).sum(-1).to(score_compute) + availability
                read_scores, read_slots = all_scores.topk(read_k, dim=-1)
                read_weights = torch.softmax(read_scores / temperature, dim=-1).to(compute_dtype)
                index_expanded = read_slots.unsqueeze(-1).expand(-1, -1, self.width)
                # Genau ``read_k`` Werte werden angefasst, nicht alle Slots.
                gathered = values_c.gather(1, index_expanded)
                pooled = (gathered * read_weights.unsqueeze(-1)).sum(1)
                read_output = F.linear(pooled, read_weight) * read_gates[index]
                if track_counters:
                    # Ein einziger Streuvorgang für alle additiven Zähler.
                    counters = counters.scatter_add(
                        1, read_slots.unsqueeze(-1).expand(-1, -1, 3), read_delta
                    )
                if track_timestamps:
                    last_read = last_read.scatter(
                        1, read_slots, torch.full_like(read_scores, float(step + index), dtype=score_dtype)
                    )
            outputs.append(read_output)

            # --- Schreiben --------------------------------------------
            if disable_write or write_k == 0:
                write_slots = torch.zeros(batch, 0, dtype=torch.long, device=source.device)
                write_scores = write_strength = write_slots.to(score_dtype)
                replaced = write_scores
                gate = torch.zeros(batch, 1, device=source.device, dtype=compute_dtype)
            else:
                gate = write_gates[index]
                replacement = self._replacement_score_with(
                    strength, age, counters, occupied, free_bonus
                )
                if write_mask is not None:
                    replacement = replacement + write_mask
                write_scores, write_slots = replacement.topk(write_k, dim=-1)
                # Nur für die Telemetrie: Ein Ersetzungsereignis liegt vor,
                # wenn ein bereits belegter Slot als Ziel gewählt wurde. Wie
                # stark er überschrieben wird, steht in ``write_strength``.
                if collect_metrics:
                    replaced = occupied.gather(1, write_slots)
                    write_strength = gate.expand(-1, write_k)
                else:
                    replaced = write_strength = None

                slot_column = write_slots.unsqueeze(-1)
                key_expanded = slot_column.expand(-1, -1, self.key_dim)
                value_expanded = slot_column.expand(-1, -1, self.width)
                gate_column = gate.unsqueeze(-1)
                # Bei einem einzelnen Schreibziel genügt Broadcasting; erst ab
                # write_k > 1 wird tatsächlich aufgeweitet.
                key_source = write_keys[index] if single_write else write_keys[index].expand(-1, write_k, -1)
                value_source = write_values[index] if single_write else write_values[index].expand(-1, write_k, -1)
                # Interpolation statt harter Ersetzung: ein schwaches Gate lässt
                # den bestehenden Eintrag weitgehend stehen.
                new_keys = torch.lerp(keys_c.gather(1, key_expanded), key_source, gate_column)
                new_values = torch.lerp(values_c.gather(1, value_expanded), value_source, gate_column)
                keys_c = keys_c.scatter(1, key_expanded, new_keys)
                values_c = values_c.scatter(1, value_expanded, new_values)
                gate_score = gate.to(score_compute).expand(-1, write_k)
                # Stärke hoch, Alter auf null – beide über dasselbe Gate.
                strength = strength.scatter(
                    1, write_slots, torch.lerp(strength.gather(1, write_slots), ones_k, gate_score)
                )
                # Belegung ist Buchhaltung, kein Lernsignal: ein einmal
                # angesteuerter Slot bleibt lesbar.
                occupied = occupied.scatter(1, write_slots, ones_k)
                age = (age + 1.0).scatter(
                    1, write_slots,
                    torch.lerp(age.gather(1, write_slots) + 1.0, zeros_k, gate_score),
                )
                if track_counters:
                    counters = counters.scatter_add(1, slot_column.expand(-1, -1, 3), write_delta)
                if track_timestamps:
                    last_write = last_write.scatter(
                        1, write_slots, torch.full_like(gate_score, float(step + index))
                    )

            if disable_write or write_k == 0:
                age = age + 1.0
            if self.decay < 1.0:
                strength = strength * self.decay

            if metrics is not None:
                metrics.append(
                    MemoryStepMetrics(
                        read_slots=read_slots,
                        read_scores=read_scores,
                        read_weights=read_weights,
                        read_output=read_output,
                        read_gate=read_gates[index] if not disable_read else zero_read[:, :1],
                        write_slots=write_slots,
                        write_scores=write_scores,
                        write_strength=write_strength,
                        write_gate=gate,
                        replaced=replaced,
                        slot_strength=strength,
                        slot_age=age,
                        slot_usage=counters[..., 0],
                        slot_read_count=counters[..., 1],
                        slot_write_count=counters[..., 2],
                        slot_occupied=occupied,
                        query=queries[index] if not disable_read else zero_read[:, : self.key_dim],
                        all_scores=all_scores,
                    )
                )

        if counters is not None:
            usage, reads, writes = (counters[..., i].to(score_dtype) for i in range(3))
        else:
            usage, reads, writes = state.usage_count, state.read_count, state.write_count
        next_state = MemoryState(
            keys=keys_c.to(key_dtype),
            values=values_c.to(value_dtype),
            strength=strength.to(score_dtype),
            age=age.to(score_dtype),
            usage_count=usage,
            read_count=reads,
            write_count=writes,
            last_read_step=last_read,
            last_write_step=last_write,
            occupied=occupied.to(score_dtype),
            step=step + length,
        )
        return torch.stack(outputs, dim=1), next_state, metrics

    def _replacement_score_with(
        self,
        strength: Tensor,
        age: Tensor,
        counters: Tensor | None,
        occupied: Tensor,
        free_bonus: Tensor,
    ) -> Tensor:
        """Höherer Wert bedeutet: eher überschreibbar.

        ``free_bonus`` ist bereits berechnet; freie Slots gewinnen dadurch
        immer, sodass ein leerer Speicher zuerst gefüllt wird.
        """
        if self.replacement == "age":
            return age + free_bonus
        if self.replacement == "strength":
            return free_bonus - strength
        if self.replacement == "usage":
            return free_bonus - counters[..., 0]
        if self.replacement == "lru_strength":
            # Alter belohnt Vergessen, Stärke schützt wichtige Einträge.
            return age - 8.0 * strength + free_bonus
        features = torch.stack(
            (
                age * 0.01,
                strength,
                counters[..., 0] * 0.1,
                counters[..., 1] * 0.1,
                counters[..., 2] * 0.1,
            ),
            dim=-1,
        )
        weight = linear_weight(self.replacement_score, features.dtype)
        bias = self.replacement_score.bias.to(features.dtype)
        return F.linear(features, weight, bias).squeeze(-1) + free_bonus

    def _intervention_masks(
        self,
        state: MemoryState,
        ablate_slots: frozenset[int] | None,
        interventions: dict[int, str] | None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Baut additive Masken für Slot-Ablation und Analyseoperationen.

        Sie entstehen einmal je Sequenz. Ohne Eingriff wird nichts erzeugt und
        der Tokenpfad zahlt einen einzigen ``is None``-Test.
        """
        if not ablate_slots and not interventions:
            return None, None, None
        device = state.strength.device
        dtype = dtype or state.strength.dtype
        read_mask = torch.zeros(1, self.slots, device=device, dtype=dtype)
        write_mask = torch.zeros(1, self.slots, device=device, dtype=dtype)
        frozen_mask = torch.zeros(1, self.slots, device=device, dtype=dtype)
        touched = False
        for slot in sorted(ablate_slots or ()):
            if not 0 <= slot < self.slots:
                raise ValueError(f"Slot {slot} liegt außerhalb von 0..{self.slots - 1}")
            # Deutlich unter dem Freibonus (1e6), sonst könnte ein freier,
            # abladierter Slot trotzdem gewinnen.
            read_mask[0, slot] = -1e4
            write_mask[0, slot] = -1e9
            touched = True
        for slot, operation in sorted((interventions or {}).items()):
            if not 0 <= slot < self.slots:
                raise ValueError(f"Slot {slot} liegt außerhalb von 0..{self.slots - 1}")
            if operation not in INTERVENTIONS:
                raise ValueError(
                    f"Unbekannte Intervention {operation!r}; erlaubt: {', '.join(INTERVENTIONS)}"
                )
            if operation in ("clear", "mute_read"):
                read_mask[0, slot] = -1e4
            if operation in ("freeze", "mute_write", "clear"):
                write_mask[0, slot] = -1e9
                frozen_mask[0, slot] = -1e9
            touched = True
        if not touched:
            return None, None, None
        return read_mask, write_mask, frozen_mask if frozen_mask.abs().sum() > 0 else None

    def extra_repr(self) -> str:
        return (
            f"slots={self.slots}, width={self.width}, key_dim={self.key_dim}, "
            f"read_k={self.read_k}, write_k={self.write_k}, "
            f"replacement={self.replacement}, query_source={self.query_source}, "
            f"routing={self.routing}, track_usage={self.track_usage}"
        )


def apply_slot_intervention(state: MemoryState, slot: int, operation: str) -> MemoryState:
    """Wendet eine Analyseoperation direkt auf den Laufzeitzustand an.

    Nur für Analyse und Replay gedacht. Der normale Inferenzpfad ruft das nie
    auf und wird dadurch nicht langsamer.
    """
    if operation not in INTERVENTIONS:
        raise ValueError(f"Unbekannte Intervention {operation!r}; erlaubt: {', '.join(INTERVENTIONS)}")
    if not 0 <= slot < state.slots:
        raise ValueError(f"Slot {slot} liegt außerhalb von 0..{state.slots - 1}")
    if operation != "clear":
        # freeze/mute wirken über die Masken im Forward, nicht auf den Inhalt.
        return state
    keys = state.keys.clone()
    values = state.values.clone()
    keys[:, slot] = 0.0
    values[:, slot] = 0.0
    strength = state.strength.clone()
    occupied = state.occupied.clone()
    strength[:, slot] = 0.0
    occupied[:, slot] = 0.0
    return replace(state, keys=keys, values=values, strength=strength, occupied=occupied)


def replace_slot_value(state: MemoryState, slot: int, value: Tensor) -> MemoryState:
    """Setzt den Wert eines Slots auf einen vorgegebenen Vektor."""
    if not 0 <= slot < state.slots:
        raise ValueError(f"Slot {slot} liegt außerhalb von 0..{state.slots - 1}")
    values = state.values.clone()
    values[:, slot] = value.to(values.dtype)
    occupied = state.occupied.clone()
    occupied[:, slot] = 1.0
    return replace(state, values=values, occupied=occupied)


def memory_utilisation(state: MemoryState) -> dict[str, Any]:
    """Welche Slots wurden tatsächlich benutzt? Reine Messung, keine Deutung."""
    occupied = state.occupied[0]
    reads, writes = state.read_count[0], state.write_count[0]
    used = int((occupied > 0).sum())
    # Ohne mitgeführte Zähler wären Lese- und Schreibzahlen schlicht null. Das
    # wird gemeldet, statt eine leere Statistik als Ergebnis auszugeben.
    tracked = bool(float(reads.sum()) > 0 or float(writes.sum()) > 0)
    return {
        "counters_tracked": tracked,
        "slots": int(state.slots),
        "occupied_slots": used,
        "occupancy_rate": used / max(int(state.slots), 1),
        "slots_ever_read": int((reads > 0).sum()),
        "total_reads": float(reads.sum()),
        "total_writes": float(writes.sum()),
        "mean_strength": float(state.strength[0].mean()),
        "max_strength": float(state.strength[0].max()) if state.slots else 0.0,
        "mean_age": float(state.age[0].mean()),
        "read_distribution": reads.tolist(),
        "write_distribution": writes.tolist(),
    }
