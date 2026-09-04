from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isqrt
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 128
    n_layers: int = 4
    local_kernel_size: int = 3
    dropout: float = 0.0
    telemetry_clusters: int = 8
    tie_embeddings: bool = True
    activity_activation_weight: float = 0.35
    activity_delta_weight: float = 0.35
    activity_gate_weight: float = 0.15
    activity_flow_weight: float = 0.15
    activity_update_threshold: float = 0.02
    state_interactions: bool = False
    # Der rekurrente Token-Pfad rechnet standardmäßig in der Präzision des
    # Zustands. Autocast würde dort pro Token und Projektion je einen eigenen
    # Cast-Kernel einfügen; das kostet bei kleinen Schrittgrößen messbar mehr,
    # als die halbpräzisen Matmuls einsparen. Die sequenzweite Eingangs-
    # projektion bleibt davon unberührt und nutzt Autocast weiterhin.
    recurrent_autocast: bool = False
    # --- Milestone 3: bounded sparse external memory -------------------
    # 0 Slots bedeutet: kein Memory. Das ist der Standard, damit jede ältere
    # Konfiguration unverändert dasselbe Modell beschreibt.
    memory_slots: int = 0
    memory_width: int = 64
    memory_key_dim: int = 32
    memory_read_k: int = 2
    memory_write_k: int = 1
    #: Nach welchem Block der Speicher eingehängt wird (-1 = letzter Block).
    memory_layer: int = -1
    memory_query_source: str = "output"
    memory_replacement: str = "lru_strength"
    memory_routing: str = "cosine_strength"
    #: Zugriffszähler mitführen. Sie kosten rund ein Viertel des
    #: Durchsatzes und beeinflussen die Rechnung nicht – für
    #: Nutzungsanalysen sind sie aber die Datengrundlage.
    memory_track_usage: bool = False
    memory_decay: float = 1.0

    def __post_init__(self) -> None:
        if self.vocab_size < 8:
            raise ValueError("vocab_size muss mindestens 8 sein")
        if self.d_model < 8:
            raise ValueError("d_model muss mindestens 8 sein")
        if self.n_layers < 1:
            raise ValueError("n_layers muss mindestens 1 sein")
        if self.local_kernel_size < 1:
            raise ValueError("local_kernel_size muss mindestens 1 sein")
        if self.telemetry_clusters < 1 or self.telemetry_clusters > self.d_model:
            raise ValueError("telemetry_clusters muss zwischen 1 und d_model liegen")
        if self.state_interactions and self.telemetry_clusters > self.binding_size:
            raise ValueError(
                "telemetry_clusters darf im gebundenen Pfad die Breite des semantischen "
                f"Zustands ({self.binding_size}) nicht überschreiten"
            )
        activity_weights = (
            self.activity_activation_weight,
            self.activity_delta_weight,
            self.activity_gate_weight,
            self.activity_flow_weight,
        )
        if any(weight < 0 for weight in activity_weights) or sum(activity_weights) <= 0:
            raise ValueError("Aktivitätsgewichte müssen nichtnegativ sein und eine positive Summe besitzen")
        if self.activity_update_threshold < 0:
            raise ValueError("activity_update_threshold darf nicht negativ sein")
        if self.memory_slots < 0:
            raise ValueError("memory_slots darf nicht negativ sein")
        if self.memory_width < 1 or self.memory_key_dim < 1:
            raise ValueError("memory_width und memory_key_dim müssen mindestens 1 sein")
        if self.memory_read_k < 0 or self.memory_write_k < 0:
            raise ValueError("memory_read_k und memory_write_k dürfen nicht negativ sein")
        if not -self.n_layers <= self.memory_layer < self.n_layers:
            raise ValueError(
                f"memory_layer muss zwischen {-self.n_layers} und {self.n_layers - 1} liegen"
            )
        if not 0.0 < self.memory_decay <= 1.0:
            raise ValueError("memory_decay muss in (0, 1] liegen")
        # Die erlaubten Namen stehen im Memory-Modul; der Import bleibt lokal,
        # damit die Konfiguration nicht von der Modellschicht abhängt.
        from glassmind.model.memory import QUERY_SOURCES, REPLACEMENT_POLICIES, ROUTING_MODES

        if self.memory_query_source not in QUERY_SOURCES:
            raise ValueError(
                f"memory_query_source={self.memory_query_source!r} ist unzulässig; "
                f"erlaubt: {', '.join(QUERY_SOURCES)}"
            )
        if self.memory_replacement not in REPLACEMENT_POLICIES:
            raise ValueError(
                f"memory_replacement={self.memory_replacement!r} ist unzulässig; "
                f"erlaubt: {', '.join(REPLACEMENT_POLICIES)}"
            )
        if self.memory_routing not in ROUTING_MODES:
            raise ValueError(
                f"memory_routing={self.memory_routing!r} ist unzulässig; "
                f"erlaubt: {', '.join(ROUTING_MODES)}"
            )

    @property
    def has_memory(self) -> bool:
        """Ohne Slots existiert kein Speicher – und kein einziger Aufruf."""
        return self.memory_slots > 0

    @property
    def memory_layer_index(self) -> int:
        return self.memory_layer % self.n_layers

    @property
    def memory_query_width(self) -> int:
        """Breite der Query-Eingabe – ``semantic`` ist im gebundenen Pfad schmaler."""
        source = self.memory_query_source
        if source == "semantic":
            return self.semantic_width
        if source == "context_semantic":
            return self.d_model + self.semantic_width
        if source == "fast_context":
            return 2 * self.d_model
        return self.d_model

    @property
    def binding_rank(self) -> int:
        """Rang der gebundenen State-Interaktion; 0 ohne aktivierte Interaktion."""
        return max(2, isqrt(self.d_model)) if self.state_interactions else 0

    @property
    def binding_size(self) -> int:
        return self.binding_rank * self.binding_rank

    @property
    def semantic_width(self) -> int:
        """Breite des semantischen Zustands im jeweils aktiven Pfad."""
        return self.binding_size if self.state_interactions else self.d_model

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        return cls(**values)

    @classmethod
    def tiny(cls, vocab_size: int = 260) -> "ModelConfig":
        return cls(vocab_size=vocab_size, d_model=32, n_layers=1, telemetry_clusters=4)
