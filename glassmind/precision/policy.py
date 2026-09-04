"""Zentrale, reproduzierbare Precision- und Quantisierungskonfiguration.

Die Policy ist bewusst *keine* Architektureigenschaft: Sie beschreibt, in
welcher Zahlendarstellung ein bereits definiertes Modell gerechnet, gespeichert
und ausgeliefert wird. Deshalb lebt sie neben ``ModelConfig`` und nicht darin.

Alle Quantisierungs- und Präzisionsentscheidungen laufen über dieses Objekt.
Im Modellcode selbst steht keine verstreute Sonderlogik.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping

import torch

#: Von GlassMind unterstützte Gleitkomma-Rechenformate.
FLOAT_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

#: ``inherit`` bedeutet: den dtype übernehmen, den das Modell bzw. die Eingabe
#: ohnehin hat. Damit ist die Milestone-2.5-Baseline der Standardfall und bleibt
#: numerisch unverändert.
INHERIT = "inherit"

#: Gewichtsschemata. ``none`` lässt die Gewichte unangetastet.
WEIGHT_SCHEMES = (
    "none",
    "int8",
    "int4",
    "float16",
    "bfloat16",
    "float8_e4m3",
    "float8_e5m2",
)

#: Quantisierbare Modulgruppen. Die Zuordnung zu konkreten Modulen steht in
#: :func:`glassmind.precision.quantization.component_modules`.
#:
#: Hinweis zur Ehrlichkeit: Die Block-Gates besitzen seit Milestone 2.5 keine
#: eigene Matrix mehr – Wert- und Gate-Anteile teilen sich ``input_proj``.
#: ``gate`` bezeichnet deshalb nur noch das Gate des lokalen Mixers.
WEIGHT_COMPONENTS = (
    "embedding",
    "local_mixer",
    "input_projection",
    "state_projection",
    "gate",
    "output_projection",
    "lm_head",
)


def resolve_dtype(name: str, fallback: torch.dtype) -> torch.dtype:
    """Löst einen Policy-Eintrag gegen den geerbten dtype auf."""
    if name == INHERIT:
        return fallback
    try:
        return FLOAT_DTYPES[name]
    except KeyError as exc:
        allowed = ", ".join((INHERIT, *FLOAT_DTYPES))
        raise ValueError(f"Unbekannter Rechendatentyp {name!r}; erlaubt: {allowed}") from exc


@dataclass(frozen=True)
class PrecisionPolicy:
    """Beschreibt Rechen-, Zustands- und Gewichtsdarstellung vollständig.

    Alle Felder mit dem Wert ``inherit`` übernehmen den dtype des Modells. Eine
    Policy, in der ausschließlich ``inherit`` steht und keine Gewichts-
    quantisierung aktiv ist, verhält sich bitgleich zur Milestone-2.5-Baseline.
    """

    profile: str = "inherit"
    #: dtype des rekurrenten Tokenpfads.
    compute: str = INHERIT
    #: dtype der sequenzweiten Vorprojektion nach dem Autocast-Ergebnis.
    activations: str = INHERIT
    fast_state: str = INHERIT
    context_state: str = INHERIT
    semantic_state: str = INHERIT
    #: Eigene Achse für das externe Memory (Milestone 3). Werte und Schlüssel
    #: dürfen reduziert werden; die Scores bleiben davon getrennt, weil Top-K
    #: und Ersetzungsentscheidungen auf kleinen Differenzen beruhen.
    memory_value: str = INHERIT
    memory_key: str = INHERIT
    memory_score: str = "float32"
    #: Modulgruppe → Gewichtsschema.
    weights: Mapping[str, str] = field(default_factory=dict)
    #: 0 = eine Skala je Ausgangskanal, >0 = gruppenweise entlang der Eingänge.
    weight_group_size: int = 0
    #: Hält dequantisierte Gewichte im Speicher vor. Ohne Cache wird bei jedem
    #: Aufruf neu dequantisiert – im Streaming also einmal pro Token.
    dequantization_cache: bool = True
    #: Herkunftsvermerk, wenn ``auto`` die Policy gewählt hat.
    selection_notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "compute", "activations", "fast_state", "context_state", "semantic_state",
            "memory_value", "memory_key", "memory_score",
        ):
            value = getattr(self, name)
            if value != INHERIT and value not in FLOAT_DTYPES:
                allowed = ", ".join((INHERIT, *FLOAT_DTYPES))
                raise ValueError(f"{name}={value!r} ist unzulässig; erlaubt: {allowed}")
        unknown_components = set(self.weights) - set(WEIGHT_COMPONENTS)
        if unknown_components:
            raise ValueError(
                "Unbekannte Modulgruppe(n): "
                f"{', '.join(sorted(unknown_components))}; erlaubt: {', '.join(WEIGHT_COMPONENTS)}"
            )
        unknown_schemes = set(self.weights.values()) - set(WEIGHT_SCHEMES)
        if unknown_schemes:
            raise ValueError(
                "Unbekannte(s) Gewichtsschema(ta): "
                f"{', '.join(sorted(unknown_schemes))}; erlaubt: {', '.join(WEIGHT_SCHEMES)}"
            )
        if self.weight_group_size < 0:
            raise ValueError("weight_group_size darf nicht negativ sein")
        # Ein Mapping in einer frozen dataclass soll nicht nachträglich mutierbar sein.
        object.__setattr__(self, "weights", dict(self.weights))
        # Abgeleitete Flags. Der Tokenpfad fragt sie bei jedem Schritt ab; als
        # Bool-Attribut kostet das einen Pointer-Vergleich statt eines
        # String-Vergleichs. Sie sind bewusst keine dataclass-Felder, damit
        # ``asdict`` und ``from_dict`` unverändert bleiben.
        object.__setattr__(self, "inherits_compute", self.compute == INHERIT)
        object.__setattr__(self, "inherits_activations", self.activations == INHERIT)

    # ------------------------------------------------------------------

    @property
    def quantizes_weights(self) -> bool:
        return any(scheme != "none" for scheme in self.weights.values())

    @property
    def is_neutral(self) -> bool:
        """Wahr, wenn die Policy nichts am Milestone-2.5-Verhalten ändert."""
        return not self.quantizes_weights and all(
            getattr(self, name) == INHERIT
            for name in ("compute", "activations", "fast_state", "context_state", "semantic_state")
        )

    @property
    def memory_is_neutral(self) -> bool:
        """Wahr, wenn das Memory dem Modell-dtype folgt."""
        return self.memory_value == INHERIT and self.memory_key == INHERIT

    def scheme_for(self, component: str) -> str:
        return self.weights.get(component, "none")

    def state_dtypes(self, fallback: torch.dtype) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        return (
            resolve_dtype(self.fast_state, fallback),
            resolve_dtype(self.context_state, fallback),
            resolve_dtype(self.semantic_state, fallback),
        )

    def memory_dtypes(self, fallback: torch.dtype) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        """(Werte, Schlüssel, Scores) des externen Memory."""
        return (
            resolve_dtype(self.memory_value, fallback),
            resolve_dtype(self.memory_key, fallback),
            resolve_dtype(self.memory_score, fallback),
        )

    def with_memory(
        self, value: str | None = None, key: str | None = None, score: str | None = None
    ) -> "PrecisionPolicy":
        return replace(
            self,
            memory_value=value or self.memory_value,
            memory_key=key or self.memory_key,
            memory_score=score or self.memory_score,
        )

    def with_weights(self, **components: str) -> "PrecisionPolicy":
        merged = dict(self.weights)
        merged.update(components)
        return replace(self, weights=merged)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["weights"] = dict(self.weights)
        data["selection_notes"] = list(self.selection_notes)
        return data

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PrecisionPolicy":
        data = dict(values)
        data["selection_notes"] = tuple(data.get("selection_notes", ()))
        return cls(**data)

    def describe(self) -> str:
        weights = ", ".join(f"{k}={v}" for k, v in sorted(self.weights.items()) if v != "none") or "keine"
        return (
            f"Profil={self.profile}  compute={self.compute}  activations={self.activations}  "
            f"fast={self.fast_state}  context={self.context_state}  semantic={self.semantic_state}  "
            f"memory={self.memory_value}/{self.memory_key}/{self.memory_score}  "
            f"Gewichte: {weights}"
        )


# ----------------------------------------------------------------------
# Profile
# ----------------------------------------------------------------------

def _all_components(scheme: str) -> dict[str, str]:
    return {component: scheme for component in WEIGHT_COMPONENTS}


def safe_profile() -> PrecisionPolicy:
    """Vollständig FP32: Rechenpfad, Aktivierungen und alle drei Zustände."""
    return PrecisionPolicy(
        profile="safe",
        compute="float32",
        activations="float32",
        fast_state="float32",
        context_state="float32",
        semantic_state="float32",
        memory_value="float32",
        memory_key="float32",
        memory_score="float32",
    )


def balanced_profile(compute: str = "bfloat16") -> PrecisionPolicy:
    """Reduzierte Aktivierungen, konservative Zustände.

    ``fast_state`` wird pro Token ohnehin fast vollständig überschrieben und
    verträgt deshalb reduzierte Präzision. ``context_state`` und
    ``semantic_state`` akkumulieren über viele Schritte und bleiben FP32.
    """
    return PrecisionPolicy(
        profile="balanced",
        compute=compute,
        activations=compute,
        fast_state=compute,
        context_state="float32",
        semantic_state="float32",
        # Konservativer Start laut Milestone-3-Vorgabe: Werte und Schlüssel
        # reduziert, Scores in voller Präzision.
        memory_value=compute,
        memory_key=compute,
        memory_score="float32",
    )


def fast_profile(compute: str = "bfloat16") -> PrecisionPolicy:
    """Auch ``context_state`` reduziert; ``semantic_state`` bleibt FP32.

    Ob dieses Profil tatsächlich schneller ist, entscheidet die Messung –
    ``scripts/precision_matrix.py`` weist es aus.
    """
    return PrecisionPolicy(
        profile="fast",
        compute=compute,
        activations=compute,
        fast_state=compute,
        context_state=compute,
        semantic_state="float32",
        memory_value=compute,
        memory_key=compute,
        memory_score="float32",
    )


def experimental_profile(
    compute: str = "bfloat16", weight_scheme: str = "int8", group_size: int = 0
) -> PrecisionPolicy:
    """Reduzierte Rechenpräzision plus quantisierte Gewichte."""
    return PrecisionPolicy(
        profile="experimental",
        compute=compute,
        activations=compute,
        fast_state=compute,
        context_state=compute,
        semantic_state="float32",
        memory_value=compute,
        memory_key=compute,
        memory_score="float32",
        weights=_all_components(weight_scheme),
        weight_group_size=group_size,
    )


#: Profile ohne Hardwareabfrage. ``auto`` liegt in ``selection.py``, weil es
#: Microbenchmarks braucht.
STATIC_PROFILES = {
    "safe": safe_profile,
    "balanced": balanced_profile,
    "fast": fast_profile,
    "experimental": experimental_profile,
}
PROFILE_NAMES = ("safe", "balanced", "fast", "experimental", "auto")
