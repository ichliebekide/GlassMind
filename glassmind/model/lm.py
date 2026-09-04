from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Collection
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from glassmind.model.config import ModelConfig
from glassmind.model.local_mixer import CausalLocalMixer
from glassmind.model.memory import MemoryState, MemoryStepMetrics, SparseMemory
from glassmind.model.state_core import BlockState, SelectiveStateBlock, StepMetrics
from glassmind.observe.bus import OFF_BUS, ObservationBus, ObservationMode
from glassmind.observe.events import ObservationEvent
from glassmind.precision.policy import PrecisionPolicy
from glassmind.observe.metrics import (
    CLUSTER_STATISTIC_COUNT,
    build_cluster_nodes,
    cluster_flow_rms,
    cluster_statistics,
    sequence_tensor_summary,
    tensor_summary,
)


@dataclass
class ModelState:
    local: Tensor
    blocks: tuple[BlockState, ...]
    position: int = 0
    #: Laufzeitinhalt des externen Memory; ``None``, wenn keines konfiguriert ist.
    memory: MemoryState | None = None

    def detach(self) -> "ModelState":
        return ModelState(
            self.local.detach(),
            tuple(block.detach() for block in self.blocks),
            self.position,
            None if self.memory is None else self.memory.detach(),
        )


class GlassMindLM(nn.Module):
    """Kausales Sprachmodell mit O(n)-Rekurrenz und begrenztem Zustand."""

    def __init__(self, config: ModelConfig, precision: PrecisionPolicy | None = None) -> None:
        super().__init__()
        self.config = config
        # Die Policy beschreibt die Zahlendarstellung, nicht die Architektur.
        # Der Standardwert ist neutral und lässt das Modell unverändert.
        self.precision = precision or PrecisionPolicy()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.local_mixer = CausalLocalMixer(config.d_model, config.local_kernel_size)
        self.blocks = nn.ModuleList(
            [
                SelectiveStateBlock(
                    config.d_model,
                    config.dropout,
                    state_interactions=config.state_interactions,
                    recurrent_autocast=config.recurrent_autocast,
                    precision=self.precision,
                )
                for _ in range(config.n_layers)
            ]
        )
        # Das externe Memory entsteht nur, wenn Slots konfiguriert sind. Ohne
        # Slots existiert das Modul nicht und wird nirgends aufgerufen – der
        # Milestone-2.6-Pfad bleibt damit unangetastet.
        self.memory = (
            SparseMemory(
                config.d_model,
                slots=config.memory_slots,
                width=config.memory_width,
                key_dim=config.memory_key_dim,
                read_k=config.memory_read_k,
                write_k=config.memory_write_k,
                replacement=config.memory_replacement,
                query_source=config.memory_query_source,
                routing=config.memory_routing,
                source_width=config.memory_query_width,
                track_usage=config.memory_track_usage,
                decay=config.memory_decay,
                precision=self.precision,
            )
            if config.has_memory
            else None
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        self.apply(self._initialize)
        for block in self.blocks:
            block.reset_interaction_parameters()
        if self.memory is not None:
            self.memory.reset_parameters()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def initial_state(self, batch_size: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> ModelState:
        # Die abschließende Normalisierung trägt immer einen echten
        # Gleitkommaparameter – auch wenn Embedding und LM-Head quantisiert
        # sind. Ihn abzufragen ist konstant teuer, anders als ein Durchlauf
        # durch alle Parameter.
        reference = self.final_norm.weight
        device = device or reference.device
        dtype = dtype or reference.dtype
        local = self.local_mixer.initial_state(batch_size, device=device, dtype=dtype)
        blocks = tuple(block.initial_state(batch_size, device=device, dtype=dtype) for block in self.blocks)
        memory = (
            None
            if self.memory is None
            else self.memory.initial_state(batch_size, device=device, dtype=dtype)
        )
        return ModelState(local, blocks, 0, memory)

    def forward(
        self,
        input_ids: Tensor,
        state: ModelState | None = None,
        *,
        observer: ObservationBus | None = None,
        ablate_states: Collection[str] | None = None,
        disable_memory: bool = False,
        disable_memory_read: bool = False,
        disable_memory_write: bool = False,
        ablate_memory_slots: Collection[int] | None = None,
        memory_interventions: dict[int, str] | None = None,
    ) -> tuple[Tensor, ModelState]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids muss die Form [batch, sequence] besitzen")
        if input_ids.shape[1] == 0:
            raise ValueError("Eine leere Sequenz ist nicht zulässig")
        observer = observer or OFF_BUS
        ablations = self._validate_ablations(ablate_states)
        x = self.embedding(input_ids)
        if state is None:
            state = self.initial_state(input_ids.shape[0], device=x.device, dtype=x.dtype)
        if len(state.blocks) != len(self.blocks):
            raise ValueError("Der übergebene Zustand passt nicht zur Zahl der Modellblöcke")

        # Ohne konfiguriertes Memory ist ``memory_layer`` -1 und die Prüfung im
        # Blockdurchlauf schlägt nie an: es entsteht kein einziger Zusatzaufruf.
        memory_active = self.memory is not None and not disable_memory
        memory_layer = self.config.memory_layer_index if memory_active else -1
        needs_state_trace = memory_active and self.memory.query_source != "output"
        memory_state = state.memory

        x, local_state = self.local_mixer(x, state.local)
        block_states: list[BlockState] = []
        collect_metrics = observer.traces_tokens
        for layer, (block, block_state) in enumerate(zip(self.blocks, state.blocks, strict=True)):
            trace: dict[str, Tensor] | None = {} if (needs_state_trace and layer == memory_layer) else None
            x, next_block_state, metrics = block(
                x,
                block_state,
                collect_metrics=collect_metrics,
                ablate_states=ablations,
                state_trace=trace,
            )
            block_states.append(next_block_state)
            if metrics is not None:
                self._emit_trace(observer, input_ids, layer, state.position, metrics)
            elif observer.mode >= ObservationMode.SUMMARY:
                self._emit_summary(observer, layer, state.position, input_ids.shape[1], next_block_state)
            if layer == memory_layer:
                x, memory_state = self._run_memory(
                    x,
                    trace,
                    memory_state,
                    observer,
                    input_ids,
                    state.position,
                    collect_metrics,
                    disable_memory_read,
                    disable_memory_write,
                    ablate_memory_slots,
                    memory_interventions,
                )

        logits = self.lm_head(self.final_norm(x))
        if observer.traces_tokens:
            self._emit_predictions(observer, input_ids, logits, state.position)
        next_state = ModelState(
            local_state,
            tuple(block_states),
            state.position + input_ids.shape[1],
            memory_state,
        )
        return logits, next_state

    def _memory_source(self, x: Tensor, trace: dict[str, Tensor] | None) -> Tensor:
        """Baut die Query-Eingabe aus der konfigurierten Zustandsquelle.

        Welche Quelle die beste Query erzeugt, entscheidet die Messung in
        ``scripts/memory_study.py`` – nicht eine Annahme.
        """
        source = self.memory.query_source
        if source == "output" or trace is None:
            return x
        if source in ("fast", "context", "semantic"):
            value = trace[source]
        elif source == "fast_context":
            value = torch.cat((trace["fast"], trace["context"]), dim=-1)
        else:  # context_semantic
            value = torch.cat((trace["context"], trace["semantic"]), dim=-1)
        return value.to(x.dtype)

    def _run_memory(
        self,
        x: Tensor,
        trace: dict[str, Tensor] | None,
        memory_state: MemoryState | None,
        observer: ObservationBus,
        input_ids: Tensor,
        start: int,
        collect_metrics: bool,
        disable_read: bool,
        disable_write: bool,
        ablate_slots: Collection[int] | None,
        interventions: dict[int, str] | None,
    ) -> tuple[Tensor, MemoryState | None]:
        source = self._memory_source(x, trace)
        contribution, memory_state, metrics = self.memory(
            source,
            memory_state,
            collect_metrics=collect_metrics,
            disable_read=disable_read,
            disable_write=disable_write,
            ablate_slots=frozenset(ablate_slots) if ablate_slots else None,
            interventions=interventions,
        )
        # Residual: Der Speicher ergänzt den Strom, er ersetzt ihn nicht. Der
        # rekurrente State Core bleibt der durchgehende Träger.
        x = x + contribution.to(x.dtype)
        if metrics:
            self._emit_memory_trace(observer, input_ids, start, metrics, memory_state)
        elif observer.mode >= ObservationMode.SUMMARY and memory_state is not None:
            self._emit_memory_summary(observer, start, input_ids.shape[1], memory_state)
        return x, memory_state

    def step(
        self,
        token_ids: Tensor,
        state: ModelState | None = None,
        *,
        observer: ObservationBus | None = None,
        ablate_states: Collection[str] | None = None,
        **memory_options: Any,
    ) -> tuple[Tensor, ModelState]:
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if token_ids.shape[1] != 1:
            raise ValueError("step erwartet genau ein Token pro Batch-Eintrag")
        logits, state = self.forward(
            token_ids,
            state,
            observer=observer,
            ablate_states=ablate_states,
            **memory_options,
        )
        return logits[:, -1], state

    @staticmethod
    def _validate_ablations(states: Collection[str] | None) -> frozenset[str] | None:
        if not states:
            return None
        normalized = frozenset(str(state).lower() for state in states)
        unknown = normalized - {"fast", "context", "semantic"}
        if unknown:
            raise ValueError(f"Unbekannte State-Ablation: {', '.join(sorted(unknown))}")
        return normalized

    def _emit_summary(self, observer: ObservationBus, layer: int, start: int, length: int, state: BlockState) -> None:
        observer.emit(
            ObservationEvent(
                event="state_summary",
                step=start + length - 1,
                token_index=start + length - 1,
                layer_id=f"core.{layer}",
                payload={
                    "sequence_length": length,
                    "fast": tensor_summary(state.fast),
                    "context": tensor_summary(state.context),
                    "semantic": tensor_summary(state.semantic),
                },
            )
        )

    # Reihenfolge der Kantengruppen; hält Berechnung und Auswertung synchron.
    _EDGE_TRANSITIONS = (
        "input_fast",
        "fast_context",
        "context_semantic",
        "fast_output",
        "context_output",
        "semantic_output",
        "query_read_output",
    )

    @staticmethod
    def _state_data(metric: StepMetrics) -> tuple[tuple[str, Tensor, Tensor, Tensor, Tensor], ...]:
        return (
            ("fast", metric.fast, metric.fast_delta, metric.fast_gate, metric.input_flow),
            ("context", metric.context, metric.context_delta, metric.context_gate, metric.fast_context_flow),
            ("semantic", metric.semantic, metric.semantic_delta, metric.semantic_gate, metric.context_semantic_flow),
        )

    def _packed_token_metrics(self, metric: StepMetrics, cluster_count: int) -> Tensor:
        """Packt alle Kennzahlen eines Tokens in einen einzigen Gerätetensor.

        Der Trace-Modus brauchte zuvor pro Token und Block über hundert
        einzelne ``.item()``-Aufrufe. Jeder davon ist ein eigener CPU/GPU-
        Synchronisationspunkt. Hier bleibt alles bis zu einem gemeinsamen
        Transfer je Block auf dem Gerät.
        """
        pieces = [
            cluster_statistics(value, delta, gate, flow, cluster_count).reshape(-1)
            for _, value, delta, gate, flow in self._state_data(metric)
        ]
        for flow in (
            metric.input_flow,
            metric.fast_context_flow,
            metric.context_semantic_flow,
            metric.fast_output_flow,
            metric.context_output_flow,
            metric.semantic_output_flow,
            metric.query_read_flow,
        ):
            pieces.append(cluster_flow_rms(flow, cluster_count))
        health = torch.stack(
            [
                torch.stack(
                    (
                        torch.isnan(value).any().float(),
                        torch.isinf(value).any().float(),
                    )
                )
                for _, value, *_ in self._state_data(metric)
            ]
        ).amax(dim=0)
        pieces.append(
            torch.stack(
                (
                    metric.input_flow.detach().float().square().mean().sqrt(),
                    metric.output_flow.detach().float().square().mean().sqrt(),
                )
            )
        )
        pieces.append(health)
        return torch.cat(pieces)

    def _emit_trace(
        self,
        observer: ObservationBus,
        input_ids: Tensor,
        layer: int,
        start: int,
        metrics: list[StepMetrics],
    ) -> None:
        cluster_count = self.config.telemetry_clusters
        activity_weights = (
            self.config.activity_activation_weight,
            self.config.activity_delta_weight,
            self.config.activity_gate_weight,
            self.config.activity_flow_weight,
        )
        # Ein Transfer für die gesamte Sequenz dieses Blocks.
        packed = torch.stack(
            [self._packed_token_metrics(metric, cluster_count) for metric in metrics]
        ).cpu().tolist()
        token_ids = input_ids[0, : len(metrics)].tolist()
        full_states = None
        if observer.mode >= ObservationMode.FULL:
            full_states = [
                {
                    name: value[0].detach().float().cpu().tolist()
                    for name, value, *_ in self._state_data(metric)
                }
                for metric in metrics
            ]

        input_id = f"core.{layer}.input"
        output_id = f"core.{layer}.output"
        state_block = cluster_count * CLUSTER_STATISTIC_COUNT
        edge_offset = 3 * state_block
        extra_offset = edge_offset + len(self._EDGE_TRANSITIONS) * cluster_count
        for offset, values in enumerate(packed):
            token_index = start + offset
            nodes: list[dict[str, Any]] = []
            for index, name in enumerate(("fast", "context", "semantic")):
                rows = values[index * state_block : (index + 1) * state_block]
                nodes.extend(
                    build_cluster_nodes(
                        layer=layer,
                        state_name=name,
                        rows=[
                            rows[cluster * CLUSTER_STATISTIC_COUNT : (cluster + 1) * CLUSTER_STATISTIC_COUNT]
                            for cluster in range(cluster_count)
                        ],
                        activity_weights=activity_weights,
                    )
                )
            nodes.extend(
                [
                    {"id": input_id, "kind": "input", "activity": values[extra_offset], "components": {}},
                    {"id": output_id, "kind": "output", "activity": values[extra_offset + 1], "components": {}},
                ]
            )
            token_id = int(token_ids[offset])
            observer.annotate_cluster_activity(
                nodes,
                token_id=token_id,
                update_threshold=self.config.activity_update_threshold,
            )
            edges = self._metric_edges(
                layer,
                nodes,
                values[edge_offset:extra_offset],
                input_id,
                output_id,
            )
            payload: dict[str, Any] = {
                "nodes": nodes,
                "edges": edges,
                "health": {
                    "has_nan": bool(values[extra_offset + 2]),
                    "has_inf": bool(values[extra_offset + 3]),
                },
                "top_active_clusters": [
                    node["id"]
                    for node in sorted(
                        (item for item in nodes if ".cluster." in str(item["id"])),
                        key=lambda item: float(item.get("activity", 0.0)),
                        reverse=True,
                    )[:5]
                ],
                "activity_weights": {
                    "activation": self.config.activity_activation_weight,
                    "delta": self.config.activity_delta_weight,
                    "gate": self.config.activity_gate_weight,
                    "flow": self.config.activity_flow_weight,
                },
            }
            if full_states is not None:
                payload["full"] = full_states[offset]
            observer.emit(
                ObservationEvent(
                    event="network_step",
                    step=token_index,
                    token_index=token_index,
                    token_id=token_id,
                    layer_id=f"core.{layer}",
                    payload=payload,
                )
            )

    def _metric_edges(
        self,
        layer: int,
        nodes: list[dict[str, Any]],
        flows: list[float],
        input_id: str,
        output_id: str,
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[str]] = {}
        for node in nodes:
            grouped.setdefault(str(node["kind"]), []).append(str(node["id"]))
        cluster_count = self.config.telemetry_clusters
        endpoints = (
            ([input_id] * cluster_count, grouped["fast"]),
            (grouped["fast"], grouped["context"]),
            (grouped["context"], grouped["semantic"]),
            (grouped["fast"], [output_id] * cluster_count),
            (grouped["context"], [output_id] * cluster_count),
            (grouped["semantic"], [output_id] * cluster_count),
            (grouped["semantic"], [output_id] * cluster_count),
        )
        edges: list[dict[str, Any]] = []
        for group, ((sources, targets), name) in enumerate(
            zip(endpoints, self._EDGE_TRANSITIONS, strict=True)
        ):
            base = group * cluster_count
            for index, (source, target) in enumerate(zip(sources, targets, strict=True)):
                edges.append(
                    {
                        "id": f"core.{layer}.{name}.{index}",
                        "source": source,
                        "target": target,
                        "flow": flows[base + index],
                    }
                )
        return edges

    # ------------------------------------------------------------------
    # Memory-Telemetrie
    # ------------------------------------------------------------------

    def _emit_memory_summary(
        self, observer: ObservationBus, start: int, length: int, state: MemoryState
    ) -> None:
        occupied = state.occupied
        observer.emit(
            ObservationEvent(
                event="memory_summary",
                step=start + length - 1,
                token_index=start + length - 1,
                layer_id="memory",
                payload={
                    "sequence_length": length,
                    "slots": int(state.slots),
                    "occupied_slots": int((occupied[0] > 0).sum().item()),
                    "strength": tensor_summary(state.strength),
                    "age": tensor_summary(state.age),
                    "read_count": tensor_summary(state.read_count),
                    "write_count": tensor_summary(state.write_count),
                },
            )
        )

    def _emit_memory_trace(
        self,
        observer: ObservationBus,
        input_ids: Tensor,
        start: int,
        metrics: list[MemoryStepMetrics],
        final_state: MemoryState | None,
    ) -> None:
        """Ein Ereignis je Token mit echten Zugriffs- und Slotwerten.

        Wie beim State Core wandern alle Zahlen in einem einzigen Transfer zum
        Host; einzelne ``.item()``-Aufrufe wären hier je Token ein eigener
        Synchronisationspunkt.
        """
        slots = self.memory.slots
        # Die tatsächlichen Breiten stammen aus den Messwerten, nicht aus der
        # Konfiguration: Bei abgeschaltetem Lesen oder Schreiben sind die
        # entsprechenden Felder leer.
        read_k = int(metrics[0].read_slots.shape[1])
        write_k = int(metrics[0].write_slots.shape[1])
        # Alles je Token in einen Vektor packen und gemeinsam übertragen.
        packed = torch.stack(
            [
                torch.cat(
                    (
                        # Die Telemetrie beschreibt den ersten Batch-Eintrag,
                        # genau wie die Zustands-Telemetrie des State Core.
                        metric.read_slots[0].reshape(-1).float(),
                        metric.read_scores[0].reshape(-1).float(),
                        metric.read_weights[0].reshape(-1).float(),
                        metric.write_slots[0].reshape(-1).float(),
                        metric.write_scores[0].reshape(-1).float(),
                        metric.write_strength[0].reshape(-1).float(),
                        metric.replaced[0].reshape(-1).float(),
                        metric.read_gate[0].reshape(-1).float(),
                        metric.write_gate[0].reshape(-1).float(),
                        metric.read_output[0].detach().float().square().mean().sqrt().reshape(1),
                        metric.slot_strength[0].float(),
                        metric.slot_age[0].float(),
                        metric.slot_usage[0].float(),
                        metric.slot_read_count[0].float(),
                        metric.slot_write_count[0].float(),
                        metric.slot_occupied[0].float(),
                    )
                )
                for metric in metrics
            ]
        ).cpu().tolist()
        token_ids = input_ids[0, : len(metrics)].tolist()
        offsets: list[tuple[str, int]] = [
            ("read_slots", read_k), ("read_scores", read_k), ("read_weights", read_k),
            ("write_slots", write_k), ("write_scores", write_k), ("write_strength", write_k),
            ("replaced", write_k), ("read_gate", 1), ("write_gate", 1), ("read_output_rms", 1),
            ("slot_strength", slots), ("slot_age", slots), ("slot_usage", slots),
            ("slot_read_count", slots), ("slot_write_count", slots), ("slot_occupied", slots),
        ]
        # Zähler sind ganzzahlig, Messwerte brauchen keine 17 Stellen. Das
        # halbiert die Tracegröße, ohne dass eine Zahl verloren geht: Die
        # Darstellung liest ohnehin nur vier Nachkommastellen.
        # Alter, Belegung und Zugriffszahlen sind ganzzahlig; nur Stärken,
        # Scores und Gewichte brauchen Nachkommastellen.
        integer_fields = {"slot_usage", "slot_read_count", "slot_write_count",
                          "slot_age", "slot_occupied", "read_slots", "write_slots"}
        for offset, values in enumerate(packed):
            fields: dict[str, list[float]] = {}
            cursor = 0
            for name, size in offsets:
                chunk = values[cursor : cursor + size]
                fields[name] = (
                    [int(value) for value in chunk]
                    if name in integer_fields
                    else [round(float(value), 4) for value in chunk]
                )
                cursor += size
            token_index = start + offset
            read_slots = [int(value) for value in fields["read_slots"]]
            write_slots = [int(value) for value in fields["write_slots"]]
            replacement_events = [
                slot for slot, flag in zip(write_slots, fields["replaced"], strict=True) if flag > 0.5
            ]
            observer.emit(
                ObservationEvent(
                    event="memory_step",
                    step=token_index,
                    token_index=token_index,
                    token_id=int(token_ids[offset]),
                    layer_id="memory",
                    payload={
                        "slots": slots,
                        "query_source": self.memory.query_source,
                        "replacement_policy": self.memory.replacement,
                        # Was der Zugriff selbst getan hat.
                        "selected_read_slots": read_slots,
                        "read_scores": fields["read_scores"],
                        "read_weights": fields["read_weights"],
                        "read_gate": fields["read_gate"][0] if fields["read_gate"] else 0.0,
                        "read_output_rms": fields["read_output_rms"][0],
                        "selected_write_slots": write_slots,
                        "write_scores": fields["write_scores"],
                        "write_strength": fields["write_strength"],
                        "write_gate": fields["write_gate"][0] if fields["write_gate"] else 0.0,
                        "replacement_events": replacement_events,
                        # Zustand der gesamten Bank nach dem Zugriff.
                        "slot_strength": fields["slot_strength"],
                        "slot_age": fields["slot_age"],
                        "slot_usage": fields["slot_usage"],
                        "slot_read_count": fields["slot_read_count"],
                        "slot_write_count": fields["slot_write_count"],
                        "slot_occupied": fields["slot_occupied"],
                        # Woher die Query kam und wohin das Ergebnis fließt.
                        "source_state": self.memory.query_source,
                        "destination_state": f"core.{self.config.memory_layer_index}.output",
                    },
                )
            )

    def _emit_predictions(self, observer: ObservationBus, input_ids: Tensor, logits: Tensor, start: int) -> None:
        probabilities = F.softmax(logits.detach().float(), dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)
        top_probability, top_token = probabilities.topk(min(5, probabilities.shape[-1]), dim=-1)
        top_logits = torch.gather(logits.detach().float(), dim=-1, index=top_token)
        # Ein Transfer je Größe für die gesamte Sequenz statt einer je Token.
        entropies = entropy[0].cpu().tolist()
        tokens = top_token[0].cpu().tolist()
        token_probabilities = top_probability[0].cpu().tolist()
        token_logits = top_logits[0].cpu().tolist()
        summaries = sequence_tensor_summary(logits)
        token_ids = input_ids[0].tolist()
        for offset in range(input_ids.shape[1]):
            observer.emit(
                ObservationEvent(
                    event="prediction",
                    step=start + offset,
                    token_index=start + offset,
                    token_id=int(token_ids[offset]),
                    layer_id="lm_head",
                    payload={
                        "entropy": entropies[offset],
                        "top_tokens": tokens[offset],
                        "top_probabilities": token_probabilities[offset],
                        "top_logits": token_logits[offset],
                        "logit_summary": summaries[offset],
                    },
                )
            )

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.0,
        observer: ObservationBus | None = None,
        ablate_states: Collection[str] | None = None,
        **memory_options: Any,
    ) -> Tensor:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens darf nicht negativ sein")
        logits, state = self.forward(
            prompt_ids,
            observer=observer,
            ablate_states=ablate_states,
            **memory_options,
        )
        generated = [prompt_ids]
        next_logits = logits[:, -1]
        for _ in range(max_new_tokens):
            if temperature <= 0:
                token = next_logits.argmax(dim=-1)
            else:
                probabilities = F.softmax(next_logits / temperature, dim=-1)
                token = torch.multinomial(probabilities, num_samples=1).squeeze(1)
            generated.append(token[:, None])
            next_logits, state = self.step(
                token,
                state,
                observer=observer,
                ablate_states=ablate_states,
                **memory_options,
            )
        return torch.cat(generated, dim=1)
