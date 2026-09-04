from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from glassmind.precision.policy import PrecisionPolicy, resolve_dtype
from glassmind.precision.quantization import linear_weight


@dataclass
class BlockState:
    fast: Tensor
    context: Tensor
    semantic: Tensor

    def detach(self) -> "BlockState":
        return BlockState(self.fast.detach(), self.context.detach(), self.semantic.detach())


@dataclass
class StepMetrics:
    fast: Tensor
    context: Tensor
    semantic: Tensor
    fast_delta: Tensor
    context_delta: Tensor
    semantic_delta: Tensor
    fast_gate: Tensor
    context_gate: Tensor
    semantic_gate: Tensor
    input_flow: Tensor
    fast_context_flow: Tensor
    context_semantic_flow: Tensor
    output_flow: Tensor
    fast_output_flow: Tensor
    context_output_flow: Tensor
    semantic_output_flow: Tensor
    binding_flow: Tensor
    query_read_flow: Tensor


class SelectiveStateBlock(nn.Module):
    """Dreistufiger selektiv-rekurrenter Block ohne Token-zu-Token-Mischung.

    Zur Einordnung: Die gebundene State-Interaktion
    (``semantic = tanh(prev + gate * (key ⊗ value))`` mit
    ``read = semantic · key``) ist mathematisch mit linearer Attention
    verwandt. Der Unterschied ist derselbe wie beim externen Speicher: Der
    Zustand hat feste Breite und wächst nicht mit der Sequenz, es gibt keine
    Token-zu-Token-Matrix und keinen Cache über vergangene Positionen.

    Der Block ist in zwei Teile getrennt:

    * Ein **sequenzweiter** Teil, der ausschließlich von der Eingabe abhängt
      (LayerNorm, Eingangsprojektion, Gates, Ausgangs-Gate). Er wird einmal für
      die gesamte Sequenz berechnet, nicht pro Token.
    * Ein **rekurrenter** Teil, der pro Token laufen muss. Dort sind alle
      Projektionen zusammengefasst, die dieselbe Eingabe teilen.

    Beide Teile sind mathematisch identisch zur ursprünglichen, entbündelten
    Formulierung; :meth:`reference_forward` hält diesen langsamen Referenzpfad
    zum Vergleich bereit.
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.0,
        *,
        state_interactions: bool = False,
        recurrent_autocast: bool = False,
        precision: PrecisionPolicy | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.state_interactions = state_interactions
        self.recurrent_autocast = recurrent_autocast
        self.precision = precision or PrecisionPolicy()
        if state_interactions:
            self.binding_rank = max(2, math.isqrt(d_model))
            self.binding_size = self.binding_rank * self.binding_rank
            # Im gebundenen Pfad trägt der semantische Zustand genau die
            # Rang-r-Bindungsmatrix. Frühere Versionen führten ihn auf voller
            # Breite und ließen die überzähligen Kanäle dauerhaft auf null.
            self.semantic_width = self.binding_size
            # f_value, f_gate, c_value, c_gate – der semantische Zweig der
            # Eingangsprojektion wird im gebundenen Pfad nicht gelesen.
            self.input_parts = 4
            side = self.binding_rank + 1
        else:
            self.binding_rank = 0
            self.binding_size = 0
            self.semantic_width = d_model
            self.input_parts = 6
            side = 0

        self.norm = nn.LayerNorm(d_model)
        # Eingangsprojektion und Ausgangs-Gate teilen dieselbe Eingabe und sind
        # deshalb zu einer Projektion zusammengefasst; der letzte d_model-Block
        # ist das Ausgangs-Gate.
        self.input_proj = nn.Linear(d_model, (self.input_parts + 1) * d_model)
        # Alles, was aus dem vorherigen fast_state gelesen wird.
        self.pre_state_proj = nn.Linear(d_model, d_model + side, bias=False)
        # Alles, was aus dem neuen fast_state gelesen wird.
        self.post_state_proj = nn.Linear(d_model, d_model + side, bias=False)
        self.context_recurrent = nn.Linear(d_model, d_model, bias=False)
        self.integrator = nn.Linear(
            2 * d_model + self.semantic_width + self.binding_rank, d_model
        )
        if state_interactions:
            self.binding_gate_bias = nn.Parameter(torch.zeros(1))
        else:
            self.semantic_from_context = nn.Linear(d_model, d_model, bias=False)
            self.semantic_recurrent = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # Unterschiedliche Start-Zeitskalen; alle Werte bleiben lernbar.
        self.fast_bias = nn.Parameter(torch.full((d_model,), 1.5))
        self.context_bias = nn.Parameter(torch.full((d_model,), -1.0))
        if not state_interactions:
            self.semantic_bias = nn.Parameter(torch.full((d_model,), -2.5))

    # ------------------------------------------------------------------
    # Aufbau und Zustand
    # ------------------------------------------------------------------

    def reset_interaction_parameters(self) -> None:
        if not self.state_interactions:
            return
        d, rank = self.d_model, self.binding_rank
        with torch.no_grad():
            # Schlüssel-, Wert- und Lesecode behalten ihre ursprüngliche
            # Xavier-Initialisierung; nur ihre Ablage ist jetzt fusioniert.
            nn.init.xavier_uniform_(self.pre_state_proj.weight[d : d + rank])
            nn.init.xavier_uniform_(self.post_state_proj.weight[d : d + rank])
            nn.init.xavier_uniform_(self.integrator.weight[:, 2 * d + self.semantic_width :])
            # Das Schreib-Gate sah ursprünglich beide fast_states als eine
            # Matrix [1, 2*d_model]; die Fan-in-Berechnung bleibt deshalb bei 2*d.
            write_gate = torch.empty(1, 2 * d, device=self.pre_state_proj.weight.device)
            nn.init.xavier_uniform_(write_gate)
            self.pre_state_proj.weight[d + rank :].copy_(write_gate[:, :d])
            self.post_state_proj.weight[d + rank :].copy_(write_gate[:, d:])
            self.binding_gate_bias.fill_(-2.0)

    def state_dtypes(self, fallback: torch.dtype) -> tuple[torch.dtype, torch.dtype, torch.dtype]:
        """Speicher-dtypes der drei Zustände laut Policy."""
        return self.precision.state_dtypes(fallback)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> BlockState:
        fast_dtype, context_dtype, semantic_dtype = self.state_dtypes(dtype)
        fast = torch.zeros(batch_size, self.d_model, device=device, dtype=fast_dtype)
        context = (
            fast.clone()
            if context_dtype is fast_dtype
            else torch.zeros(batch_size, self.d_model, device=device, dtype=context_dtype)
        )
        semantic = torch.zeros(
            batch_size, self.semantic_width, device=device, dtype=semantic_dtype
        )
        return BlockState(fast, context, semantic)

    # ------------------------------------------------------------------
    # Sequenzweiter Vorlauf
    # ------------------------------------------------------------------

    def _project_sequence(
        self, x: Tensor, activation_dtype: torch.dtype, compute_dtype: torch.dtype
    ) -> tuple[Tensor, ...]:
        """Berechnet alle rein eingabeabhängigen Größen einmal für die Sequenz.

        Die Gates entstehen im Aktivierungsdatentyp; die Umwandlung in den
        Rechendatentyp geschieht danach genau einmal je Sequenz, nicht je Token.
        """
        projected = self.input_proj(self.norm(x))
        if projected.dtype is not activation_dtype:
            projected = projected.to(activation_dtype)
        parts = projected.split(self.d_model, dim=-1)
        if self.state_interactions:
            f_value, f_gate_raw, c_value, c_gate_raw, out_gate_raw = parts
            s_value = s_gate_raw = None
        else:
            f_value, f_gate_raw, c_value, c_gate_raw, s_value, s_gate_raw, out_gate_raw = parts
        fast_gate = torch.sigmoid(f_gate_raw + self.fast_bias)
        context_gate = torch.sigmoid(c_gate_raw + self.context_bias)
        semantic_gate = (
            None if self.state_interactions else torch.sigmoid(s_gate_raw + self.semantic_bias)
        )
        out_gate = torch.sigmoid(out_gate_raw)
        result = (f_value, fast_gate, c_value, context_gate, s_value, semantic_gate, out_gate)
        # Die Bias-Parameter liegen im Modell-dtype; ihre Addition kann die Gates
        # wieder hochstufen. Stimmen Aktivierungs-, Rechen- und Parameterdatentyp
        # überein – der Normalfall –, ist nichts zu tun und der Tokenpfad zahlt
        # nur zwei Typidentitätsvergleiche.
        if activation_dtype is compute_dtype and self.fast_bias.dtype is compute_dtype:
            return result
        return tuple(
            item if item is None or item.dtype is compute_dtype else item.to(compute_dtype)
            for item in result
        )

    def _integrator_parts(
        self, dtype: torch.dtype | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
        """Zerlegt die fusionierte Integrator-Matrix für die Telemetrie."""
        d, width = self.d_model, self.semantic_width
        weight = linear_weight(self.integrator, dtype or torch.get_default_dtype())
        read = weight[:, 2 * d + width :] if self.state_interactions else None
        return weight[:, :d], weight[:, d : 2 * d], weight[:, 2 * d : 2 * d + width], read

    # ------------------------------------------------------------------
    # Rekurrenter Pfad
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        state: BlockState | None = None,
        *,
        collect_metrics: bool = False,
        ablate_states: frozenset[str] | None = None,
        state_trace: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, BlockState, list[StepMetrics] | None]:
        """``state_trace`` sammelt optional die Zustände je Token.

        Das braucht nur das externe Memory aus Milestone 3, und nur wenn seine
        Query aus einem Zustand statt aus dem Blockausgang gebildet wird. Ohne
        das Dictionary kostet die Erweiterung im Tokenpfad einen einzigen
        ``is not None``-Test.
        """
        if x.ndim != 3:
            raise ValueError("x muss die Form [batch, sequence, d_model] besitzen")
        batch, length, _ = x.shape
        if state is None:
            state = self.initial_state(batch, device=x.device, dtype=x.dtype)
        policy = self.precision
        # ``inherit`` behält exakt das Milestone-2.5-Verhalten bei: gerechnet
        # wird in der Präzision des übergebenen Zustands.
        compute_dtype = (
            state.fast.dtype
            if policy.inherits_compute
            else resolve_dtype(policy.compute, x.dtype)
        )
        activation_dtype = (
            compute_dtype
            if policy.inherits_activations
            else resolve_dtype(policy.activations, compute_dtype)
        )

        f_value, fast_gate, c_value, context_gate, s_value, semantic_gate, out_gate = (
            self._project_sequence(x, activation_dtype, compute_dtype)
        )

        device_type = x.device.type
        # Der rekurrente Teil rechnet in der Präzision des Zustands. Ohne diese
        # Abschaltung fügt Autocast pro Token und Projektion je einen eigenen
        # Cast-Kernel ein, was bei kleinen Schrittgrößen teurer ist als die
        # eingesparten Matmul-Zyklen.
        disable_autocast = (
            not self.recurrent_autocast and torch.is_autocast_enabled(device_type)
        )
        if disable_autocast:
            with torch.autocast(device_type=device_type, enabled=False):
                flows, next_state, metrics = self._recurrent_scan(
                    batch,
                    length,
                    state,
                    compute_dtype,
                    f_value,
                    fast_gate,
                    c_value,
                    context_gate,
                    s_value,
                    semantic_gate,
                    out_gate,
                    collect_metrics=collect_metrics,
                    ablate_states=ablate_states,
                    state_trace=state_trace,
                )
        else:
            flows, next_state, metrics = self._recurrent_scan(
                batch,
                length,
                state,
                compute_dtype,
                f_value,
                fast_gate,
                c_value,
                context_gate,
                s_value,
                semantic_gate,
                out_gate,
                collect_metrics=collect_metrics,
                ablate_states=ablate_states,
                state_trace=state_trace,
            )

        # Residual und Dropout wirken pro Token unabhängig und werden deshalb
        # einmal auf die gestapelte Sequenz angewendet.
        output = x + self.dropout(flows.to(x.dtype))
        return output, next_state, metrics

    def _recurrent_scan(
        self,
        batch: int,
        length: int,
        state: BlockState,
        compute_dtype: torch.dtype,
        f_value: Tensor,
        fast_gate: Tensor,
        c_value: Tensor,
        context_gate: Tensor,
        s_value: Tensor | None,
        semantic_gate: Tensor | None,
        out_gate: Tensor,
        *,
        collect_metrics: bool,
        ablate_states: frozenset[str] | None,
        state_trace: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, BlockState, list[StepMetrics] | None]:
        d = self.d_model
        linear, tanh, sigmoid, lerp, cat, silu, vector_norm = (
            F.linear,
            torch.tanh,
            torch.sigmoid,
            torch.lerp,
            torch.cat,
            F.silu,
            torch.linalg.vector_norm,
        )
        # Genau ein Zugriff je Block und Sequenz. Bei quantisierten Modulen
        # findet hier die einzige Dequantisierung statt, nicht im Tokenpfad.
        w_pre = linear_weight(self.pre_state_proj, compute_dtype)
        w_post = linear_weight(self.post_state_proj, compute_dtype)
        w_context = linear_weight(self.context_recurrent, compute_dtype)
        w_integrator = linear_weight(self.integrator, compute_dtype)
        b_integrator = self.integrator.bias
        if b_integrator is not None and b_integrator.dtype is not compute_dtype:
            b_integrator = b_integrator.to(compute_dtype)

        f_values = f_value.unbind(1)
        f_gates = fast_gate.unbind(1)
        c_values = c_value.unbind(1)
        c_gates = context_gate.unbind(1)
        o_gates = out_gate.unbind(1)
        s_values = None if s_value is None else s_value.unbind(1)
        s_gates = None if semantic_gate is None else semantic_gate.unbind(1)

        ablations = ablate_states or frozenset()
        fast_stored, context_stored, semantic_stored = state.fast, state.context, state.semantic
        # Speicher-dtypes der Zustände. Stimmen sie mit dem Rechendatentyp
        # überein – der Normalfall –, entsteht im Tokenpfad kein einziger
        # zusätzlicher Cast; die Prüfung ist ein reiner Bool-Vergleich.
        fast_dtype = fast_stored.dtype
        context_dtype = context_stored.dtype
        semantic_dtype = semantic_stored.dtype
        fast_native = fast_dtype is compute_dtype
        context_native = context_dtype is compute_dtype
        semantic_native = semantic_dtype is compute_dtype
        zero_fast = (
            torch.zeros(batch, d, device=f_value.device, dtype=compute_dtype)
            if "fast" in ablations
            else None
        )
        zero_context = (
            torch.zeros(batch, d, device=f_value.device, dtype=compute_dtype)
            if "context" in ablations
            else None
        )
        zero_semantic = (
            torch.zeros(
                batch, self.semantic_width, device=f_value.device, dtype=compute_dtype
            )
            if "semantic" in ablations
            else None
        )

        # Lokale Listenreferenzen: im Tokenpfad ist das ein Bool-Test, kein
        # Dictionary-Zugriff.
        trace_fast: list[Tensor] | None = [] if state_trace is not None else None
        trace_context: list[Tensor] | None = [] if state_trace is not None else None
        trace_semantic: list[Tensor] | None = [] if state_trace is not None else None

        metrics: list[StepMetrics] | None = [] if collect_metrics else None
        if collect_metrics:
            fast_weight, context_weight, semantic_weight, read_weight = self._integrator_parts(
                compute_dtype
            )

        outputs: list[Tensor] = []
        if self.state_interactions:
            rank, width = self.binding_rank, self.semantic_width
            pre_sizes = (d, rank, 1)
            read_scale = math.sqrt(rank)
            # Auch skalare Parameter müssen im Rechendatentyp vorliegen, sonst
            # stuft ihre Addition den ganzen Zweig wieder hoch.
            gate_bias = self.binding_gate_bias
            if gate_bias.dtype is not compute_dtype:
                gate_bias = gate_bias.to(compute_dtype)
            for index in range(length):
                previous_fast_stored = fast_stored
                previous_semantic_stored = semantic_stored
                # Gespeicherte Zustände in den Rechendatentyp heben. Bei
                # gleichem dtype ist das ein reiner Bool-Test ohne Kernel.
                previous_fast = (
                    previous_fast_stored if fast_native else previous_fast_stored.to(compute_dtype)
                )
                previous_context = (
                    context_stored if context_native else context_stored.to(compute_dtype)
                )
                previous_semantic = (
                    previous_semantic_stored
                    if semantic_native
                    else previous_semantic_stored.to(compute_dtype)
                )

                pre = linear(previous_fast, w_pre)
                fast_recurrent, key_raw, write_gate_previous = pre.split(pre_sizes, dim=-1)
                fast_candidate = tanh(f_values[index] + fast_recurrent)
                fast = (
                    zero_fast
                    if zero_fast is not None
                    else lerp(previous_fast, fast_candidate, f_gates[index])
                )
                fast_stored = fast if fast_native else fast.to(fast_dtype)

                post = linear(fast, w_post)
                fast_to_context, value_raw, write_gate_current = post.split(pre_sizes, dim=-1)
                context_candidate = tanh(
                    c_values[index] + fast_to_context + linear(previous_context, w_context)
                )
                context = (
                    zero_context
                    if zero_context is not None
                    else lerp(previous_context, context_candidate, c_gates[index])
                )
                context_stored = context if context_native else context.to(context_dtype)

                # Entspricht F.normalize(..., dim=-1, eps=1e-6), spart aber den
                # zusätzlichen expand_as-Aufruf im Tokenpfad.
                key_code = tanh(key_raw)
                key_code = key_code / vector_norm(key_code, dim=-1, keepdim=True).clamp_min(1e-6)
                value_code = tanh(value_raw)
                binding_flow = (key_code.unsqueeze(-1) * value_code.unsqueeze(1)).flatten(1)
                binding_gate = sigmoid(write_gate_previous + write_gate_current + gate_bias)
                semantic = (
                    zero_semantic
                    if zero_semantic is not None
                    else tanh(previous_semantic + binding_gate * binding_flow)
                )
                semantic_stored = semantic if semantic_native else semantic.to(semantic_dtype)

                # Bewusst als Broadcast-Multiplikation mit Reduktion und nicht
                # als torch.bmm: der bmm-Backward wird in aktuellen Builds über
                # einen Triton-Kernel geführt, der ohne CPython-Header und
                # passenden Compiler nicht baubar ist. Diese Form ist reines
                # ATen und läuft auf jedem Backend.
                read_code = (
                    semantic.view(batch, rank, rank) * key_code.unsqueeze(-1)
                ).sum(dim=1) * read_scale
                integrated = linear(
                    cat((fast, context, semantic, read_code), dim=-1),
                    w_integrator,
                    b_integrator,
                )
                output_flow = silu(integrated) * o_gates[index]
                outputs.append(output_flow)
                if trace_fast is not None:
                    trace_fast.append(fast)
                    trace_context.append(context)
                    trace_semantic.append(semantic)
                if metrics is not None:
                    metrics.append(
                        StepMetrics(
                            fast=fast,
                            context=context,
                            semantic=semantic,
                            fast_delta=fast - previous_fast,
                            context_delta=context - previous_context,
                            semantic_delta=semantic - previous_semantic,
                            fast_gate=f_gates[index],
                            context_gate=c_gates[index],
                            semantic_gate=binding_gate.expand(batch, width),
                            input_flow=f_values[index],
                            fast_context_flow=fast_to_context,
                            context_semantic_flow=binding_flow,
                            output_flow=output_flow,
                            fast_output_flow=linear(fast, fast_weight),
                            context_output_flow=linear(context, context_weight),
                            semantic_output_flow=linear(semantic, semantic_weight),
                            binding_flow=binding_flow,
                            query_read_flow=linear(read_code, read_weight),
                        )
                    )
        else:
            w_semantic_from_context = linear_weight(self.semantic_from_context, compute_dtype)
            w_semantic_recurrent = linear_weight(self.semantic_recurrent, compute_dtype)
            for index in range(length):
                previous_fast = fast_stored if fast_native else fast_stored.to(compute_dtype)
                previous_context = (
                    context_stored if context_native else context_stored.to(compute_dtype)
                )
                previous_semantic = (
                    semantic_stored if semantic_native else semantic_stored.to(compute_dtype)
                )

                fast_candidate = tanh(f_values[index] + linear(previous_fast, w_pre))
                fast = (
                    zero_fast
                    if zero_fast is not None
                    else lerp(previous_fast, fast_candidate, f_gates[index])
                )
                fast_stored = fast if fast_native else fast.to(fast_dtype)

                fast_to_context = linear(fast, w_post)
                context_candidate = tanh(
                    c_values[index] + fast_to_context + linear(previous_context, w_context)
                )
                context = (
                    zero_context
                    if zero_context is not None
                    else lerp(previous_context, context_candidate, c_gates[index])
                )
                context_stored = context if context_native else context.to(context_dtype)

                context_to_semantic = linear(context, w_semantic_from_context)
                semantic_candidate = tanh(
                    s_values[index]
                    + context_to_semantic
                    + linear(previous_semantic, w_semantic_recurrent)
                )
                semantic = (
                    zero_semantic
                    if zero_semantic is not None
                    else lerp(previous_semantic, semantic_candidate, s_gates[index])
                )
                semantic_stored = semantic if semantic_native else semantic.to(semantic_dtype)

                integrated = linear(
                    cat((fast, context, semantic), dim=-1), w_integrator, b_integrator
                )
                output_flow = silu(integrated) * o_gates[index]
                outputs.append(output_flow)
                if trace_fast is not None:
                    trace_fast.append(fast)
                    trace_context.append(context)
                    trace_semantic.append(semantic)
                if metrics is not None:
                    zero_flow = torch.zeros_like(integrated)
                    metrics.append(
                        StepMetrics(
                            fast=fast,
                            context=context,
                            semantic=semantic,
                            fast_delta=fast - previous_fast,
                            context_delta=context - previous_context,
                            semantic_delta=semantic - previous_semantic,
                            fast_gate=f_gates[index],
                            context_gate=c_gates[index],
                            semantic_gate=s_gates[index],
                            input_flow=f_values[index],
                            fast_context_flow=fast_to_context,
                            context_semantic_flow=context_to_semantic,
                            output_flow=output_flow,
                            fast_output_flow=linear(fast, fast_weight),
                            context_output_flow=linear(context, context_weight),
                            semantic_output_flow=linear(semantic, semantic_weight),
                            binding_flow=torch.zeros_like(context_to_semantic),
                            query_read_flow=zero_flow,
                        )
                    )

        if state_trace is not None:
            state_trace["fast"] = torch.stack(trace_fast, dim=1)
            state_trace["context"] = torch.stack(trace_context, dim=1)
            state_trace["semantic"] = torch.stack(trace_semantic, dim=1)
        return (
            torch.stack(outputs, dim=1),
            BlockState(fast_stored, context_stored, semantic_stored),
            metrics,
        )

    # ------------------------------------------------------------------
    # Langsamer Referenzpfad
    # ------------------------------------------------------------------

    @torch.no_grad()
    def reference_forward(
        self,
        x: Tensor,
        state: BlockState | None = None,
        *,
        ablate_states: frozenset[str] | None = None,
    ) -> tuple[Tensor, BlockState]:
        """Bewusst einfache, unfusionierte Referenz für Korrektheitsvergleiche.

        Diese Variante führt jede Teilprojektion einzeln und pro Token aus. Sie
        ist deutlich langsamer als :meth:`forward`, macht aber sichtbar, welche
        Rechnung der optimierte Pfad ausführt. Quantisierte Gewichte werden
        dafür dequantisiert; der Referenzpfad kennt keine Sonderfälle.
        """
        if x.ndim != 3:
            raise ValueError("x muss die Form [batch, sequence, d_model] besitzen")
        batch = x.shape[0]
        if state is None:
            state = self.initial_state(batch, device=x.device, dtype=x.dtype)
        d, rank = self.d_model, self.binding_rank
        ablations = ablate_states or frozenset()
        dtype = x.dtype
        w_pre = linear_weight(self.pre_state_proj, dtype)
        w_post = linear_weight(self.post_state_proj, dtype)
        w_context = linear_weight(self.context_recurrent, dtype)
        fast = state.fast.to(dtype)
        context = state.context.to(dtype)
        semantic = state.semantic.to(dtype)
        outputs: list[Tensor] = []
        for token in x.unbind(dim=1):
            normalized = self.norm(token)
            projected = self.input_proj(normalized)
            parts = projected.split(d, dim=-1)
            out_gate = torch.sigmoid(parts[-1])
            fast_gate = torch.sigmoid(parts[1] + self.fast_bias)
            context_gate = torch.sigmoid(parts[3] + self.context_bias)

            previous_fast, previous_context, previous_semantic = fast, context, semantic
            fast_candidate = torch.tanh(parts[0] + F.linear(previous_fast, w_pre[:d]))
            fast = torch.lerp(previous_fast, fast_candidate, fast_gate)
            if "fast" in ablations:
                fast = torch.zeros_like(fast)

            fast_to_context = F.linear(fast, w_post[:d])
            context_candidate = torch.tanh(
                parts[2] + fast_to_context + F.linear(previous_context, w_context)
            )
            context = torch.lerp(previous_context, context_candidate, context_gate)
            if "context" in ablations:
                context = torch.zeros_like(context)

            if self.state_interactions:
                key_code = F.normalize(
                    torch.tanh(F.linear(previous_fast, w_pre[d : d + rank])),
                    dim=-1,
                    eps=1e-6,
                )
                value_code = torch.tanh(F.linear(fast, w_post[d : d + rank]))
                binding_flow = (key_code.unsqueeze(-1) * value_code.unsqueeze(1)).flatten(1)
                binding_gate = torch.sigmoid(
                    F.linear(previous_fast, w_pre[d + rank :])
                    + F.linear(fast, w_post[d + rank :])
                    + self.binding_gate_bias
                )
                semantic = torch.tanh(previous_semantic + binding_gate * binding_flow)
            else:
                semantic_gate = torch.sigmoid(parts[5] + self.semantic_bias)
                semantic_candidate = torch.tanh(
                    parts[4]
                    + F.linear(context, linear_weight(self.semantic_from_context, dtype))
                    + F.linear(previous_semantic, linear_weight(self.semantic_recurrent, dtype))
                )
                semantic = torch.lerp(previous_semantic, semantic_candidate, semantic_gate)
            if "semantic" in ablations:
                semantic = torch.zeros_like(semantic)

            pieces = [fast, context, semantic]
            if self.state_interactions:
                read_code = (
                    semantic.view(batch, rank, rank) * key_code.unsqueeze(-1)
                ).sum(dim=1) * math.sqrt(rank)
                pieces.append(read_code)
            integrated = F.linear(
                torch.cat(pieces, dim=-1),
                linear_weight(self.integrator, dtype),
                self.integrator.bias,
            )
            outputs.append(F.silu(integrated) * out_gate)
        return x + torch.stack(outputs, dim=1), BlockState(fast, context, semantic)
