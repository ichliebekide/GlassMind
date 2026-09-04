#!/usr/bin/env python3
"""Erzeugt aus der Precision-Matrix lesbare Tabellen (Text oder Markdown).

Die Auswertung rechnet nichts nach, sie ordnet nur die gemessenen Werte. Alle
Zahlen stammen unverändert aus ``benchmarks/milestone2_6-precision.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

STATE_NAMES = ("fast", "context", "semantic")


def accuracy(record: dict[str, Any], task: str, distance: int) -> float | None:
    for entry in record.get("tasks", []):
        if entry["task"] == task and entry["distance"] == distance:
            return float(entry["accuracy"])
    return None


def distances_of(record: dict[str, Any]) -> list[int]:
    return sorted({int(entry["distance"]) for entry in record.get("tasks", [])})


def _bytes(value: Any) -> str:
    if value is None:
        return "–"
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{value:,.0f} B"
        value /= 1024
    return f"{value:,.1f} GiB"


def _number(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "ja" if value else "nein"
    value = float(value)
    if value != value:  # NaN
        return "NaN"
    if value in (float("inf"), float("-inf")):
        return "∞"
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:.{digits}f}"


def _percent(value: Any) -> str:
    return "–" if value is None else f"{float(value):.1%}"


def render(rows: list[list[str]], header: list[str], markdown: bool) -> str:
    widths = [max(len(header[i]), *(len(row[i]) for row in rows)) for i in range(len(header))]
    if markdown:
        lines = ["| " + " | ".join(header) + " |",
                 "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)
    lines = ["  ".join(head.ljust(widths[i]) for i, head in enumerate(header))]
    lines.append("-" * len(lines[0]))
    lines += ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join(lines)


def overview_table(records: Sequence[dict[str, Any]], markdown: bool) -> str:
    distances = distances_of(records[0]) if records else []
    header = ["Konfiguration", "Token/s", "ms/Tok", "Train Tok/s", "VRAM", "RAM",
              "Gewichte", "Checkpoint"]
    header += [f"Recall {d}" for d in distances] + [f"Copy {d}" for d in distances]
    header += ["Logit-Drift"] + [f"Norm-Δ {n}" for n in STATE_NAMES]
    rows = []
    for record in records:
        training = record.get("training_tokens_per_second")
        row = [
            record["label"],
            _number(record["tokens_per_second"]),
            _number(record["streaming_ms_per_token"]),
            "n. a." if training is None else _number(training),
            _bytes(record.get("peak_memory_bytes")),
            _bytes(record.get("host_peak_rss_bytes")),
            _bytes(record.get("parameter_storage_bytes")),
            _bytes(record.get("checkpoint_bytes")),
        ]
        row += [_percent(accuracy(record, "associative_recall", d)) for d in distances]
        row += [_percent(accuracy(record, "selective_copy", d)) for d in distances]
        row.append(_number(record.get("logit_relative_rms_vs_reference")))
        row += [_number(record.get("norm_delta", {}).get(name)) for name in STATE_NAMES]
        rows.append(row)
    return render(rows, header, markdown)


def state_matrix_table(records: Sequence[dict[str, Any]], markdown: bool) -> str:
    distances = distances_of(records[0]) if records else []
    header = ["fast", "context", "semantic", "Token/s", "ms/Tok"]
    header += [f"Recall {d}" for d in distances] + [f"Copy {d}" for d in distances]
    header += [f"Norm-Δ {n}" for n in STATE_NAMES]
    short = {"float32": "fp32", "bfloat16": "bf16", "float16": "fp16"}
    rows = []
    for record in records:
        policy = record["policy"]
        row = [short.get(policy["fast_state"], policy["fast_state"]),
               short.get(policy["context_state"], policy["context_state"]),
               short.get(policy["semantic_state"], policy["semantic_state"]),
               _number(record["tokens_per_second"]),
               _number(record["streaming_ms_per_token"])]
        row += [_percent(accuracy(record, "associative_recall", d)) for d in distances]
        row += [_percent(accuracy(record, "selective_copy", d)) for d in distances]
        row += [_number(record.get("norm_delta", {}).get(name)) for name in STATE_NAMES]
        rows.append(row)
    return render(rows, header, markdown)


def drift_table(entry: dict[str, Any], markdown: bool) -> str:
    header = ["Länge", "fast", "context", "semantic", "Logit", "Vorhersagen", "NaN/Inf"]
    rows = []
    for point in entry["points"]:
        rows.append([
            f"{point['length']:,}",
            _number(point["state_relative_rms"]["fast"]),
            _number(point["state_relative_rms"]["context"]),
            _number(point["state_relative_rms"]["semantic"]),
            _number(point["logit_relative_rms"]),
            _percent(point["prediction_change_rate"]),
            "ja" if point["has_nan"] or point["has_inf"] else "nein",
        ])
    return render(rows, header, markdown)


def telemetry_table(entries: Sequence[dict[str, Any]], markdown: bool) -> str:
    header = ["Variante", "Zustand", "Aktivität", "Delta", "Fluss", "Persistenz", "Reaktivierung"]
    rows = []
    for entry in entries:
        for kind, values in entry.get("by_state", {}).items():
            rows.append([
                entry["label"], kind,
                _number(values["activity_deviation"]),
                _number(values["delta_deviation"]),
                _number(values["flow_deviation"]),
                _number(values["persistence_deviation"]),
                _number(values["reactivation_deviation"]),
            ])
    return render(rows, header, markdown)


def add_norm_delta(records: Sequence[dict[str, Any]], reference: dict[str, Any]) -> None:
    for record in records:
        record["norm_delta"] = {
            name: abs(record["final_state_norms"][name] - reference["final_state_norms"][name])
            / max(abs(reference["final_state_norms"][name]), 1e-12)
            for name in STATE_NAMES
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Bereitet die Precision-Matrix als Tabellen auf")
    parser.add_argument("--input", type=Path, default=Path("benchmarks/milestone2_6-precision.json"))
    parser.add_argument("--reference", type=Path, default=Path("benchmarks/milestone2_5-reference.json"))
    parser.add_argument("--markdown", action="store_true")
    parser.add_argument("--sections", nargs="+",
                        default=["float", "states", "quantization", "drift", "telemetry", "winners"])
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    reference = json.loads(args.reference.read_text(encoding="utf-8"))["reference"]
    sections = data["sections"]
    heading = (lambda text: f"\n## {text}\n") if args.markdown else (lambda text: f"\n=== {text} ===\n")

    for key in ("float", "states", "quantization"):
        if key in sections:
            add_norm_delta(sections[key], reference)

    if "float" in args.sections and "float" in sections:
        print(heading("Gleitkomma-Grundvarianten"))
        print(overview_table(sections["float"], args.markdown))
    if "quantization" in args.sections and "quantization" in sections:
        print(heading("Weight-Only-Quantisierung"))
        print(overview_table(sections["quantization"], args.markdown))
    if "states" in args.sections and "states" in sections:
        print(heading("State-dtype-Matrix"))
        print(state_matrix_table(sections["states"], args.markdown))
    if "drift" in args.sections and "drift" in sections:
        print(heading("Langzeitdrift gegen FP32"))
        for entry in sections["drift"]:
            print(f"\n### {entry['label']}\n" if args.markdown else f"\n[{entry['label']}]")
            print(drift_table(entry, args.markdown))
            summary = entry["summary"]
            onset = summary["state_drift_onset_length"]
            print(f"\nDrift über 1 % ab Länge: fast={onset['fast'] or '–'}, "
                  f"context={onset['context'] or '–'}, semantic={onset['semantic'] or '–'}; "
                  f"Logits={summary['logit_drift_onset_length'] or '–'}; "
                  f"erste geänderte Vorhersage={summary['prediction_change_onset_length'] or '–'}")
    if "telemetry" in args.sections and "telemetry" in sections:
        print(heading("Abweichung der sichtbaren Aktivität (Observation Bus)"))
        print(telemetry_table(sections["telemetry"], args.markdown))
    if "winners" in args.sections and "winners" in data:
        print(heading("Bewertung"))
        labels = {"schnellste": "schnellste Variante",
                  "speichereffizienteste": "speichereffizienteste Variante",
                  "stabilste": "stabilste Variante",
                  "beste_gesamt": "beste Gesamtvariante"}
        for key, value in data["winners"].items():
            detail = ", ".join(f"{k}={_number(v)}" for k, v in value.items() if k != "label")
            print(f"- **{labels.get(key, key)}**: {value['label']} ({detail})")
        policy = data["auto_policy"]
        print(f"\n- **auto-Profil**: compute={policy['compute']}, fast={policy['fast_state']}, "
              f"context={policy['context_state']}, semantic={policy['semantic_state']}")
        for note in policy.get("selection_notes", []):
            print(f"  - {note}")


if __name__ == "__main__":
    main()
