from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
from typing import Any


class RunDirectory:
    def __init__(self, root: str | Path = "runs", *, prefix: str = "run") -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.path = Path(root) / f"{prefix}-{stamp}-{secrets.token_hex(3)}"
        self.checkpoints = self.path / "checkpoints"
        self.traces = self.path / "traces"
        self.checkpoints.mkdir(parents=True)
        self.traces.mkdir()
        self.metrics_path = self.path / "metrics.jsonl"
        self.log_path = self.path / "train.log"

    def write_json(self, name: str, data: dict[str, Any]) -> None:
        (self.path / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def metric(self, data: dict[str, Any]) -> None:
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n")

    def log(self, message: str) -> None:
        print(message, flush=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    def write_summary(self, values: dict[str, Any]) -> None:
        status = values.get("result", "UNKNOWN")
        lines = [
            f"# Laufzusammenfassung: {status}",
            "",
            f"- Architektur: `{values.get('architecture', 'unbekannt')}`",
            f"- Parameter: {values.get('parameter_count', 'unbekannt')}",
            f"- Finaler Loss: {values.get('final_loss', 'nicht gemessen')}",
            f"- Bester Loss: {values.get('best_loss', 'nicht gemessen')}",
            f"- Durchsatz: {values.get('tokens_per_second', 'nicht gemessen')} Token/s",
            f"- Peak-Speicher: {values.get('peak_memory_bytes', 'nicht verfügbar')} Byte",
            f"- NaN/Inf: {values.get('nan_inf', 'nicht erkannt')}",
            "- Memory-System: in Milestone 1 nicht aktiviert",
            "- Expert-Router: in Milestone 1 nicht aktiviert",
            "",
            "## Zustandsstatistik",
            "",
            f"{values.get('state_statistics', 'Siehe metrics.jsonl.')}",
            "",
            "## Hinweise",
            "",
            f"{values.get('warnings', 'Keine.')}",
            "",
        ]
        (self.path / "summary.md").write_text("\n".join(lines), encoding="utf-8")

