# Benchmarks

`scripts/benchmark.py` schreibt maschinenlesbare JSONL-Dateien in dieses Verzeichnis. `smoke.jsonl` ist ein kurzer Funktions- und Overhead-Nachweis auf der bei der Entwicklung verfügbaren Hardware; er ist kein belastbarer Architekturvergleich.

## Vorhandene Messreihen

| Datei | Inhalt |
|---|---|
| `milestone1-baseline.jsonl` | Ausgangswert vor der State-Intelligence-Erweiterung, unoptimierter Kern |
| `milestone2.jsonl` | Bounded State-Interaktion, unoptimierter Kern – die gemessene Regression |
| `milestone2_5.jsonl` | Bounded State-Interaktion, optimierter Kern aus Milestone 2.5 |
| `milestone2_5-plain-core.jsonl` | Milestone-1-Pfad (`state_interactions=false`) mit demselben optimierten Kern |

Alle vier Reihen verwenden dieselbe Breite (`d_model=64`), Blockzahl (2), Batchgröße (1), Sequenzlängen (64, 128), Hardware, Precision (BFloat16) und Wiederholungszahl (3). Die Milestone-2- und Milestone-2.5-Zeilen enthalten zusätzlich `baseline_tokens_per_second` und `throughput_change_vs_baseline_percent` gegenüber `milestone1-baseline.jsonl`.

## Reproduktion

```bash
# Milestone 2.5, gebundener State-Interaction-Pfad
python scripts/benchmark.py --d-model 64 --layers 2 --batch-size 1 \
  --lengths 64 128 --iterations 3 --state-interactions \
  --baseline benchmarks/milestone1-baseline.jsonl \
  --output benchmarks/milestone2_5.jsonl

# Derselbe optimierte Kern ohne State-Interaktion, zur Trennung der Kostenanteile
python scripts/benchmark.py --d-model 64 --layers 2 --batch-size 1 \
  --lengths 64 128 --iterations 3 \
  --baseline benchmarks/milestone1-baseline.jsonl \
  --output benchmarks/milestone2_5-plain-core.jsonl
```

## Aufgezeichnete Felder

Jede Zeile hält Modellkonfiguration, Parameterzahl, Hardware, Backend, PyTorch-Version, Precision, Batchgröße, Sequenzlänge, Operation, Compile-Status, Token/s, Beobachtungs-Overhead, Streaming-Latenz, Peak-Gerätespeicher (`peak_memory_bytes`, sofern das Backend einen echten Zähler anbietet) und den maximalen residenten Hauptspeicher des Prozesses (`host_peak_rss_bytes`) fest.

## Einordnung

Kurze Smoke-Benchmarks sind stark aufwärm- und systemlastabhängig. Sie belegen Funktionsfähigkeit, nicht allgemeine Effizienz. Belastbare Architekturvergleiche benötigen längere Läufe, mehrere Wiederholungen und später eine dokumentierte GRU/LSTM-Referenz.

`torch.compile` ist optional. `--compile` fällt mit einem deutschen Hinweis auf den Eager-Pfad zurück, wenn der Backend-Compiler in der Umgebung nicht übersetzen kann; `--require-compile` erzwingt stattdessen einen Abbruch. GlassMind benötigt `torch.compile` in keinem Pfad.

## Milestone 2.6: Precision und Quantisierung

| Datei | Inhalt |
|---|---|
| `milestone2_5-reference.json` | eingefrorene Milestone-2.5-Referenz: Durchsatz, Latenz, Speicher, Aufgaben, Ablationen, Zustandsrollen, Logits |
| `milestone2_6-precision.json` | vollständige Precision-Matrix: Gleitkommavarianten, State-dtype-Matrix, Quantisierung, Langzeitdrift, Telemetrievergleich |
| `microbench-rtx3070.json` | Microbenchmark GlassMind-typischer Operationen samt `auto`-Empfehlung |
| `milestone2_6-baseline-check.jsonl` | Nachweis, dass der Milestone-2.5-Durchsatz nach den Precision-Umbauten erhalten ist |
| `milestone2_6-long-distance.json` | Recall und Selective Copy bei Distanz 1024 und 4096 mit 128 Beispielen je Zelle |

Reproduktion:

```bash
# 1. Referenz einfrieren (einmalig)
python scripts/precision_baseline.py \
  --checkpoint runs/<state-lauf>/checkpoints/final.pt

# 2. Hardware vermessen und auto-Profil ableiten
python scripts/microbench.py --output benchmarks/microbench-rtx3070.json

# 3. Vollständige Precision-Matrix
python scripts/precision_matrix.py \
  --checkpoint runs/<state-lauf>/checkpoints/final.pt \
  --output benchmarks/milestone2_6-precision.json

# Einzelne Abschnitte nachziehen, ohne die übrigen zu verlieren
python scripts/precision_matrix.py --checkpoint <…> \
  --sections quantization --merge

# 4. Tabellen erzeugen (Text oder Markdown)
python scripts/precision_report.py --markdown
```

Die Messwerte gelten für eine RTX 3070 mit `d_model=64` und zwei Blöcken. Bei anderen Modellgrößen und anderer Hardware verschieben sich die Verhältnisse; `scripts/microbench.py` misst sie neu, statt sie zu übernehmen.

## Milestone 3: Sparse External Memory

| Datei | Inhalt |
|---|---|
| `milestone3-memory.json` | Nutzenvergleich ohne/mit Memory, Ablationen, Query-Quellen, Kapazität, Replacement-Policies |
| `milestone3-cost.json` | Kosten des Speichers: Durchsatz, Latenz, Trainingsdurchsatz, VRAM, Operationen je Token, Bandbreite |

Reproduktion:

```bash
# Nutzen: dieselbe Architektur ohne und mit Speicher, gleiche Trainingsschritte
python scripts/memory_study.py --steps 900 --sections compare ablation

# Varianten nachziehen, ohne die übrigen Abschnitte zu verlieren
python scripts/memory_study.py --steps 400 --sections query capacity policy --merge

# Kosten inklusive Regressionsnachweis gegen Milestone 2.6
python scripts/memory_bench.py
```

Die Kostenmessung enthält als erste Zeile ausdrücklich den Pfad *ohne* Memory und als zweite den Pfad mit konfiguriertem, aber abgeschaltetem Memory. Beide müssen praktisch gleich schnell sein; andernfalls hätte Milestone 3 die Milestone-2.6-Baseline beschädigt.
