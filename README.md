# GlassMind

GlassMind ist ein experimentelles, **nicht auf Transformern basierendes** kausales Sprachmodell. Milestone 2 „State Intelligence“ ergänzte messbaren assoziativen Abruf, selektives Kopieren, State-Ablationen und reproduzierbare Aktivitätscluster. Milestone 2.5 „Performance + Context Specialization“ macht denselben Kern profilierbasiert deutlich schneller und untersucht ehrlich, ob `context_state` auf Aufgaben mit mittlerer zeitlicher Struktur kausal relevant wird. Milestone 2.6 „Numerical Efficiency + Quantization Foundation“ misst systematisch, welche Zahlendarstellung der Kern tatsächlich braucht, und legt eine portable Quantisierungsschicht darunter. Der begrenzte rekurrente Zustand kann während Inferenz, Aufzeichnung und Replay als reales Netzwerk beobachtet werden.

Der aktuelle Kern verwendet weder Self-Attention noch eine quadratische Token-zu-Token-Matrix und besitzt keinen Attention-Cache. Externes Memory und Sparse Experts sind bewusst noch nicht enthalten. Alle Aufgaben werden ausschließlich mit fest begrenzten internen Zuständen gelöst.

## Architektur

```text
Token-ID
  → Token-Embedding
  → kausaler lokaler Tiefenfilter (fester Puffer)
  → selektiv-rekurrenter Block × L
       sequenzweiter Vorlauf   LayerNorm, Eingangsprojektion, Gates, Ausgangs-Gate
       fast_state      kurze, schnell aktualisierte Dynamik
       context_state   mittlere Zeitskala
       semantic_state  langsame Zeitskala und optional gebundene State-Interaktion
       gelernter Integrator + Residualpfad
  → LayerNorm
  → geteilter LM-Head
  → nächstes Token
```

Jeder Zustand besitzt ein gelerntes Gate und eine eigene Initial-Zeitskala. `fast_state` beeinflusst `context_state`, `context_state` beeinflusst im Referenzpfad `semantic_state`, und alle drei Zustände tragen über getrennte Gewichtsteile zum Blockausgang bei. Der rekurrente Zustand hat pro Batch ungefähr `3 × L × d_model` Werte; der lokale Mixer hält zusätzlich nur `kernel_size - 1` Token. Er wächst nicht mit der Kontextlänge.

Der Block ist seit Milestone 2.5 in zwei Teile getrennt:

- Ein **sequenzweiter Vorlauf** berechnet alles, was nur von der Eingabe abhängt (LayerNorm, Eingangsprojektion, alle Gates, Ausgangs-Gate) einmal für die ganze Sequenz statt einmal pro Token.
- Der **rekurrente Tokenpfad** enthält nur noch die Rechnungen, die wirklich vom vorherigen Zustand abhängen. Projektionen mit gemeinsamer Eingabe sind dort zu je einer Matrix zusammengefasst.

Beide Teile sind mathematisch identisch zur ursprünglichen entbündelten Formulierung. `SelectiveStateBlock.reference_forward` hält den langsamen, unfusionierten Referenzpfad bereit; `tests/test_optimized_core.py` vergleicht beide für jeden Pfad und jede Ablation.

Für die gebundene State-Interaktion (`state_interactions=true`) bildet der Kern gelernte, niedrigdimensionale Schlüssel-/Wert-Codes und akkumuliert deren äußeres Produkt direkt im begrenzten `semantic_state`. Eine Abfrage liest denselben bounded State mit einem gelernten Schlüsselcode. Das ist kein externer Slot-Speicher: Es gibt keine wachsende Tabelle, keine Top-K-Suche und keinen Token-Cache. Der Milestone-1-Pfad bleibt mit `state_interactions=false` erhalten.

Bei fester Modellbreite und Blockzahl benötigt eine Sequenz der Länge `n` `O(n)` rekurrente Schritte und `O(1)` Zustand bezüglich `n`. Die linearen Projektionen kosten pro Schritt ungefähr `O(L × d_model²)`.

## Installation

PyTorch unterstützt in der verwendeten Umgebung Python 3.14 noch nicht zuverlässig. Verwende Python 3.10 bis 3.13:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev,visualize]'
```

Die Visualisierung ist optional. Ohne GUI genügt `pip install -e '.[dev]'`.

## Geräte und Precision

Der gesamte Modellpfad besteht aus portablen PyTorch-Operationen. Eine zentrale Laufzeiterkennung unterstützt, sofern der installierte PyTorch-Build das Backend bereitstellt:

- NVIDIA über CUDA,
- AMD über ROCm,
- Apple Silicon über MPS,
- Intel-GPUs über XPU,
- CPU als verpflichtenden Fallback.

Im Modellcode gibt es kein fest verdrahtetes `.cuda()`. AMP und BFloat16/Float16 werden nach Fähigkeiten gewählt; CPU verwendet standardmäßig Float32. Der rekurrente Tokenpfad rechnet bewusst in der Präzision des Zustands und nicht unter Autocast – das ist in jeder gemessenen Konfiguration schneller und für die Langzeit-Bindung genauer (siehe unten). Die sequenzweite Eingangsprojektion nutzt Autocast weiterhin. `ModelConfig.recurrent_autocast=true` stellt das alte Verhalten wieder her; es ist ausschließlich für Reproduktionsmessungen gedacht.

Seit Milestone 2.6 steuert eine zentrale `PrecisionPolicy`, in welcher Darstellung gerechnet, gespeichert und quantisiert wird – getrennt für Gewichte, Aktivierungen und jeden der drei Zustände. Die Profile heißen `safe`, `balanced`, `fast`, `experimental` und `auto`; `auto` entscheidet anhand eines Microbenchmarks auf der vorliegenden Hardware und nicht anhand von Annahmen. Details und Messwerte stehen unter „Milestone 2.6".

```bash
python scripts/device_info.py
python scripts/device_info.py --device cpu --json
```

Checkpoints normalisieren Modell-, Optimizer- und Scheduler-Tensoren vor dem Speichern auf CPU und können anschließend auf jedem verfügbaren Backend geladen werden. Es gibt keine herstellerspezifischen Custom-Kernel oder Fast Paths.

### Checkpoint-Format

Das Format steht auf Version 3 und speichert zusätzlich zur Architektur die vollständige Zahlendarstellung: Gewichts-, Rechen- und Aktivierungsdatentyp, die dtypes von `fast_state`, `context_state` und `semantic_state`, die Quantisierungsart samt Gruppengröße und Skalen sowie Backend-Metadaten.

Ältere Formate werden weiterhin geladen. Format 1 wird über `migrate_block_parameters` in die fusionierten Matrizen aus Milestone 2.5 überführt; dabei bleiben alle Gewichtswerte exakt erhalten, und verworfen werden nur Parameter, die im gebundenen Pfad nie gelesen wurden. Der Referenzlauf aus Milestone 2 reproduziert nach der Migration seine Ablationszahlen bis auf die dritte Nachkommastelle. Ein nicht quantisierter Checkpoint bleibt backendunabhängig ladbar; ein quantisierter lässt sich mit `load_checkpoint(..., dequantize=True)` als dichtes Modell öffnen.

## Schnellstart

```bash
python scripts/smoke_test.py
python scripts/overfit_test.py
python scripts/train_tiny.py --steps 200
python scripts/infer.py 'Mira sieht ' --checkpoint runs/<lauf>/checkpoints/final.pt
```

Ein Trainingscheckpoint kann auf demselben oder einem anderen verfügbaren Backend fortgesetzt werden:

```bash
python scripts/train_tiny.py --resume runs/<lauf>/checkpoints/final.pt --steps 100
```

Ohne `--checkpoint` startet `infer.py` ein deterministisch initialisiertes, aber untrainiertes Mini-Modell und kennzeichnet die Ausgabe ausdrücklich.

Jeder Trainingslauf erzeugt einen eindeutigen Ordner:

```text
runs/<lauf>/
├── config.json
├── environment.json
├── metrics.jsonl
├── train.log
├── checkpoints/final.pt
├── traces/
└── summary.md
```

Konfiguration, Seed, Python-/PyTorch-Version, Git-Commit (wenn vorhanden), Hardware, Backend, Dataset- und Tokenizer-Metadaten werden gespeichert.

## Milestone 2: State Intelligence

Der vollständige reproduzierbare Lauf trainiert abwechselnd assoziativen Abruf und selektives Kopieren:

```bash
python scripts/state_intelligence.py
python scripts/state_intelligence.py --steps 100 --minimum-accuracy 0   # kurzer Funktionslauf
```

Referenzlauf mit Seed 41, zwei Blöcken und BFloat16 auf einer RTX 3070, jeweils auf neu erzeugten Beispielen. Die Spalte „Milestone 2“ ist der Lauf vor der Optimierung, „Milestone 2.5“ derselbe Preset mit dem optimierten Kern:

| Aufgabe | Metrik | Abstand 16 | Abstand 64 | Abstand 256 | Abstand 1024 |
|---|---|---:|---:|---:|---:|
| Assoziativer Abruf | M2 Accuracy | 100,0 % | 98,4 % | 96,9 % | 82,8 % |
| Assoziativer Abruf | **M2.5 Accuracy** | 100,0 % | 100,0 % | 98,4 % | **90,6 %** |
| Selektives Kopieren | M2 Accuracy | 100,0 % | 100,0 % | 100,0 % | 98,4 % |
| Selektives Kopieren | **M2.5 Accuracy** | 99,2 % | 100,0 % | 100,0 % | **100,0 %** |

Der Zugewinn bei großem Abstand ist kein Architekturzuwachs, sondern eine direkte Folge der entfernten Präzisionswechsel: Die Bindungsakkumulation über 1024 Schritte läuft jetzt durchgängig in Zustandspräzision statt in BFloat16.

Die Ablation bei Abstand 64 verändert jeweils nur die Inferenz; der Trainingspfad bleibt unberührt:

| Aufgabe | deaktivierter State | Δ Loss | Δ Accuracy | Logit-RMS | geänderte Vorhersagen |
|---|---|---:|---:|---:|---:|
| Assoziativer Abruf | `fast` | +2,8660 | −65,6 % | 2,7474 | 65,6 % |
| Assoziativer Abruf | `context` | +0,0294 | −1,6 % | 0,9674 | 1,6 % |
| Assoziativer Abruf | `semantic` | +2,7020 | −73,4 % | 2,6547 | 73,4 % |
| Selektives Kopieren | `fast` | +3,8734 | −67,2 % | 3,2232 | 67,2 % |
| Selektives Kopieren | `context` | +0,0211 | −1,6 % | 0,7696 | 1,6 % |
| Selektives Kopieren | `semantic` | +0,6146 | −19,5 % | 2,0870 | 19,5 % |

`fast_state` ist für die unmittelbare Kodierung und Ausgabe kritisch, `semantic_state` für die gebundene Langzeitinformation. Das ist gemessene Spezialisierung, keine Behauptung allgemeiner semantischer Rollen.

Die State-Ablation steht auch in der Streaming-Inferenz zur Verfügung und kann mehrfach angegeben werden:

```bash
python scripts/infer.py 'STORE KEY_0 VALUE_1 QUERY KEY_0 ANSWER' \
  --checkpoint runs/<state-lauf>/checkpoints/final.pt \
  --max-new-tokens 1 --ablate-state semantic --trace
```

## Milestone 2.5: Performance

### Gemessene Ursache der Regression

Milestone 2 verlor gegenüber Milestone 1 rund 33–34 % Durchsatz im `off`-Modus. Die Vermutung „die zusätzlichen Projektionen kosten Rechenzeit“ war falsch. Das Profil zeigt:

| Messung | Milestone 1 | Milestone 2 |
|---|---:|---:|
| ATen-Operationen pro Block-Schritt | 63,1 | 87,1 |
| Token/s (`off`, Länge 64, RTX 3070) | 898 | 659 |
| Verhältnis | 1,00 | 0,73 |

Das Verhältnis der Durchsätze (0,73) entspricht fast exakt dem Kehrwert des Verhältnisses der Operationszahlen (63,1/87,1 = 0,72). Der Kern war also nicht rechen-, sondern **dispatch- und launch-gebunden**. `torch.profiler` bestätigte es direkt:

- Von 188 ms Wanduhr entfielen **157 ms auf CPU-Zeit und nur 37,9 ms auf GPU-Zeit** – die GPU war zu rund 80 % unbeschäftigt.
- 13 743 `cudaLaunchKernel`-Aufrufe für fünf Forwards à 64 Token, also **43 Kernel-Starts pro Token**.
- Die häufigste Einzeloperation war `aten.to.dtype` mit **489 von 1394 Operationen (35 %)**: reine Autocast-Präzisionswechsel.
- 1010 der 1394 Operationen entfielen auf zehn Linear-Aufrufe pro Block-Schritt sowie deren Ein-/Ausgangscasts.
- 48 `zeros_like`-Aufrufe je acht Token waren Platzhalter, die nur die Telemetrie liest.

Die entscheidende Einzelmessung: **AMP/BFloat16 war netto ein Verlust.** Derselbe Kern lief ohne Autocast mit 1058 statt 663 Token/s. Bei `d_model=64` sind die eingesparten Matmul-Zyklen kleiner als der Preis der Cast-Kernel.

### Umgesetzte Optimierungen

Alle Änderungen sind Umformungen derselben Rechnung, keine Näherungen:

1. **Sequenzweiter Vorlauf.** LayerNorm, Eingangsprojektion, die drei Gate-Sigmoids und das Ausgangs-Gate hängen nur von der Eingabe ab. Sie laufen jetzt einmal für die Sequenz statt einmal pro Token.
2. **Eingangsprojektion und Ausgangs-Gate fusioniert.** Beide lesen dieselbe normalisierte Eingabe; eine Matrix statt zwei.
3. **Zustandsprojektionen mit gemeinsamer Eingabe fusioniert.** `fast_recurrent`, `key_projection` und der erste Teil des Schreib-Gates lesen alle den vorherigen `fast_state`; `context_from_fast`, `value_projection` und der zweite Teil des Schreib-Gates lesen alle den neuen `fast_state`. Aus sechs Matmuls werden zwei. Damit fallen im gebundenen Pfad **vier statt zehn Linear-Aufrufe pro Block-Schritt** an – weniger als im Milestone-1-Pfad mit sechs.
4. **Lesecode in den Integrator gezogen.** `integrator(cat(fast, context, semantic)) + read_to_output(read)` ist eine einzige Matrix über `cat(fast, context, semantic, read)`.
5. **Autocast aus dem Tokenpfad entfernt.** Der rekurrente Teil rechnet in Zustandspräzision. Die `aten.to.dtype`-Aufrufe fielen von 489 auf 19 pro acht Token.
6. **Tote Parameter entfernt.** Im gebundenen Pfad wurden `semantic_from_context`, `semantic_recurrent`, `semantic_bias` und der semantische Zweig der Eingangsprojektion nie gelesen; der `semantic_state` führte jenseits der Bindungsbreite dauerhaft Nullen mit. Das waren 21,6 % der Parameter und ein Drittel der Ausgangsbreite der größten Projektion.
7. **Residual und Dropout aus dem Loop gezogen**, `zeros_like`-Platzhalter nur noch unter aktiver Telemetrie erzeugt, Ablationsnullen einmal statt pro Token alloziert.
8. **Telemetrie-Transfers gebündelt.** Der `trace`-Modus setzte pro Token und Block über hundert einzelne `.item()`-Aufrufe ab – jeder ein eigener CPU/GPU-Synchronisationspunkt. Jetzt bleiben alle Kennzahlen bis zu **einem** Transfer je Block auf dem Gerät, und die Cluster-Reduktionen laufen bei teilbarer Breite vektorisiert statt einzeln je Cluster.

Nach Punkt 5 ist AMP nicht mehr schädlich: mit 2103 gegenüber 2120 Token/s ohne AMP liegt der Unterschied im Messrauschen, und bei `d_model=256`, vier Blöcken und Batch 8 bringt AMP wieder +6,5 %. `recurrent_autocast=true` ist in jeder gemessenen Konfiguration schlechter (bei `d_model=256`: 6889 statt 13 254 Token/s) und bleibt nur als Reproduktionsschalter erhalten.

### Ergebnis

Gleiche Hardware (RTX 3070), gleiche Precision (BFloat16), `d_model=64`, zwei Blöcke, Batch 1, drei Wiederholungen, ohne `torch.compile`. Rohdaten in `benchmarks/milestone2_5.jsonl`:

| Länge | Modus | M1 Token/s | M2 Token/s | **M2.5 Token/s** | gegen M2 | gegen M1 |
|---:|---|---:|---:|---:|---:|---:|
| 64 | `off` | 977 | 650 | **2 243** | +245 % | **+130 %** |
| 64 | `summary` | 961 | 658 | **2 109** | +221 % | +120 % |
| 64 | `trace` | 57 | 52 | **342** | +554 % | +501 % |
| 128 | `off` | 990 | 650 | **2 413** | +271 % | **+144 %** |
| 128 | `summary` | 960 | 620 | **2 329** | +276 % | +143 % |
| 128 | `trace` | 54 | 52 | **318** | +507 % | +484 % |

| Kennzahl | Milestone 1 | Milestone 2 | Milestone 2.5 |
|---|---:|---:|---:|
| Streaming-Latenz (Median aus 5 × 128 Schritten) | 1,469 ms/Token | 1,833 ms/Token | **1,127 ms/Token** |
| Trainingsdurchsatz (State-Intelligence-Preset) | – | 6 473 Token/s | **16 744 Token/s** |
| Parameter (`d_model=64`, 2 Blöcke, Vokabular 260) | 149 824 | 153 154 | **120 002** |
| ATen-Operationen je Block-Schritt (`off`) | 63,1 | 87,1 | **40,5** |
| Präzisionswechsel je Forward (8 Token, 2 Blöcke) | – | 489 | **19** |
| GPU-Anteil an der Wanduhr (`off`, Länge 128) | – | 20,1 % | **21,8 %** |
| Beobachtungs-Overhead `trace` gegen `off` | 94,2 % | 92,0 % | **86,8 %** |
| Peak-VRAM `off`, Länge 128 | 9 346 560 B | 9 360 896 B | 9 446 912 B |
| Peak-VRAM `trace`, Länge 128 | 11 531 264 B | 11 676 160 B | **11 468 800 B** |

Die 33–34 % Regression im `off`-Modus sind damit nicht nur zurückgewonnen, sondern der Kern liegt deutlich über dem Milestone-1-Ausgangswert. Der einzige gemessene Nachteil ist +0,9 % Peak-VRAM im `off`-Modus bei Länge 128: Der sequenzweite Vorlauf materialisiert die Eingangsprojektion für die ganze Sequenz statt Token für Token. Im `trace`-Modus sinkt der Peak, weil die gebündelten Transfers weniger Zwischentensoren erzeugen.

### Was die Optimierung nicht ändert

Der gebundene State-Interaction-Pfad bleibt gegenüber demselben Kern ohne Interaktion teurer. `benchmarks/milestone2_5-plain-core.jsonl` misst genau das:

| Länge | Modus | mit Interaktion | ohne Interaktion | Aufschlag |
|---:|---|---:|---:|---:|
| 64 | `off` | 2 243 | 3 264 | −31,3 % |
| 128 | `off` | 2 413 | 3 333 | −27,6 % |
| – | Streaming | 1,127 ms | 1,031 ms | +9,3 % |

Der *relative* Aufschlag ist also nur wenig kleiner als die ursprünglichen 33–34 %; verändert hat sich das absolute Niveau beider Pfade. Die Kostenstelle ist real und benannt: 40,5 statt 25,0 Operationen je Block-Schritt für Schlüsselcode, Wertcode, Normalisierung, äußeres Produkt, Schreib-Gate und Lesepfad. Sie wird nicht versteckt.

Ebenfalls unverändert: Der Kern ist weiterhin dispatch-gebunden, nicht rechen-gebunden. Der GPU-Anteil an der Wanduhr stieg nur von 20,1 % auf 21,8 % – der Durchsatz stieg, weil weniger Arbeit pro Token dispatcht wird, nicht weil die GPU besser ausgelastet wäre. Ein weiterer großer Sprung bräuchte eine zeitlich parallelisierte Rekurrenz oder einen funktionierenden Compile-Pfad.

### `torch.compile`

`torch.compile` wurde untersucht, ist in dieser Umgebung aber nicht nutzbar: Der Inductor-Backend benötigt CPython-Header (`Python.h`), die hier fehlen. GlassMind funktioniert in jedem Pfad ohne `torch.compile`. `scripts/benchmark.py --compile` fällt mit einem deutschen Hinweis auf den Eager-Pfad zurück; `--require-compile` erzwingt stattdessen einen Abbruch.

Ein verwandter Portabilitätsbefund: `torch.bmm` hätte im Bindungs-Lesepfad einen Kernel-Start gespart, führt seinen Rückwärtspfad in aktuellen Builds aber über einen Triton-Kernel, der ohne dieselben Header nicht baubar ist. Auf CPU fiel das nicht auf, im CUDA-Training brach es ab. Der Lesepfad nutzt deshalb bewusst reine ATen-Operationen. `tests/test_optimized_core.py::test_training_step_runs_on_the_accelerator` prüft seitdem einen echten Trainingsschritt auf dem Beschleuniger.

## Milestone 2.5: Context Specialization

In Milestone 2 hatte `context_state` eine messbar langsamere Dynamik, aber kaum kausale Wirkung: Seine Ablation kostete auf assoziativem Abruf und selektivem Kopieren 0,0 bis 0,8 % Accuracy. Beide Aufgaben brauchen keine mittlere Zeitskala – ein Schlüssel bestimmt seinen Wert eindeutig, und die Zuordnung gilt für die ganze Sequenz.

Milestone 2.5 ergänzt drei Aufgaben, bei denen eine mittlere Zeitskala nützlich *sein kann*:

| Aufgabe | Struktur |
|---|---|
| `sectioned_recall` | Mehrere Abschnitte, gleiche Schlüssel, pro Abschnitt andere Werte. Die Frage nennt Abschnitt und Schlüssel. |
| `topic_resumption` | Abschnitt A, Unterbrechung durch B, Wiederaufnahme von A. Gefragt wird über die Unterbrechung hinweg. |
| `hierarchical_scope` | Eine dokumentweite Konstante neben abschnittslokalen Fakten mit demselben Schlüssel. |

Ein Beispiel aus `sectioned_recall`:

```text
BOS
SECTION_3  STORE KEY_5 VALUE_7   STORE KEY_7 VALUE_1   NOISE …
SECTION_1  STORE KEY_5 VALUE_13  STORE KEY_7 VALUE_5   NOISE …
SECTION_0  STORE KEY_5 VALUE_10  STORE KEY_7 VALUE_12  NOISE …
QUERY SECTION_1 KEY_5 ANSWER → VALUE_13
```

Eine rein globale Schlüssel-Wert-Bindung kollidiert hier zwangsläufig: `KEY_5` trägt drei verschiedene Werte. Das Vokabular bleibt abwärtskompatibel – `section_count=0` liefert unverändert die 64 Token aus Milestone 2.

```bash
python scripts/context_specialization.py                  # gemischt mit den Milestone-2-Aufgaben
python scripts/context_specialization.py --only-context   # nur die drei neuen Aufgaben
```

### Ergebnis

Lauf mit Seed 41, 2400 Schritten, allen fünf Aufgaben, `d_model=64`, zwei Blöcken:

| Aufgabe | Distanz 16 | Distanz 64 | Distanz 256 |
|---|---:|---:|---:|
| `associative_recall` | 100,0 % | 100,0 % | 95,3 % |
| `sectioned_recall` | 92,2 % | 96,9 % | 89,1 % |
| `selective_copy` | 100,0 % | 100,0 % | 100,0 % |
| `topic_resumption` | 100,0 % | 100,0 % | 100,0 % |
| `hierarchical_scope` | 100,0 % | 100,0 % | 93,8 % |

State-Ablation bei Distanz 64:

| Aufgabe | `fast` Δ Acc | `context` Δ Acc | `semantic` Δ Acc | `context` Δ Loss |
|---|---:|---:|---:|---:|
| `associative_recall` | −70,3 % | −3,1 % | −64,1 % | +0,0567 |
| `sectioned_recall` | −78,1 % | **−7,8 %** | −76,6 % | **+0,2642** |
| `selective_copy` | −84,4 % | ±0,0 % | −84,4 % | +0,0023 |
| `topic_resumption` | −75,0 % | −1,6 % | −79,7 % | +0,0248 |
| `hierarchical_scope` | −76,6 % | −3,1 % | −78,1 % | +0,0816 |

### Der Gegenlauf: nur die Kontextaufgaben

Derselbe Preset, aber ohne die beiden Milestone-2-Aufgaben im Trainingsmix (`--only-context`), liefert ein auffällig anderes Bild:

| Aufgabe | Accuracy (Distanz 64) | `fast` Δ Loss | `context` Δ Loss | `semantic` Δ Loss |
|---|---:|---:|---:|---:|
| `sectioned_recall` | 26,6 % | +0,6765 | **+1,8971** | +0,1606 |
| `topic_resumption` | 37,5 % | +1,8064 | **+2,6372** | +0,2401 |
| `hierarchical_scope` | 26,6 % | +1,5175 | **+2,8773** | +0,0824 |

Hier ist `context_state` in allen drei Aufgaben der **wichtigste** Zustand (im Mittel −19,3 % Accuracy gegen −9,1 % für die beiden anderen), und `semantic_state` wird nebensächlich. Diese Zahlen sind jedoch ausdrücklich **nicht** als Erfolgsmeldung zu lesen: Das Modell löst die Aufgaben in diesem Lauf nur zu 32,1 % im Mittel (20–47 % je nach Aufgabe und Distanz). Eine Ablation an einem Modell, das die Aufgabe nicht beherrscht, zeigt bestenfalls, worauf sein unvollständiges Verfahren beruht – nicht, welcher Zustand für eine Lösung nötig wäre. `scripts/context_specialization.py` hängt diese Einschränkung deshalb automatisch an den Befund an, sobald die mittlere Accuracy der Kontextaufgaben unter 75 % liegt.

### Befund

`context_state` wird auf Aufgaben mit expliziter Abschnittsstruktur **messbar relevanter, bleibt aber kausal deutlich untergeordnet, sobald das Modell die Aufgaben tatsächlich löst.**

Konkret: Auf `sectioned_recall` – der Aufgabe mit der stärksten Abschnittskollision – steigt der Loss-Effekt seiner Ablation von +0,0103 (Milestone 2, assoziativer Abruf) auf +0,2642, also um etwa Faktor 25, und der Accuracy-Verlust von 0,0 % auf 7,8 %. Gleichzeitig kosten `fast` und `semantic` auf derselben Aufgabe 78,1 % beziehungsweise 76,6 % Accuracy. Über alle drei neuen Aufgaben gemittelt liegt `context` bei 4,2 % gegenüber 77,3 % für die beiden anderen Zustände.

Die gemessene Zeitskala trennt die Zustände dabei klar: mittlere geschätzte Zeitkonstante 2,14 Token für `fast`, 13,90 für `context`, 12,88 für `semantic`. `context_state` ist also eine reale langsame Dynamik, die den Ausgang beeinflusst – nur wird die Abschnittsauflösung überwiegend anders gelöst, nämlich über den Schlüsselcode der gebundenen Interaktion, der die Abschnittsmarke mitkodieren kann.

Der Vergleich beider Läufe erlaubt eine schärfere Aussage als jeder einzelne: **Die Rollenverteilung der Zustände ist keine feste Eigenschaft der Aufgabe, sondern eine Eigenschaft des Trainingsmixes.** Wird die gebundene Interaktion durch einfachere Aufgaben eingeübt, löst sie anschließend auch die Abschnittsaufgaben, und `context_state` bleibt Nebensache. Wird sie nicht eingeübt, verlagert sich die Last auf `context_state` – aber die Aufgaben werden dann deutlich schlechter gelöst. Die Abschnittsstruktur macht eine mittlere Zeitskala also *möglich*, nicht *notwendig*.

**Die Architektur wurde deshalb nicht verändert.** Es wäre einfach gewesen, `context_state` durch eine erzwungene Kopplung künstlich unverzichtbar zu machen; das hätte bessere Ablationszahlen und keinen Erkenntnisgewinn gebracht. Der offene Punkt bleibt: Entweder braucht es eine Aufgabe, die eine mittlere Zeitskala wirklich erzwingt und die das Modell trotzdem löst, oder der dreistufige Zustand ist an dieser Stelle um eine Stufe zu breit angelegt. Beides ist ohne weitere Messungen nicht entschieden.

## Milestone 2.6: Numerical Efficiency + Quantization Foundation

Milestone 2.6 beantwortet eine Frage, die bis dahin nur vermutet war: **Welche Zahlendarstellung braucht GlassMind wirklich?** Alle Aussagen unten sind Messwerte auf einer RTX 3070 mit `d_model=64`, zwei Blöcken und dem trainierten Milestone-2-Checkpoint. Rohdaten liegen in `benchmarks/milestone2_6-precision.json`, die eingefrorene Vergleichsgrundlage in `benchmarks/milestone2_5-reference.json`.

### Zentrale Bausteine

Die gesamte Precision- und Quantisierungslogik liegt in `glassmind/precision/`. Im Modellcode steht keine verstreute Sonderbehandlung; der Kern fragt beim Rechnen nur über `linear_weight` nach seiner Gewichtsmatrix – genau einmal je Block und Sequenz, nie im Tokenpfad.

| Modul | Aufgabe |
|---|---|
| `policy.py` | `PrecisionPolicy` und die Profile `safe`, `balanced`, `fast`, `experimental` |
| `quantization.py` | portables INT8/INT4/FP8 als Weight-Only, Straight-Through-Estimator für späteres QAT |
| `apply.py` | die einzige Stelle, die Modulgruppen auf konkrete Module abbildet |
| `microbench.py` | Hardwaremessung mit GlassMind-Formen und das daraus abgeleitete `auto`-Profil |
| `drift.py` | Langzeitdrift gegen eine FP32-Referenz, je Zustand getrennt |
| `compare.py` | Abweichung der sichtbaren Aktivität über den Observation Bus |
| `reference.py` | die vollständige Kennzahlenerhebung, die für Referenz und Prüfling identisch läuft |

Eine Policy mit ausschließlich `inherit` und ohne Quantisierung ist **bitgleich** zur Milestone-2.5-Baseline. `tests/test_precision.py` prüft das für beide Architekturpfade.

```python
from glassmind.precision import PrecisionPolicy, apply_precision

policy = PrecisionPolicy(
    compute="bfloat16",
    fast_state="bfloat16",
    context_state="bfloat16",
    semantic_state="float32",       # der einzige Zustand mit echter Langzeitdrift
    weights={"embedding": "int8", "input_projection": "int4", "lm_head": "int8"},
)
apply_precision(model, policy)
```

### 1. Welche Precision ist am schnellsten?

**Keine – jedenfalls nicht bei dieser Modellgröße.** Über acht Gleitkommavarianten liegt die Spanne bei 1 666 bis 2 360 Token/s, und der Abstand zwischen den Formaten bleibt in der Größenordnung der Messstreuung.

| Konfiguration | Token/s | ms/Token | Training Tok/s | Gewichte | Logit-Drift |
|---|---:|---:|---:|---:|---:|
| fp32 (Referenz) | 2 255 | 1,038 | 7 350 | 419,8 KiB | 0 |
| fp32 + AMP-bf16 (Milestone 2.5) | 2 195 | 1,182 | 6 201 | 419,8 KiB | 0,002 |
| **Gewichte+Compute bf16** | **2 360** | **0,994** | 7 284 | 209,9 KiB | 0,012 |
| Gewichte+Compute fp16 | 2 092 | 1,022 | 7 437 | 209,9 KiB | 0,002 |
| balanced (compute bf16, ctx+sem fp32) | 1 876 | 1,320 | 6 104 | 419,8 KiB | 0,012 |
| fast (compute bf16, sem fp32) | 1 855 | 1,317 | 6 514 | 419,8 KiB | 0,012 |

Der Microbenchmark erklärt, warum: Bei den Formen, die GlassMind tatsächlich ausführt – `[1×64]→73` für die Zustandsprojektion, `[1×192]→64` für den Integrator – liegen alle drei Formate bei 9 bis 16 µs, und das ist überwiegend Dispatch- und Launch-Zeit. Erst bei `[128×256]→1024` wird BF16 mit 8,93 gegenüber 14,05 µs **36 % schneller**. Reduzierte Precision hilft GlassMind also erst beim Hochskalieren, nicht heute.

Bemerkenswert ist die Gegenrichtung: Die gemischten Profile `balanced` und `fast` sind **langsamer** als sowohl FP32 als auch das durchgehende BF16-Modell. Sie halten FP32-Gewichte und rechnen in BF16, müssen die Gewichte also bei jedem Vorwärtslauf casten. Wer dauerhaft in BF16 rechnen will, konvertiert das Modell (`model.to(torch.bfloat16)`), statt den Rechendatentyp allein umzustellen.

### 2. Welche Precision brauchen `fast`, `context` und `semantic` wirklich?

Alle 27 Kombinationen aus fast/context/semantic × fp32/bf16/fp16 lösen die kurzen Aufgaben **identisch**: 100 % bei Recall 16/64/256 und bei allen Selective-Copy-Distanzen bis 256. Aufgabenqualität allein trennt die Formate dort also nicht.

Die Langzeitdrift trennt sie sehr wohl. Relative Abweichung gegen FP32 bei durchgehend BF16-Zuständen:

| Länge | `fast` | `context` | `semantic` | Logits | geänderte Vorhersagen |
|---:|---:|---:|---:|---:|---:|
| 64 | 0,006 | 0,008 | 0,016 | 0,004 | 0,0 % |
| 256 | 0,007 | 0,018 | 0,028 | 0,006 | 0,0 % |
| 1 024 | 0,007 | 0,026 | 0,084 | 0,011 | 0,0 % |
| 4 096 | 0,005 | 0,033 | **0,318** | 0,008 | 0,0 % |
| 8 192 | 0,008 | 0,030 | **0,410** | 0,008 | 0,0 % |

Das Muster folgt genau der Architektur:

- **`fast_state` akkumuliert nicht.** Sein Gate überschreibt ihn pro Token weitgehend; die Abweichung bleibt über 8 192 Token bei 0,5–0,8 %.
- **`context_state` akkumuliert und sättigt** bei rund 3 %.
- **`semantic_state` akkumuliert monoton** von 1,6 % auf 41 %. Er ist der Bindungsakkumulator: Jeder Schreibvorgang addiert auf den bestehenden Zustand, und der Rundungsfehler bleibt drin.

Eine Beobachtung, die zunächst überrascht: `balanced` (BF16-Rechnung, FP32-Speicher für `context`/`semantic`) und „alle Zustände BF16" (FP32-Rechnung, BF16-Speicher) driften **exakt gleich**. In beiden Fällen wird pro Token genau einmal auf BF16 gerundet; ob beim Rechnen oder beim Speichern, ändert nichts. Einen Zustand höher aufzulösen als den Rechenpfad bringt hier also nichts.

### 2b. Der unerwartete Befund: Drift ist nicht gleich Qualitätsverlust

Die naheliegende Schlussfolgerung aus der Drifttabelle wäre: `semantic_state` braucht FP32. **Die Messung widerlegt das deutlich.**

Jenseits der trainierten Distanz von 1 024 kehrt sich das Verhältnis um. Gemessen mit 128 Recall-Beispielen und 256 Copy-Antworttoken je Zelle, alle übrigen Zustände und Gewichte in FP32:

| Reduzierter Zustand | Mantissenbits | Recall 1024 | Recall 4096 | Copy 4096 | Norm `semantic` |
|---|---:|---:|---:|---:|---:|
| keiner (FP32) | 23 | 90,6 % | **66,4 %** | 87,5 % | 0,0162 |
| `semantic` FP16 | 10 | 90,6 % | 82,0 % | 89,5 % | 0,0135 |
| `semantic` BF16 | 7 | **99,2 %** | **98,4 %** | **100,0 %** | **0,0070** |
| `fast` FP16 | 10 | 89,8 % | 89,8 % | 91,8 % | 0,0161 |
| `fast` BF16 | 7 | **99,2 %** | **98,4 %** | **100,0 %** | **0,0077** |
| `context` FP16 | 10 | 89,8 % | 70,3 % | 89,1 % | 0,0168 |
| `context` BF16 | 7 | 91,4 % | 74,2 % | 89,5 % | 0,0165 |

Bei Distanz 4 096 steigt der assoziative Abruf von 66,4 % auf 98,4 %, wenn `semantic_state` **gröber** dargestellt wird. Der Effekt ist monoton in der Mantissenlänge (23 → 10 → 7 Bits) und wird über zwei unabhängige Wege ausgelöst: direkt über `semantic_state` oder indirekt über `fast_state`, der die Schlüssel- und Wertcodes der Bindung speist. `context_state` löst ihn nicht aus – konsistent damit, dass er am Bindungspfad nicht beteiligt ist.

Beide wirksamen Wege senken zugleich die Norm des Bindungsakkumulators um rund 55 % (0,0162 → 0,0070 bzw. 0,0077). Das legt eine Erklärung nahe: **Der Bindungsakkumulator sättigt jenseits der trainierten Distanz.** Über 4 096 Schritte addieren sich viele kleine Beiträge irrelevanter Token auf und überlagern die eigentliche Bindung. Eine gröbere Rundung verschluckt diese kleinen Beiträge bei der Addition und wirkt damit wie eine implizite Schwelle.

Diese Erklärung ist eine **Hypothese, die zur Messung passt, aber nicht bewiesen ist.** Belegt sind: die Monotonie in der Mantissenlänge, die Kopplung an die Norm des Akkumulators und die Tatsache, dass nur die beiden am Bindungspfad beteiligten Zustände den Effekt auslösen.

Was daraus **nicht** folgt: dass BF16 „genauer" wäre. Bis Distanz 1 024 – dem Trainingsbereich – lösen alle Formate die Aufgaben gleich gut. Der Effekt tritt erst bei Extrapolation auf und ist deshalb eher ein Hinweis auf eine Schwäche des Akkumulators als auf eine Stärke von BF16. Die naheliegende Konsequenz wäre eine explizite Dämpfung oder ein Vergessens-Term im `semantic_state`. **Das ist bewusst nicht umgesetzt worden**: Eine Architekturänderung, deren einziger Anlass bessere Benchmarkzahlen sind, gehört nicht in diesen Milestone. Der Befund ist dokumentiert und gehört als eigene Frage in eine spätere Untersuchung.

### 3. Wo beginnt die Drift?

| Variante | `fast` > 1 % ab | `context` > 1 % ab | `semantic` > 1 % ab | Logits > 1 % ab | erste geänderte Vorhersage |
|---|---:|---:|---:|---:|---:|
| Zustände BF16 | nie | 256 | **64** | 1 024 | keine bis 8 192 |
| Zustände FP16 | nie | nie | 1 024 | nie | keine bis 8 192 |
| nur `semantic` BF16 | nie | nie | 256 | nie | keine bis 8 192 |
| alle Gewichte INT8 | 8 192 | nie | 256 | 8 192 | keine bis 8 192 |
| alle Gewichte INT4 | 64 | 64 | 64 | 64 | keine bis 8 192 |

Der wichtigste Unterschied liegt nicht in der Höhe, sondern im Verlauf:

- **Zustandspräzision erzeugt kumulative Drift.** Sie wächst mit der Sequenzlänge.
- **Gewichtsquantisierung erzeugt konstante Drift.** INT8 bleibt über alle Längen bei 0,9–1,6 %, INT4 bei 12–23 %, ohne jedes Wachstum. Ein quantisiertes Gewicht ist ein systematischer Versatz, kein Rundungsrauschen, das sich aufschaukelt.

In **keiner** gemessenen Variante änderte sich bis Länge 8 192 auch nur eine einzige Vorhersage, und in keiner traten NaN oder Inf auf. Die Drift ist real und messbar, wirkt sich auf diesen Aufgaben aber nicht auf die Ausgabe aus.

### 4. Wie weit kann Weight-Quantisierung gehen?

| Variante | Token/s | ms/Token | Gewichte | Checkpoint | Logit-Drift | Recall 16/64/256 |
|---|---:|---:|---:|---:|---:|---|
| fp32 (Referenz) | 2 255 | 1,038 | 419,8 KiB | 447,0 KiB | 0 | 100 / 100 / 100 % |
| alle Gewichte INT8 | 2 444 | 0,990 | **118,6 KiB** | **136,1 KiB** | 0,007 | 100 / 100 / 100 % |
| alle Gewichte INT4 | 2 404 | 0,934 | **67,0 KiB** | **82,3 KiB** | 0,139 | 100 / 100 / 100 % |
| alle Gewichte FP8-E4M3 | 2 480 | 0,930 | 118,6 KiB | 136,2 KiB | 0,020 | 100 / 100 / 100 % |
| alle Gewichte FP8-E5M2 | 2 442 | 0,934 | 118,6 KiB | 136,2 KiB | 0,039 | 100 / 100 / 100 % |
| INT4 mit Gruppengröße 8 | 2 481 | 0,938 | 113,3 KiB | 130,3 KiB | – | 100 / 100 / 100 % |

INT8 kostet 0,7 % Logit-Drift bei 3,54-fach kleineren Gewichten und ändert nichts an der Aufgabenqualität. INT4 kostet 13,9 % Logit-Drift bei 6,27-fach kleineren Gewichten – und ändert **immer noch** nichts an Recall und Selective Copy.

Die komponentenweise Messung zeigt, wo Empfindlichkeit sitzt:

| Komponente | INT8 Logit-Drift | INT4 Logit-Drift | INT4 Recall 1024 |
|---|---:|---:|---:|
| `gate` (Mixer-Gate) | **3,4 · 10⁻⁵** | 6,0 · 10⁻⁴ | 87,5 % |
| `local_mixer` | 1,1 · 10⁻⁴ | 0,002 | 87,5 % |
| `input_projection` | 9,1 · 10⁻⁴ | 0,025 | 87,5 % |
| `state_projection` | 0,002 | 0,029 | 87,5 % |
| `output_projection` (Integrator) | 0,005 | **0,121** | **84,4 %** |
| `embedding` / `lm_head` (gebunden) | – | – | **84,4 %** |

Der Integrator und die gebundene Embedding-/LM-Head-Matrix sind die empfindlichsten Stellen; das Mixer-Gate ist die unempfindlichste. Wer selektiv quantisiert, sollte diese Reihenfolge beachten.

Die Gruppenquantisierung ist ein Beispiel für eine Optimierung, die sich in der Messung **nicht rechnet**: INT4 mit Gruppengröße 8 braucht 113,3 KiB statt 67,0 KiB, weil bei acht Werten je FP32-Skala der Skalen-Overhead den Packungsgewinn fast vollständig auffrisst. Der Abstand zu INT8 (118,6 KiB) schrumpft auf 5 %.

### 5. Bringen INT8 oder INT4 Performance oder nur Speicher?

**Speicher: ja. Rechenzeit: nein.** Das ist der klarste Befund dieses Milestones.

Weight-Only-Quantisierung dequantisiert vor dem Matmul zurück in den Rechendatentyp. Der Microbenchmark misst die reinen Kosten:

| Operation | Zeit | Verhältnis zu dicht |
|---|---:|---:|
| dichtes FP32-Linear `[1×64]→320` | 15,98 µs | 1,00× |
| INT8 Weight-Only, ohne Cache | 43,66 µs | **2,73×** |
| INT4 Weight-Only, ohne Cache | 138,17 µs | **8,65×** |
| INT8 Weight-Only, mit Cache | 13,75 µs | 0,86× |
| INT8-Dequantisierung `[320×64]` | 15,37 µs | – |
| INT4-Dequantisierung `[320×64]` | 98,75 µs | 6,4× teurer als INT8 |

Im Streaming schlägt das voll durch:

| Variante | ms/Token | Gewichte | Laufzeit-VRAM |
|---|---:|---:|---:|
| alle Gewichte INT8, **mit** Cache | 0,990 | 118,6 KiB | 100,8 MiB |
| alle Gewichte INT8, **ohne** Cache | 1,227 (+24 %) | 118,6 KiB | 100,5 MiB |
| alle Gewichte INT4, **mit** Cache | 0,934 | 67,0 KiB | 100,8 MiB |
| alle Gewichte INT4, **ohne** Cache | **2,299 (+146 %)** | 67,0 KiB | 100,4 MiB |

Der Dequantisierungs-Cache beseitigt den Zeitnachteil – aber er hält das dichte Gewicht zusätzlich im Speicher und macht damit den Laufzeitvorteil zunichte. Beides zugleich gibt es auf dieser Hardware nicht, weil es keine INT8- oder INT4-Matmul-Kernel im portablen PyTorch-Pfad gibt.

Der echte Gewinn liegt deshalb dort, wo Gewichte *ruhen*: Checkpoints schrumpfen von 447,0 auf 136,1 KiB (INT8) beziehungsweise 82,3 KiB (INT4). Für Verteilung, Laden und Halten vieler Modelle ist das erheblich; für den Durchsatz eines laufenden Modells ist es neutral.

### 6. Was wählt `auto` auf dieser Hardware?

**Durchgehend FP32.**

```text
auto-Profil: compute=float32  fast=float32  context=float32  semantic=float32
  - Kein reduziertes Format schlägt float32 im Tokenpfad deutlich:
    bester Kandidat bfloat16 mit 2,9 % bei 5,9 % Messrauschen.
  - State-Update gewinnt in bfloat16 nur 0,7 % bei 4,8 % Rauschen.
  - FP8-Compute nicht verfügbar: Compute Capability 8.6 besitzt keine FP8-Tensorkerne.
```

`auto` vergleicht den gemessenen Vorsprung gegen die gemessene Streuung derselben Messreihen. Liegt der Vorsprung im Rauschen, bleibt das genauere Format stehen. Das ist die direkte Lehre aus Milestone 2.5, wo die Annahme „BF16 ist schneller" 37 % Durchsatz gekostet hatte.

```bash
python scripts/microbench.py                      # Hardware vermessen
python scripts/microbench.py --json               # maschinenlesbar
python scripts/device_info.py                     # welche Formate trägt dieses Backend?
```

### 7. Welche Formate beim späteren Scaling?

Aus den Messungen, nicht aus Konvention:

- **Rechenpfad:** BF16, sobald die Matrizen groß genug sind, dass der Matmul den Dispatch dominiert. Bei `[128×256]→1024` sind das bereits 36 %. Bei `d_model=64` lohnt es nicht.
- **Gewichte im Betrieb:** BF16, wenn das ganze Modell konvertiert wird – halber Speicher, 1,2 % Logit-Drift, unveränderte Aufgabenqualität. Nicht als reine Rechendatentyp-Umstellung bei FP32-Gewichten, das kostet Zeit für Casts.
- **Gewichte auf Datenträger:** INT8. 3,54× kleiner bei 0,7 % Drift. INT4 nur, wenn Speicher wirklich knapp ist.
- **`fast_state` und `context_state`:** BF16 unbedenklich; `fast_state` zeigt über 8 192 Token keinerlei kumulative Drift.
- **`semantic_state`:** BF16 ist im Trainingsbereich unbedenklich und jenseits davon sogar deutlich besser (siehe 2b). Die naheliegende FP32-Empfehlung aus der reinen Drifttabelle ist durch die Aufgabenmessung widerlegt. Solange der Akkumulator kein Dämpfungsglied hat, ist BF16 hier die messbar bessere Wahl.
- **FP8:** erst ab Compute Capability 8.9. Bis dahin nur als Speicherformat sinnvoll, und dort ist INT8 bei gleicher Größe genauer (0,7 % gegen 2,0 % Drift).

Der übergreifende Schluss: **Drift gegen FP32 ist eine Messgröße, kein Qualitätsmaß.** Beide Kennzahlen zusammen zu erheben – Abweichung *und* Aufgabenqualität – war in diesem Milestone der entscheidende Punkt. Wer nur die Drift betrachtet hätte, hätte die falsche Empfehlung ausgesprochen.

### Abschlussvergleich

Alle Werte auf einer RTX 3070, `d_model=64`, zwei Blöcke, aus demselben trainierten Checkpoint. Recall 16/64/256 liegt bei **jeder** Variante bei 100 %, Selective Copy bis Distanz 256 ebenfalls; die Spalten sind deshalb weggelassen. Recall 1024 und 4096 stammen aus einer eigenen Messung mit 128 Beispielen je Zelle (`benchmarks/milestone2_6-long-distance.json`) – mit acht Beispielen wäre die Aussage bei diesen Distanzen wertlos.

| Konfiguration | Token/s | ms/Tok | Train Tok/s | VRAM | RAM | Gewichte | Checkpoint | R1024 | R4096 | C4096 | State-Drift `sem` | Logit-Drift |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 (Referenz) | 2 255 | 1,038 | 7 350 | 99,5 MiB | 1,3 GiB | 419,8 KiB | 447,0 KiB | 90,6 % | 66,4 % | 87,5 % | 0 | 0 |
| fp32 + AMP-bf16 (M2.5) | 2 195 | 1,182 | 6 201 | 107,5 MiB | 1,7 GiB | 419,8 KiB | 447,0 KiB | 90,6 % | 66,4 % | 87,5 % | 6,1 · 10⁻⁴ | 0,002 |
| **Gewichte+Compute bf16** | 2 360 | 0,994 | 7 284 | **62,8 MiB** | 1,7 GiB | 209,9 KiB | 229,1 KiB | **99,2 %** | **98,4 %** | **100 %** | 0,410 | 0,012 |
| Gewichte+Compute fp16 | 2 092 | 1,022 | 7 437 | 62,8 MiB | 2,1 GiB | 209,9 KiB | 229,1 KiB | 89,8 % | 89,8 % | 91,8 % | 0,038 | 0,002 |
| balanced (compute bf16) | 1 876 | 1,320 | 6 104 | 88,4 MiB | 2,1 GiB | 419,8 KiB | 447,0 KiB | 99,2 % | 98,4 % | 100 % | 0,410 | 0,012 |
| fast (compute bf16) | 1 855 | 1,317 | 6 514 | 88,4 MiB | 2,1 GiB | 419,8 KiB | 447,0 KiB | 99,2 % | 98,4 % | 100 % | 0,410 | 0,012 |
| alle Gewichte INT8 | 2 444 | 0,990 | n. a. | 100,8 MiB | 1,3 GiB | 118,6 KiB | 136,1 KiB | 90,6 % | 65,6 % | 86,7 % | 0,013 | 0,007 |
| alle Gewichte INT4 | 2 404 | 0,934 | n. a. | 100,7 MiB | 1,4 GiB | **67,0 KiB** | **82,3 KiB** | 85,2 % | 61,7 % | 89,8 % | 0,190 | 0,139 |
| alle Gewichte FP8-E4M3 | 2 480 | 0,930 | n. a. | 100,8 MiB | 1,4 GiB | 118,6 KiB | 136,2 KiB | 89,8 % | 67,2 % | 87,9 % | 0,018 | 0,020 |
| INT4, Gruppengröße 8 | **2 481** | 0,938 | n. a. | 100,8 MiB | 1,4 GiB | 113,3 KiB | 130,3 KiB | – | – | – | – | – |
| **gemischt** (emb/head INT8, Projektionen INT4, Zustände BF16) | 2 143 | 1,206 | n. a. | 88,4 MiB | 1,6 GiB | **98,6 KiB** | **116,1 KiB** | **100 %** | **98,4 %** | **100 %** | 0,037 | 0,027 |

`Train Tok/s` steht bei quantisierten Varianten auf „n. a.": Weight-Only-quantisierte Module haben keine trainierbaren Gewichte. `State-Drift sem` ist die relative Abweichung des `semantic_state` gegen FP32 bei 8 192 Token.

### Bewertung

| Kategorie | Gewinner | Messwert | Einordnung |
|---|---|---|---|
| **schnellste Variante** | INT4 mit Gruppengröße 8 | 2 481 Token/s | 10 % vor FP32 bei rund 6 % Messstreuung – zu knapp für eine Empfehlung |
| **speichereffizienteste Variante** | alle Gewichte INT4 | 67,0 KiB Gewichte, 82,3 KiB Checkpoint | 6,27× kleiner, kostet aber Langdistanz-Recall |
| **stabilste Variante** | fp32 (Referenz) | Logit-Drift 0 | per Definition – und zugleich die schwächste Langdistanz-Qualität |
| **beste Gesamtvariante** | **gemischt: INT8-Embedding/LM-Head, INT4-Projektionen, BF16-Zustände** | 100 % / 98,4 % Recall bei 1024 / 4096, 98,6 KiB Gewichte | volle Aufgabenqualität bei 4,26× kleineren Gewichten |

Die beste Gesamtvariante kombiniert die beiden Achsen, die sich in dieser Messung als **unabhängig** erwiesen haben:

- **Zustandspräzision** entscheidet über die Langdistanz-Qualität. BF16-Zustände heben Recall 4096 von 66,4 % auf 98,4 %, ganz gleich ob die Gewichte quantisiert sind.
- **Gewichtsquantisierung** entscheidet über den Speicherbedarf und lässt die Langdistanz-Qualität weitgehend unberührt (INT8: 65,6 % gegen 66,4 % in FP32).

Wer nur eine der beiden Achsen betrachtet, kommt zu einer schlechteren Konfiguration als wer beide zusammen einstellt. Genau dafür ist die `PrecisionPolicy` pro Modulgruppe und pro Zustand getrennt einstellbar.

Dass „stabilste" und „beste" auseinanderfallen, ist kein Widerspruch, sondern das Kernergebnis: Stabilität misst die Nähe zu einer Referenz, Qualität misst das Lösen der Aufgabe. Bei diesem Modell sind das jenseits der trainierten Distanz gegenläufige Größen.

### Sichtbare Aktivität unter reduzierter Precision

Der Observation Bus erlaubt, die Verschiebung der *dargestellten* Aktivität zu beziffern. Mittlere absolute Abweichung je Zustand über 96 Token:

| Variante | Aktivität `fast` | Aktivität `context` | Aktivität `semantic` | Persistenz `context` |
|---|---:|---:|---:|---:|
| Gewichte+Compute BF16 | 6,4 · 10⁻⁴ | 4,2 · 10⁻⁴ | 2,1 · 10⁻⁴ | 0,072 |
| alle Zustände BF16 | 6,0 · 10⁻⁴ | 4,1 · 10⁻⁴ | 2,3 · 10⁻⁴ | 0,072 |
| alle Gewichte INT8 | 7,6 · 10⁻⁴ | 4,8 · 10⁻⁴ | 2,3 · 10⁻⁴ | 0,061 |

Die Aktivitätswerte verschieben sich um weniger als ein Promille. Sichtbar betroffen ist einzig die Persistenzdauer von `context`: Sie hängt an einer Schwelle, und Werte nahe dieser Schwelle kippen. Das Bild bleibt damit unter reduzierter Precision belastbar; die Vergleichszahlen stammen aus `compare_telemetry` und sind echte Telemetrie, keine Schätzung.

### FP8 und QAT

FP8 ist als Schnittstelle vorbereitet, aber **nirgends stillschweigend aktiv**. `fp8_compute_supported()` trennt ausdrücklich zwischen dem Datentyp (in PyTorch 2.13 vorhanden) und echten FP8-Rechenwerken (ab Compute Capability 8.9, auf der RTX 3070 mit 8.6 nicht vorhanden). Ein nicht unterstütztes Format führt zu `QuantizationUnsupported`, niemals zu einem Ersatzformat. Als reines Speicherformat ist FP8 messbar, wird aber im Bericht klar als solches bezeichnet.

Für späteres Quantization Aware Training liegt `fake_quantize` bereit: Quantisierung und Rücktransformation mit Straight-Through-Estimator, sodass der Gradient nicht abreißt. Ein QAT-Trainingslauf gehört bewusst nicht zu diesem Milestone.

### Die Baseline blieb erhalten

Milestone 2.6 fasst den Kern an: getrennte Zustandsdatentypen, ein Umweg über `linear_weight` für jede Gewichtsmatrix, eine Policy-Auflösung je Vorwärtslauf. Eine erste Messung zeigte dafür 8,9 % höhere Streaming-Latenz – im Sequenzdurchsatz nichts, aber im Tokenpfad messbar. Ursache waren String-Vergleiche der Policy, ein `isinstance`-Durchlauf je Gewichtszugriff, sieben statt einem dtype-Vergleich in der Vorprojektion und eine Parameteriteration in `initial_state`.

Nach der Verschlankung – vorab berechnete Bool-Flags in der Policy, Typidentitätsvergleich im häufigen Fall, ein gemeinsamer Schnellpfad für die Gates, konstanter Zugriff auf einen Referenzparameter:

| Kennzahl | Milestone 2.5 | vor der Verschlankung | Milestone 2.6 |
|---|---:|---:|---:|
| Durchsatz `off`, Länge 128 | 2 413 Tok/s | 2 366 Tok/s | 2 405–2 449 Tok/s |
| Streaming-Latenz (3 Läufe) | 1,127 ms | 1,190 / 1,205 / 1,291 ms | **1,123 / 1,124 / 1,136 ms** |
| Parameterzahl | 120 002 | 120 002 | 120 002 |
| Peak-VRAM `off`, Länge 128 | 9 446 912 B | 9 446 912 B | 9 446 912 B |

Die Logits des eingefrorenen Referenzcheckpoints sind gegenüber Milestone 2.5 **bitgleich** (max |Δ| = 0). `tests/test_precision.py::test_frozen_reference_logits_still_match` prüft das gegen die gespeicherte Referenz, `test_neutral_policy_is_bit_identical` gegen ein frisch gebautes Modell. Rohdaten der Nachmessung: `benchmarks/milestone2_6-baseline-check.jsonl`.

### Checkpoints

Format-Version 3 speichert zusätzlich `precision_policy`, die aufgelösten dtypes für Gewichte, Rechenpfad und alle drei Zustände, die Quantisierungsart samt Gruppengröße und Skalen sowie Backend-Metadaten. Format 1 und 2 werden weiterhin geladen und migriert.

```bash
# Ein quantisiertes Modell dicht laden, etwa für ein Backend ohne das Schema
python -c "from glassmind.training.checkpoint import load_checkpoint; \
           load_checkpoint('modell.pt', dequantize=True)"
```

Ein nicht quantisierter Checkpoint bleibt backendunabhängig ladbar; ein quantisierter lässt sich auf Wunsch dequantisiert öffnen. Die Abweichung zwischen beiden Ladewegen liegt bei 7,5 · 10⁻⁸ und ist reine FP32-Assoziativität.

## Beobachtung und Aktivität

Die Modellmodule kennen VisPy nicht. Sie senden ausschließlich strukturierte Ereignisse an einen kleinen Observation Bus:

- `off`: keine Metrikberechnung im Modellpfad,
- `summary`: kompakte Zustandsstatistik pro Sequenz und Block,
- `trace`: reale Clusteraktivität, Gates, Zustandsänderungen, Flüsse und Ausgabeverteilung pro Token,
- `full`: zusätzlich vollständige Zustandswerte; nur zur gezielten Fehlersuche.

Ein sichtbarer Cluster wird nicht nur wegen großer absoluter Aktivierung hell. Sein protokollierter Score verwendet standardmäßig:

```text
0,35 × Aktivierungs-RMS
+ 0,35 × Änderungs-RMS
+ 0,15 × mittlere Gate-Aktivität
+ 0,15 × eingehender Fluss-RMS
```

Knoten-IDs wie `core.1.context.cluster.3` bleiben zwischen Schritten und Replays stabil. Kantenwerte stammen aus den tatsächlich ausgeführten Eingangs-, Zustands- und Integratorprojektionen. Das ist Aktivitätstelemetrie, keine Behauptung über eine semantische Bedeutung des Clusters.

Trace-Ereignisse enthalten pro State-Cluster zusätzlich Aktivierungsstärke, Norm, Delta, Gate-Aktivität, relative Forget-Aktivität, gemessenen Informationsfluss, Persistenzdauer und Reaktivierung. Die Retention und eine ungefähre Zeitkonstante werden aus der relativen realen Zustandsänderung abgeleitet. Diese Zeitkonstante ist ein Dynamik-Proxy und kein exakt identifizierter kausaler Parameter.

`ClusterAnalyzer` sammelt bei stabilen Kanalgruppen reproduzierbar mittlere und maximale Aktivität, Aktivierungsanzahl, Dauer, Reaktivierungen und häufig beteiligte Token. Cluster bleiben absichtlich nummeriert; das Projekt weist ihnen keine erfundenen Begriffe zu. Alle Verlaufsberechnungen laufen nur in `trace`/`full`. `off` erzeugt weder Ereignisse noch Clusterstatistiken und bleibt numerisch exakt neutral.

## Trace und VisPy-Replay

```bash
python scripts/infer.py 'Alice owns a red car.' \
  --checkpoint runs/<lauf>/checkpoints/final.pt \
  --max-new-tokens 32 \
  --record runs/demo/trace.jsonl

python scripts/visualize.py --replay runs/demo/trace.jsonl --autoplay
```

Mit `--follow` lädt die Visualisierung einen Trace nach, während ein anderer Prozess ihn schreibt. Bedienung:

- Leertaste: Pause/Wiedergabe,
- Pfeil links/rechts: ein Token zurück/vor,
- Pos1/Ende: erster/letzter Frame,
- Tab oder Mausklick: Knoten auswählen,
- Mausrad/Ziehen: zoomen und verschieben.

Farbe kennzeichnet Zustandsart, Helligkeit den Aktivitätsscore, Knotengröße Aktivität und Delta, Nachleuchten die jüngere Historie, Umriss Persistenz beziehungsweise Reaktivierung und Kantenhelligkeit den gemessenen Fluss. Die Darstellung aggregiert Kanäle in eine konfigurierbare Zahl stabiler Cluster. `--follow` ist der Live-Modus; Live und Replay verwenden dieselben `NetworkFrame`-Datenstrukturen.

## Tests

```bash
python -m pytest
python scripts/smoke_test.py
python scripts/overfit_test.py
```

Der Smoke-Test prüft Shapes, Forward/Backward, vorhandene endliche Gradienten, deterministische Initialisierung, backendunabhängiges Save/Load, NaN/Inf-Freiheit, exakte Beobachter-Neutralität und die numerische Übereinstimmung von Sequenz- und Streaming-Pfad.

Die Milestone-2-Tests prüfen deterministische Associative-Recall- und Selective-Copy-Daten, gemeinsames Tiny-Overfit, State-Ablation, numerisch konsistentes Streaming mit Ablation, Cluster-Metriken, das erweiterte Replay und die exakte `off`-Gleichheit.

`tests/test_optimized_core.py` sichert die Milestone-2.5-Umformungen ab: fusionierter gegen unfusionierten Referenzpfad (mit und ohne jede Ablation), Sequenz gegen Streaming, Migration von Checkpoint-Format 1, gebündelte gegen einzeln berechnete Telemetrie, numerische Neutralität in allen vier Beobachtungsmodi und ein echter Trainingsschritt auf dem Beschleuniger, sofern einer verfügbar ist.

`tests/test_precision.py` deckt Milestone 2.6 ab: Bitgleichheit der neutralen Policy gegen die Milestone-2.5-Baseline, alle drei Gleitkommaformate, die vollständige 27er-Matrix unabhängiger State-dtypes, Streaming-/Sequenz-Gleichheit unter gemischter Precision, Beobachter-Neutralität in jedem Profil, INT8-/INT4-Roundtrips samt Speichernachweis, komponentenweise Quantisierung, gebundene Embedding-/LM-Head-Gewichte, den Straight-Through-Estimator für späteres QAT, die FP8-Schnittstelle ohne stillen Ersatz, quantisierte Checkpoints mit und ohne Dequantisierung beim Laden, den CPU-Fallback und die Langzeitdrift einschließlich der Zusicherung, dass FP32 gegen sich selbst exakt null Drift meldet.

`tests/test_context_state.py` deckt die Kontextaufgaben ab: Reproduzierbarkeit, korrekte Auflösung der Abschnitt-Schlüssel-Bindung, tatsächlich kollidierende Schlüssel, Wiederaufnahme der ersten Abschnittsmarke, Mischung aus Dokument- und Abschnittsebene, Lösbarkeit per Overfit, saubere Ablatierbarkeit von `context` und eine messbar langsamere Dynamik als `fast_state`. **Kein Test verlangt eine bestimmte kausale Effektgröße für `context_state`** – das wäre eine erzwungene Bedeutung. Wie groß der Beitrag ausfällt, steht in `runs/context-specialization-*/summary.md`.

`scripts/memory_test.py` und `scripts/routing_test.py` dokumentieren weiterhin ausdrücklich „nicht anwendbar“. Sie simulieren keine noch nicht vorhandenen Komponenten.

## Benchmarks

```bash
python scripts/benchmark.py --lengths 256 512 1024 2048 4096
python scripts/benchmark.py --device cpu --lengths 256 512
python scripts/benchmark.py --d-model 64 --layers 2 --batch-size 1 \
  --lengths 64 128 --iterations 3 --state-interactions \
  --baseline benchmarks/milestone1-baseline.jsonl \
  --output benchmarks/milestone2_5.jsonl
```

Für Precision- und Quantisierungsmessungen gibt es eigene Werkzeuge:

```bash
python scripts/microbench.py                       # Hardware vermessen, auto-Profil ableiten
python scripts/precision_baseline.py --checkpoint <…>   # Referenz einfrieren
python scripts/precision_matrix.py --checkpoint <…>     # vollständige Matrix
python scripts/precision_report.py --markdown           # Tabellen erzeugen
```

Der Benchmark misst `off`, `summary` und `trace` getrennt. Jede JSONL-Zeile speichert Modellkonfiguration, Parameterzahl, Hardware, Backend, PyTorch-Version, Precision, Batchgröße, Sequenzlänge, Operation, Compile-Status, Token/s, Beobachtungs-Overhead, Streaming-Latenz, Peak-Gerätespeicher und den maximalen residenten Hauptspeicher des Prozesses. Details und alle Messreihen: `benchmarks/README.md`.

Kurze Smoke-Benchmarks sind stark aufwärm- und systemlastabhängig. Sie belegen Funktionsfähigkeit, nicht allgemeine Effizienz. Belastbare Architekturvergleiche benötigen längere Läufe, mehrere Wiederholungen und später eine dokumentierte GRU/LSTM-Referenz.

## Milestone 4: Scaling and Real Language Training

Fünf Größenklassen, zwei echte Korpora, zwei getrennte Studien. Alle Revisionen
sind auf Commit-SHA gepinnt; ein Test erzwingt das.

```bash
python scripts/scaling_study.py --sections benchmark            # Budget bestimmen
python scripts/scaling_study.py --sections fixed --corpus tinystories --token-budget 4000000
python scripts/scaling_study.py --sections capacity --capacity-factor 1.2 --merge
python scripts/scaling_study.py --sections profile sequence precision quantization trace --merge
```

### Datensätze

| Korpus | Quelle | Lizenz | Revision | Ausschnitt | Token |
| --- | --- | --- | --- | --- | ---: |
| TinyStories train | `roneneldan/TinyStories` | CDLA-Sharing-1.0 | `f54c09fd2331` | Shard 1 von 4 | 478 332 312 |
| TinyStories validation | dito | dito | dito | vollständig | 19 240 340 |
| WikiText-103-raw train | `Salesforce/wikitext` | CC BY-SA 3.0 / GFDL | `b08601e04326` | vollständig | 539 295 549 |
| WikiText-103-raw validation | dito | dito | dito | vollständig | 1 144 248 |

**Vorverarbeitung:** UTF-8-Bytes plus Offset 4 (Byte-Tokenizer, 260 Symbole),
Dokumente durch EOS getrennt (TinyStories) bzw. zeilenweise gefügt (WikiText),
abgelegt als `uint16`-Memmap. Kein BPE — ein Token ist wörtlich ein Byte,
wodurch die Aussage über Streaming bei langem Kontext nicht von einer
Tokenisierungskonvention abhängt. Der Preis steht daneben: Perplexity ist
**pro Byte** und damit nicht mit wortbasierten Literaturwerten vergleichbar.
Deshalb wird durchgehend `bits_per_byte` berichtet.

**Warum diese beiden:** TinyStories ist so gebaut, dass kleine Modelle darauf
zusammenhängende Sprache lernen — nur damit wird die Frage „lernt GlassMind
Sprache?" überhaupt beantwortbar. WikiText-103 ist der harte Gegenpol mit
echter Enzyklopädiesprache. Die Aussagekraft beider wird getrennt gehalten.

### Größenklassen

| Klasse | d_model | Layer | Parameter | Mikro-Batch | Accum | effektiv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 64 | 2 | 120 002 | 16 | 1 | 16 |
| xs | 160 | 4 | 1 240 804 | 16 | 1 | 16 |
| small | 384 | 6 | 10 235 526 | 16 | 1 | 16 |
| medium | 640 | 8 | 37 403 528 | 16 | 1 | 16 |
| large | 896 | 10 | 90 548 874 | 8 | 2 | 16 |

Nur `large` braucht Accumulation: Batch 16 bei Sequenz 512 sprengt die 8 GB.

### Fixed-Token Scaling (je 3 997 696 Token)

TinyStories:

| Klasse | bpb ↓ | ppl/Byte | top1 | gültige Wörter | Training | Inferenz | Streaming | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 1,990 | 4,0 | 58,9 % | 81,5 % | 7 126 tok/s | 2 204 tok/s | 1,291 ms | 130 MB |
| xs | 1,566 | 3,0 | 66,9 % | 93,4 % | 3 426 tok/s | 1 090 tok/s | 2,263 ms | 423 MB |
| small | 1,363 | 2,6 | 70,9 % | 95,7 % | 2 355 tok/s | 716 tok/s | 3,001 ms | 1 470 MB |
| medium | 1,298 | 2,5 | 72,1 % | 96,2 % | 1 773 tok/s | 557 tok/s | 3,847 ms | 3 361 MB |
| large | 1,262 | 2,4 | 72,4 % | 96,9 % | 717 tok/s | 454 tok/s | 4,715 ms | 3 999 MB |

WikiText-103:

| Klasse | bpb ↓ | top1 | gültige Wörter | Gegenkorpus bpb |
| --- | ---: | ---: | ---: | ---: |
| tiny | 2,687 | 46,2 % | 55,8 % | 3,505 |
| xs | 2,333 | 53,3 % | 69,8 % | 3,257 |
| small | 2,083 | 57,8 % | 82,1 % | 3,145 |
| medium | 2,011 | 59,1 % | 85,7 % | 3,076 |
| large | 1,994 | 59,0 % | 87,1 % | 3,086 |

Die Qualität steigt monoton, sättigt aber stark: tiny→xs bringt −0,424 bpb,
medium→large nur noch −0,036 bpb. Auf dem jeweiligen **Gegenkorpus** liegt
`large` sogar hinter `medium` — auf beiden Korpora.

### Capacity-Aware Scaling

Dieselbe Architektur, größenabhängiges Budget (Faktor 1,2 je Leiterstufe):

| Klasse | Token | bpb ↓ | top1 | gültige Wörter | Gegenkorpus bpb |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 5 758 976 | 1,238 | 73,3 % | 96,0 % | 4,079 |
| medium | 6 914 048 | 1,121 | 75,5 % | 96,9 % | **3,896** |
| large | 8 290 304 | **1,053** | **76,7 %** | **98,8 %** | 3,910 |

**Das Fixed-Token-Defizit von `large` war ein Budget-Artefakt.** Mit
angemessenem Budget gewinnt es auf dem eigenen Korpus klar; auf dem
Gegenkorpus ist der Abstand zu `medium` auf 0,014 bpb geschrumpft.

Daten wirken stärker als Parameter: Bei `medium` bringen 1,7× mehr Token
−0,177 bpb, während 3,7× mehr Parameter (small→medium bei festem Budget) nur
−0,065 bpb brachten.

### Dispatch gegen Compute

| Klasse | d_model | ATen | CPU | GPU | GPU/Wanduhr | GPU/CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| tiny | 64 | 34 646 | 80,2 ms | 23,3 ms | 0,44 | 0,29 |
| xs | 160 | 69 154 | 159,1 ms | 49,7 ms | 0,35 | 0,31 |
| small | 384 | 111 355 | 247,9 ms | 101,1 ms | 0,44 | 0,41 |
| medium | 640 | 142 278 | 317,0 ms | 164,3 ms | 0,55 | 0,52 |
| large | 896 | 172 694 | 386,4 ms | 244,6 ms | 0,68 | 0,63 |

Der Quotient überschreitet zwischen `small` und `medium` die 0,5 — aber der
Übergang ist **allmählich**. Selbst bei 90 M Parametern trägt die GPU nur 68 %
der Wanduhrzeit. GlassMind wird auf dieser Hardware nie vollständig
compute-bound.

Der Engpass ist benennbar: Der rekurrente Kern läuft als sequentielle
Python-Schleife. Die Zahl der Durchläufe je Optimizer-Schritt ist
`TOKEN / batch` — die Sequenzlänge kürzt sich heraus. Gemessen bei Sequenz 512
und 8192 Token je Schritt:

| Klasse | batch 1 | batch 2 | batch 4 | batch 8 | batch 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| small | 144 | 286 | 582 | 1 208 | 2 294 |
| medium | 107 | 217 | 442 | 858 | 1 698 |
| large | 90 | 171 | 357 | 725 | OOM |

Linear im Mikro-Batch. Die Kosten je Durchlauf und Layer sind mit 1,16 ms
(small) und 1,11 ms (large) praktisch gleich, obwohl `large` 8,8-mal mehr
Parameter hat.

### Long-Context Streaming

Latenz je Token, nachdem bereits Kontext verarbeitet wurde:

| Kontextfenster | tiny | small | large |
| --- | ---: | ---: | ---: |
| 0 – 1 024 | 1,5138 ms | 3,7346 ms | 5,8137 ms |
| 4 096 – 5 120 | 1,4502 ms | 3,7472 ms | 5,8097 ms |
| 16 384 – 17 408 | 1,4540 ms | 3,7318 ms | 5,8175 ms |
| 65 536 – 66 560 | 1,4492 ms | 3,7313 ms | 5,8274 ms |
| **Änderung** | **−4,3 %** | **−0,1 %** | **+0,2 %** |

Nach 65 536 verarbeiteten Token kostet das nächste Token dasselbe wie das
erste. Durchsatz bei Sequenzlänge 1 024 → 32 768 fällt bei `small` von 555 auf
545 tok/s, also 1,8 % über den Faktor 32.

### Precision

| Klasse | fp32 | AMP bf16 | AMP fp16 | Gewichte bf16 | Gewichte fp16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| tiny | 1 842 | 1 829 | 1 829 | 1 808 | 1 803 |
| small | 561 | 559 | 561 | **599** | 590 |
| medium | 436 | 434 | 437 | **446** | 443 |
| large | 358 | 357 | 356 | 356 | 356 |

Bester Fall: `small` mit BF16-Gewichten, +6,8 %. Bei `large` kein Unterschied.
Die Milestone-2.6-Erkenntnis überträgt sich auf große Modelle — jetzt gemessen
statt angenommen. Grund: Der Dispatch-Overhead dominiert, und den ändert die
Zahlendarstellung nicht.

### Quantisierte Ablage (`medium`, 37,4 M)

| Variante | Datei | Faktor | Laden | Inferenz | Δbpb | geänderte Vorhersagen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 150,3 MB | 1,00× | 0,32 s | 584 tok/s | — | — |
| INT8 (Gruppe 10) | 52,8 MB | 0,35× | 0,28 s | 595 tok/s | −0,0000 | 0,20 % |
| INT4 (Gruppe 10) | 34,0 MB | 0,23× | 0,28 s | 596 tok/s | +0,0027 | 2,93 % |

INT8 ist qualitativ gratis bei 2,8× kleinerer Datei. **Kein
Geschwindigkeitsvorteil, nur Speicher** — 584/595/596 tok/s liegt im Rauschen.

Die Gruppengröße wird aus dem Modell abgeleitet, nicht geraten: GlassMinds
fusionierter `integrator` liest `2·d_model + semantic_width + binding_rank`
Werte, bei d_model=640 also 1930 = 2·5·193. Keine Zweierpotenz teilt das.

### State Intelligence auf echter Sprache

Ablations-Δloss auf WikiText-103, Fixed-Token:

| Klasse | `fast` | `context` | `semantic` |
| --- | ---: | ---: | ---: |
| tiny | 2,734 | 1,089 | 0,116 |
| xs | 1,389 | 0,963 | 0,403 |
| small | 1,342 | 0,731 | 0,451 |
| medium | 1,252 | 0,716 | 0,369 |
| large | 1,328 | 0,505 | 0,293 |

**`context_state` ist auf echter Sprache kausal relevant** (Δloss 0,5–1,1) —
auf den synthetischen Aufgaben aus Milestone 2.5 lag der Effekt bei ±0,0. Das
ist die erste Messung, die den dritten Zustand rechtfertigt, und sie kommt aus
dem Wechsel auf natürliche Sprache. Alle Klassen bestehen den
Unterscheidbarkeitstest.

### Sprachqualität, ungeschönt

`large` auf WikiText, greedy dekodiert, kollabiert in eine Wiederholung:

> „the season . The second control of the season , the second season of the
> season , the second season of the season …"

Gesampelt entsteht grammatisch geformter Wortsalat mit erfundenen Wörtern
(*pressual, cancilar, brough, missengers*). Die Kennzahl „87,1 % gültige
Wörter, distinct-2 0,97" misst den **gesampelten** Text; der greedy-Text ist
degeneriert. Beides gehört nebeneinander berichtet.

Bemerkenswert ist nicht die Textqualität, sondern dass das Modell nie ein Wort
als Einheit gesehen hat — nur 260 Byte-Symbole. Wortgrenzen, Groß- und
Kleinschreibung, Interpunktion und Buchstabierung sind aus rohen Bytefolgen
gelernt.

### Regression

| Prüfung | Ergebnis |
| --- | --- |
| Testsuite (258 Tests) | grün |
| Tiny Overfit | 99,2 %, Loss 3,448 → 0,012 |
| Streaming-Gleichheit | max. Fehler 1,19e-07 |
| Observation-off-Gleichheit | max. Fehler exakt 0 |
| Associative Recall @1024 | 89,1 % |
| Selective Copy @1024 | 96,1 % |
| State-Ablation (synthetisch) | `fast` −82,8 %, `semantic` −79,7 %, `context` ±0,0 % |

## Milestone 4.5: GlassMind Visual Inspector

Der Inspector ist ein Analysewerkzeug für bereits aufgezeichnete oder gerade
entstehende Modelltelemetrie. Er zeigt ausschließlich Gemessenes; wo eine
Größe fehlt, bleibt die Anzeige leer statt sie zu schätzen.

```bash
# Aufgezeichneten Trace ansehen
python -m glassmind.visualize.app --replay runs/milestone4_5/demo-tiny-memory.jsonl

# Live an einem Modell mitschauen
python -m glassmind.visualize.app --live runs/milestone4/m4-fixed-tinystories-small.pt \
    --prompt "Once upon a time" --tokens 256 --mode summary

# Demo-Replays und Rendering-Test erzeugen
python scripts/visual_demo.py

# Oberfläche profilieren
python scripts/visual_bench.py
```

### Aufbau

Die Bedienlogik ist vollständig vom Zeichnen getrennt. Das ist kein Selbstzweck:
GUI-Logik, die einen Grafikkontext braucht, lässt sich nicht testen.

| Modul | Aufgabe |
| --- | --- |
| `visualize/scene.py` | Detailstufen, Aggregation, Flussbilanz |
| `visualize/layout.py` | Struktur-Layout und Analyse-Layout |
| `visualize/inspector.py` | Zeitachse, Auswahl, Filter, Suche, Vergleich, Eingriffe |
| `visualize/live.py` | nichtblockierender Telemetriepuffer, Frame-Zusammenbau |
| `visualize/render.py` | VisPy-Batches: Knoten, Kanten, Speicherbank |
| `visualize/app.py` | Qt-Fenster um die VisPy-Leinwand |

`tests/test_visual_inspector.py` prüft 53 Fälle ohne Fenster; ein einziger Test
baut ein echtes Fenster und wird ohne Display übersprungen.

### Detailstufen und was sie tragen dürfen

Welche Stufe überhaupt möglich ist, hängt am Observation-Modus, mit dem der
Trace entstand. Die Oberfläche erfindet keine Stufe, die nicht aufgezeichnet ist:

| Modus | Ereignis | tiefste ehrliche Stufe |
| --- | --- | --- |
| `off` | keins | keine Netzaktivität |
| `summary` | `state_summary` | Layer / State-Region |
| `trace` | `network_step` | Cluster |
| `full` | `+ payload["full"]` | Units |

Wird eine feinere Stufe angefordert, als die Daten hergeben, fällt die Ansicht
zurück **und sagt es** in der Statuszeile: „Stufe Units nicht in der Telemetrie".

Beim Aggregieren bleibt der gemessene Gesamtfluss erhalten. Über alle vier
Stufen hinweg ergibt dieselbe Telemetrie dieselbe Summe (16 750,82 im
Lasttest) — ein Test sichert das ab, damit Aggregation weder Fluss erfindet
noch verliert.

### Level of Detail: gemessene Wirkung

Auf einem künstlich erzeugten Netz mit 99 984 Knoten (**Lasttest, keine
Modelltelemetrie**):

| Stufe | Knoten | Kanten | ms/Szene |
| --- | ---: | ---: | ---: |
| Modell | 1 | 0 | 91,7 |
| Layer | 8 | 0 | 150,0 |
| State-Region | 24 | 16 | 227,3 |
| Cluster | 99 984 | 66 656 | 537,8 |

LOD spart Faktor 5,9. Der Rest ist unvermeidbar: Auch die gröbste Stufe muss
alle Rohknoten des Tokens einmal lesen.

### Zeichenpfad

Fünf Zeichenaufrufe je Bild, unabhängig von der Knotenzahl — alle Knoten in
einem `Markers`-Visual, alle Kanten in einem `Line`-Visual:

| sichtbare Knoten | Kanten | ms/Bild | FPS | Zeichenaufrufe |
| ---: | ---: | ---: | ---: | ---: |
| 984 | 656 | 7,5 | 132,9 | 5 |
| 9 984 | 6 656 | 32,3 | 30,9 | 5 |
| 49 992 | 33 328 | 147,1 | 6,8 | 5 |
| 99 984 | 66 656 | 291,5 | 3,4 | 5 |

**Praktikabel sind rund 10 000 sichtbare Knoten bei 30 FPS.** Darüber muss
gefiltert oder aggregiert werden — dafür sind LOD und die Top-N-Filter da.

Zur Einordnung: Echte GlassMind-Telemetrie erzeugt je Token
`Layer × 3 × telemetry_clusters` Clusterknoten, bei 10 Layern und 8 Clustern
also 240. Die Zehntausenderwerte oben betreffen realistisch nur die
Unit-Stufe eines großen Modells im Modus `full`.

### Telemetrie-Overhead im Modellpfad

Verschränkt gemessen, fünf Runden, Median (eine einmal zu Beginn erhobene
Referenz driftete und ergab unmögliche negative Werte):

| Modus | Tok/s | Overhead |
| --- | ---: | ---: |
| ohne Bus | 7 163 | Referenz |
| `off` | 7 136 | 0,4 % |
| `summary` | 6 577 | 8,2 % |
| `trace` | 799 | 88,9 % |
| `full` | 768 | 89,3 % |

Deshalb ist `summary` der Standard der Live-Ansicht. Das Modell schreibt nur
in einen Ringpuffer; das Zusammenbauen der Frames und das Zeichnen passieren im
Anzeigethread. Läuft der Puffer voll, werden die ältesten Ereignisse verworfen
**und gezählt** — die Anzeige darf die Inferenz nicht bremsen, aber ein
Datenverlust muss sichtbar sein.

### Was gezeichnet wird und warum

Jede sichtbare Eigenschaft ist ein Messwert. Es gibt keine dekorativen Effekte.

| Darstellung | Messgröße |
| --- | --- |
| Knotengröße | Activation, Delta, Zahl aggregierter Elemente |
| Helligkeit | Activation plus abklingende Aktivitätshistorie |
| Randfarbe rot | Reaktivierung |
| Randhelligkeit | Persistenz |
| Kantenstärke und Deckkraft | gemessener Flusswert |
| Zellenfarbe Memory | Lese-/Schreibzugriff, Belegung |
| Zellengröße Memory | Slot-Stärke und Lesegewicht |
| Zellenrand Memory | Ersetzungsereignis, sonst Alter |

Farben trennen `fast`, `context`, `semantic`, Ein- und Ausgang sowie Memory.
Sie behaupten keine Bedeutung. Cluster heißen `core.3.context.cluster.17` und
werden nirgends benannt, kategorisiert oder gedeutet.

### Zwei Layouts, streng getrennt

**Struktur** ordnet nach dem Modellaufbau: Layer von links nach rechts,
State-Regionen darin, Speicherbank mittig darunter. Eine Position hängt
ausschließlich an der Knoten-ID, nie an einem Messwert — deshalb springt beim
Tokenwechsel nichts.

**Aktivitätsinseln** ist ein *Analyse*-Layout: Cluster, deren Aktivität über
den Trace korreliert, rücken zusammen (Pearson-Korrelation, kräftebasierte
Anordnung, über den Seed reproduzierbar). Nähe heißt dort „war oft gleichzeitig
aktiv" — nicht mehr. Die Oberfläche kennzeichnet die Ansicht als ANALYSE.

### Bedienung

Alles ist mit der Maus erreichbar: Replay öffnen, Ansicht zurücksetzen,
Auswahl zentrieren, Detailstufe, Layout, Suche, Transport
(`⏮ ◀ Play/Pause ▶ ⏭`), Tempo, Zeitachsen-Slider, Sprung zu Token.

Filter reduzieren die Darstellung nachweisbar. Auf einem Replay mit 28 Knoten:
ohne Filter 28 Knoten bei 148 FPS, mit Aktivitätsschwelle 19 Knoten bei
457 FPS, mit Top-5 dann 6 Knoten bei 544 FPS.

Ein Klick wählt Knoten oder Speicherzelle dauerhaft aus und zeigt die
Messwerte samt Aktivitätsverlauf. Ein ausgewählter Knoten bleibt sichtbar, auch
wenn ein Filter ihn ausschlösse — sonst verschwände die Detailtafel unbemerkt.

### Analyse-Eingriffe

`fast`/`context`/`semantic` ablatieren, Memory-Read und Memory-Write
abschalten. Die Eingriffe wirken nur auf die Analysesitzung, verändern kein
gespeichertes Modell, und die Statuszeile schreibt „ANALYSEMODUS". Ohne
aktiven Eingriff ist die Optionsliste leer und die Inferenz läuft unverändert.

### Demo-Replays

| Datei | Inhalt |
| --- | --- |
| `runs/milestone4_5/demo-tiny-memory.jsonl` | Tiny-Modell mit 32 Memory-Slots, Modus `trace` |
| `runs/milestone4_5/demo-full-units.jsonl` | Modus `full`, schaltet die Unit-Stufe frei |
| `runs/milestone4_5/inspector-screenshot.png` | automatischer Rendering-Test |

Der Rendering-Test prüft, dass tatsächlich ein Bild entsteht: 2 978 Farben und
41 933 nicht-Hintergrundpixel. Ein einfarbiges Bild wäre ein stiller
Fehlschlag und lässt den Test fehlschlagen.

## Aktuelle Grenzen

- Der Byte-Tokenizer ist robust und reproduzierbar, aber nicht so token-effizient wie BPE.
- Die Trainingsrekurrenz ist korrekt und deutlich schneller, aber weiterhin eine Python-Schleife über Token; sie ist nicht zeitlich parallelisiert.
- Der Kern ist bei `d_model=64` und Batch 1 immer noch dispatch-gebunden, nicht rechen-gebunden: Die GPU trägt nur 21,8 % der Wanduhr. Weitere Beschleunigung bräuchte entweder eine zeitlich parallelisierte Rekurrenz oder einen funktionierenden `torch.compile`-Pfad.
- Der gebundene State-Interaction-Pfad kostet gegenüber demselben Kern ohne Interaktion weiterhin 28–31 % Durchsatz. Das ist eine reale Architekturkostenstelle: 40,5 statt 25,0 Operationen pro Block-Schritt für Schlüssel-, Wert-, Bindungs- und Lesepfad. Sie ist ausgewiesen, nicht versteckt.
- `trace` bleibt mit rund 86 % Overhead absichtlich teuer und standardmäßig ausgeschaltet, auch wenn die gebündelten Transfers ihn versechsfacht haben.
- `context_state` bleibt kausal untergeordnet, sobald das Modell die Aufgaben tatsächlich löst. Siehe Befund oben.
- Der `--only-context`-Lauf lernt die Kontextaufgaben in 2400 Schritten nicht ausreichend (20–47 % Accuracy). Ob das an der Schrittzahl, der Lernrate oder der fehlenden Aufwärmung durch einfachere Aufgaben liegt, ist nicht auseinandergehalten.
- Weight-Only-Quantisierung spart Ablagegröße, aber keinen Laufzeitspeicher und keine Rechenzeit. Ohne INT8-/INT4-Matmul-Kernel im portablen Pfad bleibt es bei Dequantisierung vor dem Matmul; der Cache, der die Zeitkosten aufhebt, hebt zugleich den Speichervorteil auf.
- Der Bindungsakkumulator sättigt jenseits der trainierten Distanz. Gröbere Zustandsdarstellung dämpft das messbar (Recall 4 096: 66,4 % in FP32 gegen 98,4 % in BF16), was auf eine Architekturfrage hindeutet, die dieser Milestone bewusst nicht angefasst hat.
- Die Precision-Messungen gelten für `d_model=64` und zwei Blöcke auf einer RTX 3070. Bei größeren Modellen verschieben sich die Verhältnisse zugunsten reduzierter Formate; belegt ist das bisher nur im Microbenchmark, nicht am ganzen Modell.
- FP8 ist als Schnittstelle vorhanden, aber auf dieser Hardware ohne Rechenwerke. Eine Aussage über FP8-Compute gibt es deshalb nicht.
- `torch.compile` ist in dieser Umgebung nicht übersetzbar; es gibt daher keine Compile-Messreihe.
- Es gibt noch kein externes Memory und keine Experts. State-Ablation ist die erste reproduzierbare Intervention, noch keine allgemeine Interventionssuite.
- Hugging-Face-Daten sind seit Milestone 4 eingebunden (TinyStories, WikiText-103, Revisionen auf Commit-SHA gepinnt); die Skalierungsläufe selbst stehen noch aus.
- Der Visual Inspector trägt rund 10 000 sichtbare Knoten bei 30 FPS. Darüber muss LOD oder ein Filter aggregieren; 100 000 einzeln gezeichnete Knoten ergeben 3,4 FPS.
- Die Detailstufe `Units` braucht den Modus `full`, und der kostet 89 % Durchsatz. Für lange Läufe ist er unbrauchbar; er ist für kurze, gezielte Ausschnitte gedacht.
- Der Bild-Export (`canvas.render`) schlägt unter der nativen Wayland-Plattform mit „FrameBuffer attachments are incomplete" fehl. `scripts/visual_demo.py` schaltet dafür auf XWayland um; die interaktive Oberfläche ist nicht betroffen.
- Die Live-Ansicht zeigt die Cluster-Stufe nur im Modus `trace`. Im empfohlenen `summary`-Modus bleibt es bei Layer- und State-Regionen — das ist die ehrliche Folge davon, dass `state_summary` keine Clusterdaten enthält.

Der sinnvollste nächste Schritt ist eine Entscheidung über `context_state`: entweder eine Aufgabe finden, die ihn wirklich erzwingt, oder den dreistufigen Zustand auf die zwei messbar getragenen Stufen zurückführen. Parallel dazu steht die Frage aus Milestone 2.6 offen, ob der Bindungsakkumulator ein explizites Dämpfungsglied braucht. Erst danach sollte ein kleines bounded Sparse Memory als eigener, gegen den internen State verglichener Milestone folgen; MoE und herstellerspezifische Kernel bleiben weiterhin außen vor.
