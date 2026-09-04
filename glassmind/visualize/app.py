"""Milestone 4.5: GlassMind Visual Inspector.

Die Oberfläche besteht aus zwei Teilen: einer VisPy-Leinwand, die das Netz
GPU-seitig in wenigen Batches zeichnet, und Qt-Bedienelementen darum herum.
Sämtliche Hauptfunktionen sind mit der Maus erreichbar; Tastenkürzel sind
Zugabe, nicht Voraussetzung.

Aufruf::

    python -m glassmind.visualize.app --replay runs/demo/trace.jsonl
    python -m glassmind.visualize.app --live runs/milestone4/m4-small.pt \\
        --prompt "Once upon a time"

Was die Oberfläche *nicht* tut: Werte schätzen, ergänzen oder glätten. Zeigt
der aktuelle Observation-Modus eine Detailstufe nicht, wird auf die tiefste
belegte Stufe zurückgefallen und das in der Statuszeile vermerkt.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Sequence

from glassmind.observe.bus import ObservationMode
from glassmind.visualize.graph import ReplayTimeline
from glassmind.visualize.inspector import Inspector
from glassmind.visualize.scene import ALL_KINDS, DetailLevel

TOOLTIP = {
    "fast": "Zeitskala fast – kurzlebige Tokendynamik",
    "context": "Zeitskala context – Satz- und Abschnittsdynamik",
    "semantic": "Zeitskala semantic – langsam veränderliche Merkmale",
    "memory": "Sparse External Memory (nur bei Modellen mit Speicher)",
    "input": "Eingang des Blocks",
    "output": "Ausgang des Blocks",
}


def _require_qt():
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except ImportError as exc:  # pragma: no cover - reine Installationsfrage
        raise RuntimeError(
            "Der Visual Inspector braucht PyQt6 und VisPy. "
            "Installation: pip install -e '.[visualize]'"
        ) from exc
    return QtCore, QtGui, QtWidgets


class Sparkline:
    """Kleiner Verlauf der Aktivität eines ausgewählten Knotens.

    Zeichnet ausschließlich aufgezeichnete Werte. Ohne Historie bleibt die
    Fläche leer statt eine Kurve zu erfinden.
    """

    def __new__(cls, *args: Any, **kwargs: Any):
        QtCore, QtGui, QtWidgets = _require_qt()

        class _Sparkline(QtWidgets.QWidget):
            def __init__(self, parent=None) -> None:
                super().__init__(parent)
                self.values: list[float] = []
                self.setMinimumHeight(56)
                self.setToolTip("Aktivitätsverlauf des ausgewählten Knotens")

            def set_values(self, values: Sequence[float]) -> None:
                self.values = list(values)
                self.update()

            def paintEvent(self, event) -> None:  # noqa: N802 - Qt-Namensschema
                painter = QtGui.QPainter(self)
                painter.fillRect(self.rect(), QtGui.QColor("#0d141d"))
                if len(self.values) < 2:
                    painter.setPen(QtGui.QColor("#5a6472"))
                    painter.drawText(self.rect(), QtCore.Qt.AlignmentFlag.AlignCenter,
                                     "kein Verlauf")
                    return
                low, high = min(self.values), max(self.values)
                span = (high - low) or 1.0
                width, height = self.width() - 8, self.height() - 12
                path = QtGui.QPainterPath()
                for index, value in enumerate(self.values):
                    x = 4 + width * index / max(len(self.values) - 1, 1)
                    y = 6 + height * (1.0 - (value - low) / span)
                    path.lineTo(x, y) if index else path.moveTo(x, y)
                painter.setPen(QtGui.QPen(QtGui.QColor("#3fd6c8"), 1.6))
                painter.drawPath(path)
                painter.setPen(QtGui.QColor("#7d8896"))
                painter.drawText(4, 12, f"{high:.3f}")
                painter.drawText(4, self.height() - 2, f"{low:.3f}")

        return _Sparkline(*args, **kwargs)


class InspectorWindow:
    """Das Hauptfenster. Die Klasse baut Qt erst beim Instanziieren auf."""

    def __new__(cls, *args: Any, **kwargs: Any):
        QtCore, QtGui, QtWidgets = _require_qt()
        from vispy import scene as vispy_scene

        from glassmind.visualize.render import DARK_BACKGROUND, NetworkRenderer

        class _Window(QtWidgets.QMainWindow):
            def __init__(
                self,
                inspector: Inspector,
                *,
                live: Any = None,
                title: str = "GlassMind Visual Inspector",
            ) -> None:
                super().__init__()
                self.inspector = inspector
                self.live = live
                self.setWindowTitle(title)
                self.resize(1600, 940)
                self._frame_times: list[float] = []
                self._suppress = False

                self.canvas = vispy_scene.SceneCanvas(
                    keys="interactive", bgcolor=DARK_BACKGROUND, show=False,
                )
                self.view = self.canvas.central_widget.add_view()
                self.view.camera = "panzoom"
                self.renderer = NetworkRenderer(self.view)
                self.canvas.events.mouse_press.connect(self._on_canvas_click)
                self.canvas.native.setParent(self)
                self.setCentralWidget(self.canvas.native)

                self._build_toolbar()
                self._build_filter_dock()
                self._build_detail_dock()
                self._build_transport()
                self.statusBar().showMessage("bereit")

                self.timer = QtCore.QTimer(self)
                self.timer.timeout.connect(self._on_tick)
                self.timer.start(int(1000 / max(self.inspector.speed, 0.5)))
                # Erst zeichnen, dann den Ausschnitt setzen: so tragen alle
                # Visuals bereits Daten, wenn die Kamera gesetzt wird.
                self.refresh()
                self.reset_view()

            # -- Aufbau ----------------------------------------------
            def _build_toolbar(self) -> None:
                bar = self.addToolBar("Ansicht")
                bar.setMovable(False)

                def action(text: str, slot, tip: str = "") -> None:
                    item = QtGui.QAction(text, self)
                    item.setToolTip(tip or text)
                    item.triggered.connect(slot)
                    bar.addAction(item)

                action("Replay öffnen…", self.open_replay,
                       "Einen aufgezeichneten JSONL-Trace laden")
                action("Ansicht zurücksetzen", self.reset_view,
                       "Zoom und Position auf das ganze Netz zurücksetzen")
                action("Auswahl zentrieren", self.center_selection,
                       "Den ausgewählten Knoten in die Bildmitte holen")
                bar.addSeparator()

                bar.addWidget(QtWidgets.QLabel(" Detailstufe: "))
                self.level_box = QtWidgets.QComboBox()
                self.level_box.addItem("automatisch (Zoom)", None)
                for level in DetailLevel:
                    self.level_box.addItem(level.label, level)
                self.level_box.setToolTip(
                    "Wie fein das Netz aufgelöst wird. 'automatisch' folgt dem Zoom."
                )
                self.level_box.currentIndexChanged.connect(self._on_level_changed)
                bar.addWidget(self.level_box)

                bar.addWidget(QtWidgets.QLabel("  Layout: "))
                self.layout_box = QtWidgets.QComboBox()
                self.layout_box.addItem("Struktur (Modellaufbau)", "structure")
                self.layout_box.addItem("Aktivitätsinseln (ANALYSE)", "activity")
                self.layout_box.setToolTip(
                    "Struktur zeigt den Modellaufbau. Aktivitätsinseln gruppieren "
                    "häufig gemeinsam aktive Cluster – das ist ein Analyse-Layout "
                    "und keine Aussage über Bedeutung."
                )
                self.layout_box.currentIndexChanged.connect(self._on_layout_changed)
                bar.addWidget(self.layout_box)

                bar.addSeparator()
                bar.addWidget(QtWidgets.QLabel("  Suche: "))
                self.search_box = QtWidgets.QLineEdit()
                self.search_box.setPlaceholderText("layer 1 · cluster 3 · slot 7 · token 42")
                self.search_box.setMaximumWidth(280)
                self.search_box.setToolTip(
                    "Sucht nach Layer, Cluster, Unit, Memory-Slot, Token oder "
                    "einem Teil der Knoten-ID"
                )
                self.search_box.returnPressed.connect(self._on_search)
                bar.addWidget(self.search_box)
                search_button = QtWidgets.QPushButton("Finden")
                search_button.clicked.connect(self._on_search)
                bar.addWidget(search_button)

            def _build_filter_dock(self) -> None:
                dock = QtWidgets.QDockWidget("Filter", self)
                dock.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea)
                panel = QtWidgets.QWidget()
                form = QtWidgets.QVBoxLayout(panel)

                group = QtWidgets.QGroupBox("Bereiche anzeigen")
                inner = QtWidgets.QVBoxLayout(group)
                self.kind_boxes: dict[str, Any] = {}
                for kind in ALL_KINDS:
                    box = QtWidgets.QCheckBox(kind)
                    box.setChecked(True)
                    box.setToolTip(TOOLTIP.get(kind, kind))
                    box.stateChanged.connect(self._on_filters_changed)
                    inner.addWidget(box)
                    self.kind_boxes[kind] = box
                self.flow_box = QtWidgets.QCheckBox("Fluss anzeigen")
                self.flow_box.setChecked(True)
                self.flow_box.setToolTip("Kanten mit gemessenem Informationsfluss")
                self.flow_box.stateChanged.connect(self._on_filters_changed)
                inner.addWidget(self.flow_box)
                form.addWidget(group)

                group = QtWidgets.QGroupBox("Reduzieren")
                grid = QtWidgets.QGridLayout(group)
                grid.addWidget(QtWidgets.QLabel("Activity-Schwelle"), 0, 0)
                self.threshold = QtWidgets.QDoubleSpinBox()
                self.threshold.setRange(0.0, 5.0)
                self.threshold.setSingleStep(0.01)
                self.threshold.setDecimals(3)
                self.threshold.setToolTip("Knoten unterhalb dieser Aktivität ausblenden")
                self.threshold.valueChanged.connect(self._on_filters_changed)
                grid.addWidget(self.threshold, 0, 1)

                grid.addWidget(QtWidgets.QLabel("Top-N Knoten"), 1, 0)
                self.top_nodes = QtWidgets.QSpinBox()
                self.top_nodes.setRange(0, 100_000)
                self.top_nodes.setSpecialValueText("alle")
                self.top_nodes.setToolTip("Nur die N aktivsten Knoten zeichnen")
                self.top_nodes.valueChanged.connect(self._on_filters_changed)
                grid.addWidget(self.top_nodes, 1, 1)

                grid.addWidget(QtWidgets.QLabel("Top-N Flüsse"), 2, 0)
                self.top_flows = QtWidgets.QSpinBox()
                self.top_flows.setRange(0, 100_000)
                self.top_flows.setSpecialValueText("alle")
                self.top_flows.setToolTip("Nur die N stärksten Kanten zeichnen")
                self.top_flows.valueChanged.connect(self._on_filters_changed)
                grid.addWidget(self.top_flows, 2, 1)
                form.addWidget(group)

                group = QtWidgets.QGroupBox("Nur zeigen")
                inner = QtWidgets.QVBoxLayout(group)
                self.only_changed = QtWidgets.QCheckBox("veränderte Knoten")
                self.only_changed.setToolTip("Knoten mit merklichem Delta zum Vortoken")
                self.only_persistent = QtWidgets.QCheckBox("persistente Knoten")
                self.only_persistent.setToolTip("Knoten, die lange durchgehend aktiv sind")
                self.only_reactivated = QtWidgets.QCheckBox("reaktivierte Knoten")
                self.only_reactivated.setToolTip(
                    "Knoten, die nach einer Pause wieder aktiv wurden"
                )
                for box in (self.only_changed, self.only_persistent, self.only_reactivated):
                    box.stateChanged.connect(self._on_filters_changed)
                    inner.addWidget(box)
                form.addWidget(group)

                group = QtWidgets.QGroupBox("Analyse-Eingriffe")
                inner = QtWidgets.QVBoxLayout(group)
                note = QtWidgets.QLabel(
                    "Eingriffe verändern nur diese Analysesitzung,\nnie das gespeicherte Modell."
                )
                note.setStyleSheet("color:#c9a227;")
                inner.addWidget(note)
                self.ablate_boxes: dict[str, Any] = {}
                for name in ("fast", "context", "semantic"):
                    box = QtWidgets.QCheckBox(f"{name} ablatieren")
                    box.stateChanged.connect(self._on_intervention_changed)
                    inner.addWidget(box)
                    self.ablate_boxes[name] = box
                self.memory_read_box = QtWidgets.QCheckBox("Memory-Read deaktivieren")
                self.memory_write_box = QtWidgets.QCheckBox("Memory-Write deaktivieren")
                for box in (self.memory_read_box, self.memory_write_box):
                    box.stateChanged.connect(self._on_intervention_changed)
                    inner.addWidget(box)
                form.addWidget(group)
                form.addStretch(1)

                dock.setWidget(panel)
                self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, dock)

            def _build_detail_dock(self) -> None:
                dock = QtWidgets.QDockWidget("Auswahl und Messwerte", self)
                panel = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(panel)

                self.detail_table = QtWidgets.QTableWidget(0, 2)
                self.detail_table.setHorizontalHeaderLabels(["Größe", "Wert"])
                self.detail_table.horizontalHeader().setStretchLastSection(True)
                self.detail_table.verticalHeader().setVisible(False)
                self.detail_table.setEditTriggers(
                    QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
                )
                layout.addWidget(self.detail_table, 3)

                layout.addWidget(QtWidgets.QLabel("Aktivitätsverlauf"))
                self.sparkline = Sparkline()
                layout.addWidget(self.sparkline, 1)

                layout.addWidget(QtWidgets.QLabel("Aktivste Cluster in diesem Token"))
                self.cluster_table = QtWidgets.QTableWidget(0, 5)
                self.cluster_table.setHorizontalHeaderLabels(
                    ["Knoten", "Activity", "Delta", "Persist.", "Reakt."]
                )
                self.cluster_table.verticalHeader().setVisible(False)
                self.cluster_table.setEditTriggers(
                    QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
                )
                self.cluster_table.cellClicked.connect(self._on_cluster_clicked)
                layout.addWidget(self.cluster_table, 3)

                dock.setWidget(panel)
                self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, dock)

            def _build_transport(self) -> None:
                dock = QtWidgets.QDockWidget("Zeitachse", self)
                dock.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
                panel = QtWidgets.QWidget()
                layout = QtWidgets.QVBoxLayout(panel)

                self.context_label = QtWidgets.QLabel("")
                self.context_label.setStyleSheet(
                    "font-family: monospace; font-size: 12px; color:#c8d4e0;"
                )
                self.context_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
                layout.addWidget(self.context_label)

                row = QtWidgets.QHBoxLayout()
                for text, slot, tip in (
                    ("⏮", lambda: self._seek(0), "An den Anfang springen"),
                    ("◀", lambda: self._step(-1), "Ein Token zurück"),
                    ("▶ Play / Pause", self._toggle_play, "Abspielen oder anhalten"),
                    ("▶", lambda: self._step(1), "Ein Token vor"),
                    ("⏭", lambda: self._seek(len(self.inspector) - 1), "Ans Ende springen"),
                ):
                    button = QtWidgets.QPushButton(text)
                    button.setToolTip(tip)
                    button.clicked.connect(slot)
                    row.addWidget(button)
                self.play_button = row.itemAt(2).widget()

                row.addWidget(QtWidgets.QLabel("  Tempo:"))
                self.speed_box = QtWidgets.QDoubleSpinBox()
                self.speed_box.setRange(0.5, 120.0)
                self.speed_box.setValue(self.inspector.speed)
                self.speed_box.setSuffix(" Token/s")
                self.speed_box.valueChanged.connect(self._on_speed_changed)
                row.addWidget(self.speed_box)

                row.addWidget(QtWidgets.QLabel("  Springe zu Token:"))
                self.jump_box = QtWidgets.QSpinBox()
                self.jump_box.setRange(0, max(0, len(self.inspector) - 1))
                self.jump_box.setToolTip("Direkt zu einem Tokenindex springen")
                self.jump_box.editingFinished.connect(
                    lambda: self._seek(self.jump_box.value())
                )
                row.addWidget(self.jump_box)
                row.addStretch(1)
                self.position_label = QtWidgets.QLabel("")
                row.addWidget(self.position_label)
                layout.addLayout(row)

                self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                self.slider.setRange(0, max(0, len(self.inspector) - 1))
                self.slider.setToolTip("Zeitachse – zu einem beliebigen Token ziehen")
                self.slider.valueChanged.connect(self._on_slider)
                layout.addWidget(self.slider)

                dock.setWidget(panel)
                self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, dock)

            # -- Ereignisse ------------------------------------------
            def _on_tick(self) -> None:
                changed = False
                if self.live is not None:
                    frames = self.live.poll()
                    if frames:
                        self.inspector.timeline = self.live.timeline
                        self._resize_timeline()
                        if self.inspector.playing:
                            self.inspector.seek(len(self.inspector) - 1)
                        changed = True
                if self.inspector.advance():
                    changed = True
                if changed:
                    self.refresh()

            def _on_canvas_click(self, event: Any) -> None:
                if getattr(event, "button", None) != 1:
                    return
                world = self.view.scene.transform.imap(event.pos)[:2]
                self.inspector.select_nearest((float(world[0]), float(world[1])))
                self.refresh()

            def _on_level_changed(self, _: int) -> None:
                value = self.level_box.currentData()
                if value is None:
                    self.inspector.auto_level = True
                else:
                    self.inspector.set_level(value)
                self.refresh()

            def _on_layout_changed(self, _: int) -> None:
                if self.layout_box.currentData() == "activity":
                    self.statusBar().showMessage("Analyse-Layout wird berechnet …")
                    QtWidgets.QApplication.processEvents()
                    self.inspector.use_activity_layout()
                else:
                    self.inspector.use_structure_layout()
                self.reset_view()
                self.refresh()

            def _on_filters_changed(self, *_: Any) -> None:
                if self._suppress:
                    return
                filters = self.inspector.filters
                filters.kinds = {
                    kind for kind, box in self.kind_boxes.items() if box.isChecked()
                }
                filters.show_flow = self.flow_box.isChecked()
                filters.activity_threshold = self.threshold.value()
                filters.top_nodes = self.top_nodes.value() or None
                filters.top_flows = self.top_flows.value() or None
                filters.only_changed = self.only_changed.isChecked()
                filters.only_persistent = self.only_persistent.isChecked()
                filters.only_reactivated = self.only_reactivated.isChecked()
                self.refresh()

            def _on_intervention_changed(self, *_: Any) -> None:
                if self._suppress:
                    return
                states = [name for name, box in self.ablate_boxes.items() if box.isChecked()]
                self.inspector.set_intervention("ablate_states", states)
                self.inspector.set_intervention(
                    "disable_memory_read", self.memory_read_box.isChecked()
                )
                self.inspector.set_intervention(
                    "disable_memory_write", self.memory_write_box.isChecked()
                )
                if self.inspector.analysis_mode:
                    self.statusBar().showMessage(
                        "ANALYSEMODUS aktiv – gilt nur für die nächste Live-Sitzung"
                    )
                self.refresh()

            def _on_search(self) -> None:
                hits = self.inspector.search(self.search_box.text())
                if hits:
                    self.center_selection()
                    self.statusBar().showMessage(f"{len(hits)} Treffer: {hits[0]}")
                else:
                    self.statusBar().showMessage("kein Treffer")
                self.refresh()

            def _on_slider(self, value: int) -> None:
                if self._suppress:
                    return
                self.inspector.seek(value)
                self.refresh()

            def _on_speed_changed(self, value: float) -> None:
                self.inspector.set_speed(value)
                self.timer.setInterval(int(1000 / max(self.inspector.speed, 0.5)))

            def _on_cluster_clicked(self, row: int, _: int) -> None:
                item = self.cluster_table.item(row, 0)
                if item is not None:
                    self.inspector.select(item.text())
                    self.center_selection()
                    self.refresh()

            # -- Aktionen --------------------------------------------
            def open_replay(self) -> None:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Replay öffnen", "runs", "Trace (*.jsonl);;Alle Dateien (*)"
                )
                if not path:
                    return
                try:
                    timeline = ReplayTimeline.from_trace(Path(path))
                except ValueError as exc:
                    QtWidgets.QMessageBox.warning(self, "Trace unbrauchbar", str(exc))
                    return
                self.inspector = Inspector(timeline, mode=self.inspector.mode)
                self.live = None
                self._resize_timeline()
                self.reset_view()
                self.refresh()
                self.statusBar().showMessage(f"geladen: {path}")

            def reset_view(self) -> None:
                # Die Grenzen kommen aus dem Layout, nicht aus den Szenenobjekten
                # von VisPy. Das ist unabhängig davon, ob ein Visual gerade Daten
                # trägt, und liefert bei leerer Auswahl trotzdem einen gültigen
                # Ausschnitt.
                xmin, ymin, xmax, ymax = self.inspector.scene().layout.bounds
                width = max(xmax - xmin, 1e-3)
                height = max(ymax - ymin, 1e-3)
                self.view.camera.rect = (xmin, ymin, width, height)

            def center_selection(self) -> None:
                scene = self.inspector.scene()
                target = self.inspector.selected_node
                if target is None and self.inspector.selected_slot is not None:
                    positions = scene.memory.get("positions", [])
                    if self.inspector.selected_slot < len(positions):
                        point = positions[self.inspector.selected_slot]
                    else:
                        return
                elif target is not None:
                    point = scene.layout.of(target)
                else:
                    return
                current = self.view.camera.rect
                self.view.camera.rect = (
                    point[0] - current.width / 2, point[1] - current.height / 2,
                    current.width, current.height,
                )

            def _toggle_play(self) -> None:
                self.inspector.toggle_play()
                self.play_button.setText(
                    "⏸ Pause" if self.inspector.playing else "▶ Play"
                )

            def _step(self, delta: int) -> None:
                self.inspector.pause()
                self.play_button.setText("▶ Play")
                self.inspector.step(delta)
                self.refresh()

            def _seek(self, index: int) -> None:
                self.inspector.seek(index)
                self.refresh()

            def _resize_timeline(self) -> None:
                self._suppress = True
                last = max(0, len(self.inspector) - 1)
                self.slider.setRange(0, last)
                self.jump_box.setRange(0, last)
                self._suppress = False

            # -- Zeichnen --------------------------------------------
            def refresh(self) -> None:
                started = time.perf_counter()
                scene = self.inspector.scene()
                stats = self.renderer.update(self.inspector, scene)

                self._suppress = True
                self.slider.setValue(self.inspector.index)
                self.jump_box.setValue(self.inspector.index)
                self._suppress = False

                self.position_label.setText(
                    f"Token {self.inspector.index + 1} / {len(self.inspector)}"
                )
                self._refresh_context()
                self._refresh_detail()
                self._refresh_clusters(scene)

                elapsed = time.perf_counter() - started
                self._frame_times.append(elapsed)
                del self._frame_times[:-60]
                fps = 1.0 / (sum(self._frame_times) / len(self._frame_times))
                message = self.inspector.status_line() + f" | {fps:.0f} FPS"
                if self.live is not None:
                    status = self.live.status()
                    message += (f" | live {status['tokens_per_second']:.0f} Tok/s"
                                f" verworfen={status['dropped']}")
                self.statusBar().showMessage(message)
                self.canvas.update()

            def _refresh_context(self) -> None:
                context = self.inspector.token_context()
                if not context["available"]:
                    self.context_label.setText(
                        "<i>kein Tokentext im Trace – die Zeitachse zeigt Indizes</i>"
                    )
                    return
                text = context["text"]
                caret = min(context["caret"], len(text))
                before = text[:caret].replace("<", "&lt;")
                after = text[caret:].replace("<", "&lt;")
                self.context_label.setText(
                    f"…{before}<span style='background:#2a4a63;color:#ffffff;'>"
                    f"{after[:1] or ' '}</span>{after[1:]}…"
                )

            def _refresh_detail(self) -> None:
                detail = self.inspector.selection_detail()
                rows = [(key, value) for key, value in detail.items()
                        if key not in ("Verlauf",)]
                self.detail_table.setRowCount(len(rows))
                for row, (key, value) in enumerate(rows):
                    self.detail_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(key)))
                    self.detail_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(value)))
                self.sparkline.set_values(detail.get("Verlauf", []))

            def _refresh_clusters(self, scene: Any) -> None:
                top = sorted(scene.nodes, key=lambda node: -node.activity)[:12]
                self.cluster_table.setRowCount(len(top))
                for row, node in enumerate(top):
                    values = (node.id, f"{node.activity:.4f}", f"{node.delta:.4f}",
                              str(node.persistence), "ja" if node.reactivation else "–")
                    for column, value in enumerate(values):
                        self.cluster_table.setItem(
                            row, column, QtWidgets.QTableWidgetItem(value)
                        )
                self.cluster_table.resizeColumnsToContents()

            def closeEvent(self, event) -> None:  # noqa: N802 - Qt-Namensschema
                if self.live is not None:
                    self.live.stop()
                super().closeEvent(event)

        return _Window(*args, **kwargs)


# ----------------------------------------------------------------------
# Einstiegspunkt
# ----------------------------------------------------------------------

def build_live_session(
    checkpoint: Path, prompt: str, *, mode: str, tokens: int, device: str,
    temperature: float,
) -> tuple[Any, Inspector]:
    """Baut eine Live-Sitzung aus einem Checkpoint."""
    import torch

    from glassmind.training.checkpoint import load_checkpoint
    from glassmind.visualize.live import LiveSession

    model, tokenizer, _ = load_checkpoint(checkpoint, device=device)
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    session = LiveSession(
        model, prompt_ids, max_new_tokens=tokens, mode=mode, device=device,
        temperature=temperature,
    )
    inspector = Inspector(
        session.timeline, mode=mode,
        decode=lambda ids: tokenizer.decode(list(ids)),
    )
    return session, inspector


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GlassMind Visual Inspector – echte Modelltelemetrie als Netz"
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--replay", type=Path, help="Aufgezeichneter JSONL-Trace")
    source.add_argument("--live", type=Path, help="Checkpoint für eine Live-Sitzung")
    parser.add_argument("--prompt", default="Once upon a time",
                        help="Prompt der Live-Sitzung")
    parser.add_argument("--tokens", type=int, default=256,
                        help="Wie viele Token die Live-Sitzung erzeugt")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default=None,
                        choices=["off", "summary", "trace", "full"],
                        help="Observation-Modus; Standard: summary live, trace im Replay")
    parser.add_argument("--autoplay", action="store_true")
    parser.add_argument("--fps", type=float, default=8.0)
    args = parser.parse_args(argv)

    QtCore, QtGui, QtWidgets = _require_qt()
    application = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    application.setStyle("Fusion")
    _apply_dark_palette(QtGui, QtCore, application)

    live = None
    if args.live is not None:
        mode = args.mode or "summary"
        live, inspector = build_live_session(
            args.live, args.prompt, mode=mode, tokens=args.tokens,
            device=args.device, temperature=args.temperature,
        )
        live.start()
    elif args.replay is not None:
        mode = args.mode or "trace"
        inspector = Inspector(ReplayTimeline.from_trace(args.replay), mode=mode)
    else:
        parser.error("Bitte --replay oder --live angeben")
        return 2

    inspector.set_speed(args.fps)
    if args.autoplay:
        inspector.play()
    window = InspectorWindow(inspector, live=live)
    window.show()
    return application.exec()


def _apply_dark_palette(QtGui: Any, QtCore: Any, application: Any) -> None:
    """Dunkles Thema mit gedämpften Tönen, damit Aktivität ablesbar bleibt."""
    palette = QtGui.QPalette()
    colors = {
        QtGui.QPalette.ColorRole.Window: "#121820",
        QtGui.QPalette.ColorRole.WindowText: "#d3dbe4",
        QtGui.QPalette.ColorRole.Base: "#0d141d",
        QtGui.QPalette.ColorRole.AlternateBase: "#161e28",
        QtGui.QPalette.ColorRole.Text: "#d3dbe4",
        QtGui.QPalette.ColorRole.Button: "#1b2530",
        QtGui.QPalette.ColorRole.ButtonText: "#d3dbe4",
        QtGui.QPalette.ColorRole.Highlight: "#2a4a63",
        QtGui.QPalette.ColorRole.HighlightedText: "#ffffff",
        QtGui.QPalette.ColorRole.ToolTipBase: "#1b2530",
        QtGui.QPalette.ColorRole.ToolTipText: "#d3dbe4",
    }
    for role, value in colors.items():
        palette.setColor(role, QtGui.QColor(value))
    application.setPalette(palette)


if __name__ == "__main__":
    raise SystemExit(main())
