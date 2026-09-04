"""Bindet den Memory-Funktionstest aus ``scripts/memory_test.py`` in die Suite ein."""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "memory_test.py"
    spec = importlib.util.spec_from_file_location("glassmind_memory_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_memory_functional_script_passes() -> None:
    result = _load_script().run_memory_test()
    assert result["passed"] is True
    assert result["occupied_slots"] == result["slots"]
    # Ersetzungen setzen erst ein, wenn der Speicher voll ist.
    assert result["replacement_events"] == result["tokens"] - result["slots"]
    assert result["max_stream_error"] < 2e-5
