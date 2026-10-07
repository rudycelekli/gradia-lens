"""Lazy import, bounded controls, and receipt validation before model loading."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gradia_lens import catalog
from gradia_lens.api import execute, load_receipt, normalize_params, replay, save_receipt

SOURCE = Path(__file__).resolve().parents[1] / "src"


@pytest.mark.parametrize("command", ["help", "catalog", "doctor", "negative"])
def test_base_operations_do_not_import_model_dependencies_or_use_network(command):
    code = """
import importlib.abc, json, socket, sys
sys.path.insert(0, sys.argv[1])
class BlockModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'transformers', 'numpy', 'huggingface_hub'):
            raise AssertionError('Optional model library was imported: ' + fullname)
sys.meta_path.insert(0, BlockModelImports())
def forbidden(*args, **kwargs):
    raise AssertionError('Network access was attempted')
socket.socket.connect = forbidden
import gradia_lens
assert len(gradia_lens.catalog()) == 3
from gradia_lens.cli import main
command = sys.argv[2]
if command == 'help':
    try:
        main(['--help'])
    except SystemExit as exc:
        assert exc.code == 0
elif command == 'negative':
    assert main(['challenge', '--budget', '25']) == 1
else:
    assert main([command]) == 0
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
"""
    process = subprocess.run(
        [sys.executable, "-I", "-c", code, str(SOURCE), command], capture_output=True, text=True, timeout=30
    )
    assert process.returncode == 0, process.stdout + process.stderr


@pytest.mark.parametrize(
    "operation,params",
    [
        ("compare", {"layer": True}),
        ("compare", {"strength": float("nan")}),
        ("compare", {"max_new_tokens": 65}),
        ("trace", {"max_tokens": 49}),
        ("head", {"head": 12}),
        ("head", {"fraction": -0.1}),
        ("weight", {"neuron": 3072}),
        ("weight", {"gain": 2.01}),
        ("evaluate", {"include_confirmation": "true"}),
        ("search", {"layers": [8], "strengths": [0], "budget": 2}),
        ("search", {"budget": 25}),
        ("challenge", {"seed": -1}),
        ("compare", {"experiment_id": []}),
        ("compare", {"module": "arbitrary"}),
    ],
)
def test_invalid_controls_are_rejected_before_model_loading(monkeypatch, operation, params):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Invalid operation attempted to load a model")

    monkeypatch.setattr("gradia_lens.api._model_engine", forbidden)
    with pytest.raises(ValueError):
        execute(operation, params)


def test_catalog_and_normalized_defaults_are_portable():
    assert len(catalog()) == 3
    assert normalize_params("evaluate", {})["include_confirmation"] is False
    assert normalize_params("search", {})["budget"] == 6
    assert normalize_params("weight", {})["gain"] == 1


def test_wrong_schema_and_tampered_receipts_fail_before_model_loading(tmp_path, monkeypatch):
    path = tmp_path / "invalid.json"
    save_receipt(path, {"schema": "not-a-research-receipt"})
    with pytest.raises(ValueError, match="gradia.lens.receipt"):
        load_receipt(path)
    path.write_text(json.dumps({"schema": "gradia.lens.receipt.v1", "sha256": "altered"}))
    with pytest.raises(ValueError, match="integrity"):
        load_receipt(path)
    monkeypatch.setattr("gradia_lens.api._model_engine", lambda *_: pytest.fail("Must not load model"))
    with pytest.raises(ValueError, match="integrity"):
        replay({"schema": "gradia.lens.receipt.v1", "sha256": "altered"})
