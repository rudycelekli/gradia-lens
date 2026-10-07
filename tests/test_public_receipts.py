"""Opt-in real public API receipts, negative controls, and tamper-resistant replay."""

from __future__ import annotations

import copy
import os

import pytest

from gradia_lens.api import execute, load_receipt, receipt_digest, replay, save_receipt

pytestmark = pytest.mark.skipif(
    os.getenv("GRADIA_LENS_REAL_MODEL_TEST") != "1", reason="Opt in with GRADIA_LENS_REAL_MODEL_TEST=1"
)


@pytest.fixture(scope="module")
def engine():
    from gradia_lens import Engine

    return Engine()


@pytest.mark.parametrize(
    "operation,params",
    [
        ("compare", {"strength": 0, "max_new_tokens": 2}),
        ("trace", {"layer": 8, "strength": 0}),
        ("head", {"layer": 8, "head": 4, "fraction": 0, "max_new_tokens": 2}),
        ("weight", {"layer": 8, "neuron": 1253, "gain": 1, "max_new_tokens": 2}),
        ("evaluate", {"strength": 0}),
        ("search", {"layers": [10], "strengths": [0], "budget": 1}),
        ("challenge", {"strength": 0, "budget": 3}),
    ],
)
def test_public_zero_controls_restore_and_replay_exactly(engine, operation, params, tmp_path):
    receipt = execute(operation, params, engine=engine)
    assert receipt["integrity"]["weights_before_sha256"] == receipt["integrity"]["weights_after_sha256"]
    assert receipt["integrity"]["hooks_after"] == 0
    if operation in ("compare", "head", "weight"):
        assert receipt["result"]["baseline"] == receipt["result"]["steered"]
    if operation == "trace":
        assert receipt["result"]["output"]["margin_gain"] == 0
    path = tmp_path / "receipt.json"
    save_receipt(path, receipt)
    verified = replay(load_receipt(path), engine=engine)
    assert verified["replay"]["measurements_exact"] is True
    altered = copy.deepcopy(receipt)
    altered["sources"]["engine.py"] = "0" * 64
    altered["sha256"] = receipt_digest(altered)
    with pytest.raises(ValueError, match="sources"):
        replay(altered, engine=engine)


def test_changed_measurement_with_recomputed_digest_still_fails_replay(engine):
    receipt = execute("compare", {"strength": 0.15, "max_new_tokens": 2}, engine=engine)
    receipt["result"]["baseline"]["continuation"] += " altered measurement"
    receipt["sha256"] = receipt_digest(receipt)
    with pytest.raises(RuntimeError, match="measurements differ"):
        replay(receipt, engine=engine)


def test_direct_parameter_drift_cannot_produce_original_checkpoint_receipt(engine):
    import torch

    row = engine.model.transformer.h[8].mlp.c_proj.weight[1253]
    original = row.detach().clone()
    try:
        with torch.no_grad():
            row.zero_()
        assert engine.weight_digest() != engine.model_info()["weights_sha256"]
        with pytest.raises(RuntimeError, match="differ from the loaded checkpoint"):
            execute("compare", {"strength": 0, "max_new_tokens": 2}, engine=engine)
    finally:
        with torch.no_grad():
            row.copy_(original)
    assert engine.weight_digest() == engine.model_info()["weights_sha256"]
