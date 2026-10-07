"""Bounded public operations, lazy inference, and integrity-bound JSON receipts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import __version__
from .experiments import get_experiment

SCHEMA = "gradia.lens.receipt.v1"
MAX_RECEIPT_BYTES = 64 * 1024 * 1024
OPERATIONS = ("compare", "trace", "head", "weight", "evaluate", "search", "challenge")
LIMITATIONS = [
    "Hand-written lexical likelihood probes measure tested candidate continuations, not general correctness.",
    "Attention weights and logit-lens views describe partial computation; causal claims require interventions.",
    "Generated wording variants inherit labels and require review. Restoration does not prove general safety.",
]


def _integer(value: Any, low: int, high: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}.")
    return value


def _number(value: Any, low: float, high: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}.")
    return float(value)


def normalize_params(operation: str, params: dict[str, Any]) -> dict[str, Any]:
    """Validate before loading libraries or a checkpoint. No unbounded operations."""
    if operation not in OPERATIONS:
        raise ValueError("Unsupported Lens operation.")
    if not isinstance(params, dict):
        raise ValueError("Operation parameters must be an object.")
    fields = {
        "compare": {"prompt", "layer", "strength", "max_new_tokens"},
        "trace": {"prompt", "layer", "strength", "max_tokens"},
        "head": {"prompt", "layer", "head", "fraction", "max_new_tokens"},
        "weight": {"prompt", "layer", "neuron", "gain", "max_new_tokens"},
        "evaluate": {"layer", "strength", "include_confirmation"},
        "search": {"layers", "strengths", "budget", "strategy", "seed"},
        "challenge": {"layer", "strength", "budget", "seed"},
    }[operation] | {"experiment_id"}
    if set(params) - fields:
        raise ValueError(f"Unknown parameters: {', '.join(sorted(set(params) - fields))}")
    identifier = params.get("experiment_id", "positive-tone")
    if not isinstance(identifier, str):
        raise ValueError("experiment_id must be a string.")
    experiment = get_experiment(identifier)
    out: dict[str, Any] = {"experiment_id": experiment.id}
    if "prompt" in fields:
        prompt = params.get("prompt", experiment.default_prompt)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16384:
            raise ValueError("Prompt must contain text and be at most 16384 characters.")
        out["prompt"] = prompt
    if "layer" in fields:
        out["layer"] = _integer(params.get("layer", experiment.default_layer), 0, 11, "layer")
    if "strength" in fields:
        out["strength"] = _number(params.get("strength", 0.15), -2, 2, "strength")
    if "max_new_tokens" in fields:
        out["max_new_tokens"] = _integer(params.get("max_new_tokens", 16), 1, 64, "max_new_tokens")
    if operation == "trace":
        out["max_tokens"] = _integer(params.get("max_tokens", 48), 1, 48, "max_tokens")
    if operation == "head":
        out["head"] = _integer(params.get("head", 0), 0, 11, "head")
        out["fraction"] = _number(params.get("fraction", 1.0), 0, 1, "fraction")
    if operation == "weight":
        out["neuron"] = _integer(params.get("neuron", 0), 0, 3071, "neuron")
        out["gain"] = _number(params.get("gain", 1.0), 0, 2, "gain")
    if operation == "evaluate":
        confirmation = params.get("include_confirmation", False)
        if not isinstance(confirmation, bool):
            raise ValueError("include_confirmation must be a boolean.")
        out["include_confirmation"] = confirmation
    if operation in ("search", "challenge"):
        out["seed"] = _integer(params.get("seed", 17), 0, 2147483647, "seed")
        out["budget"] = _integer(params.get("budget", 6 if operation == "search" else 12), 1, 24, "budget")
    if operation == "search":
        layers, strengths = params.get("layers", [8, 10]), params.get("strengths", [0.0, 0.15, 0.35])
        if not isinstance(layers, list) or not layers or len(layers) > 12:
            raise ValueError("layers must be a nonempty list with at most 12 entries.")
        if not isinstance(strengths, list) or not strengths or len(strengths) > 25:
            raise ValueError("strengths must be a nonempty list with at most 25 entries.")
        out["layers"] = sorted({_integer(value, 0, 11, "layer") for value in layers})
        out["strengths"] = sorted({_number(value, -2, 2, "strength") for value in strengths})
        if out["budget"] > len(out["layers"]) * len(out["strengths"]):
            raise ValueError("Search budget exceeds the unique candidate grid.")
        strategy = params.get("strategy", "adaptive")
        if strategy not in ("adaptive", "uniform", "random"):
            raise ValueError("strategy must be adaptive, uniform, or random.")
        out["strategy"] = strategy
    return out


def source_hashes() -> dict[str, str]:
    """Bind receipts to all public Python implementation files, without importing them."""
    directory = Path(__file__).parent
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(directory.glob("*.py"))
    }


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def receipt_digest(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(
        _json_bytes({key: value for key, value in receipt.items() if key != "sha256"})
    ).hexdigest()


def save_receipt(path: str | Path, receipt: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump(receipt, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_receipt(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("Receipt exceeds the 64 MiB replay limit.")
    receipt = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != SCHEMA:
        raise ValueError("Expected a gradia.lens.receipt.v1 receipt.")
    if receipt.get("sha256") != receipt_digest(receipt):
        raise ValueError("Receipt SHA-256 integrity mismatch.")
    if receipt.get("sources") != source_hashes():
        raise ValueError("Receipt implementation sources differ from this installed edition.")
    normalize_params(receipt["operation"], receipt["params"])
    return receipt


def _model_engine(cache_dir: str | None):
    try:
        from .engine import Engine
    except ImportError as exc:
        raise RuntimeError(
            "Model dependencies are missing. Install 'gradia-lens[model]' or inspect doctor."
        ) from exc
    return Engine(cache_dir=cache_dir)


def _dispatch(engine: Any, operation: str, params: dict[str, Any]) -> dict[str, Any]:
    if operation == "search":
        from .search import BoundarySearch, experiment_fingerprint, pareto_indices

        started = time.perf_counter()
        policy = BoundarySearch(params["layers"], params["strengths"], params["strategy"], params["seed"])
        identifier = params["experiment_id"]
        points = policy.run(
            lambda layer, strength: engine.evaluate(identifier, strength, layer, include_confirmation=False),
            params["budget"],
        )
        selected = max(points, key=lambda point: point["score"])
        confirmation = engine.evaluate(
            identifier, selected["strength"], selected["layer"], include_confirmation=True
        )
        return {
            "experiment_id": identifier,
            "config": params,
            "fingerprint": experiment_fingerprint(params),
            "points": points,
            "pareto_indices": pareto_indices(points),
            "selected": {"layer": selected["layer"], "strength": selected["strength"]},
            "confirmation_used_for_selection": False,
            "selected_evaluation": confirmation,
            "acquisition_source_sha256": source_hashes()["search.py"],
            "provenance": confirmation["provenance"],
            "elapsed_seconds": time.perf_counter() - started,
            "timing_scope": "discovery evaluations, acquisition, and selected confirmation; excludes model construction and receipt hashing",
        }
    if operation == "challenge":
        from .challenges import propose_probes, rank_findings

        control = {key: params[key] for key in ("experiment_id", "layer", "strength")}
        probes = propose_probes(params["experiment_id"], params["budget"], params["seed"])
        result = engine.measure_probes(probes=probes, **control)
        result.update(
            findings=rank_findings(result["rows"]),
            proposal_seed=params["seed"],
            proposal_budget=params["budget"],
            proposals_count=len(probes),
            proposal_source_sha256=source_hashes()["challenges.py"],
            proposal_policy="discovery_seed_mechanical_wording_variants",
        )
        return result
    methods = {
        "compare": engine.run,
        "trace": engine.inspect_computation,
        "head": engine.run_head_ablation,
        "weight": engine.run_weight_experiment,
        "evaluate": engine.evaluate,
    }
    return methods[operation](**params)


def execute(
    operation: str, params: dict[str, Any], *, cache_dir: str | None = None, engine: Any = None
) -> dict[str, Any]:
    """Run one real, bounded operation and return a portable receipt. Never synthetic."""
    normalized = normalize_params(operation, params)
    load_started = time.perf_counter()
    constructed = engine is None
    engine = engine if engine is not None else _model_engine(cache_dir)
    load_elapsed = time.perf_counter() - load_started if constructed else None
    operation_started = time.perf_counter()
    # Hash snapshots share the engine's serialization boundary, so another
    # thread's temporary parameter preview cannot contaminate this receipt.
    with engine._lock:
        if engine.hook_count() != 0:
            raise RuntimeError("Lens receipt operations require an unmodified model.")
        before = engine.weight_digest()
        if before != engine.model_info()["weights_sha256"]:
            raise RuntimeError("Model parameters differ from the loaded checkpoint. Create a fresh Engine.")
        result = _dispatch(engine, operation, normalized)
        after = engine.weight_digest()
        if before != after or engine.hook_count() != 0:
            raise RuntimeError("Lens operation did not restore model parameters and hooks.")
    receipt = {
        "schema": SCHEMA,
        "package": {"name": "gradia-lens", "version": __version__},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "operation": operation,
        "params": normalized,
        "sources": source_hashes(),
        "result": result,
        "integrity": {"weights_before_sha256": before, "weights_after_sha256": after, "hooks_after": 0},
        "limitations": LIMITATIONS,
        "timing": {
            "model_load_elapsed_seconds": load_elapsed,
            "model_load_scope": "optional library imports and checkpoint construction; unrecorded for a supplied Engine",
            "receipt_operation_elapsed_seconds": time.perf_counter() - operation_started,
            "receipt_operation_scope": "serialized operation including full before/after parameter hashing; excludes model construction, JSON serialization, and file writing",
        },
    }
    receipt["sha256"] = receipt_digest(receipt)
    return receipt


def _measurements(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _measurements(item)
            for key, item in value.items()
            if key not in {"elapsed_seconds", "runtime_seconds", "evaluation_elapsed_seconds", "provenance"}
        }
    if isinstance(value, list):
        return [_measurements(item) for item in value]
    return value


def replay(receipt: dict[str, Any], *, cache_dir: str | None = None, engine: Any = None) -> dict[str, Any]:
    """Replay this exact implementation edition and fail if any measured value differs."""
    if receipt.get("schema") != SCHEMA or receipt.get("sha256") != receipt_digest(receipt):
        raise ValueError("Receipt schema or integrity mismatch.")
    if receipt.get("sources") != source_hashes():
        raise ValueError("Receipt implementation sources differ from this installed edition.")
    params = normalize_params(receipt["operation"], receipt["params"])
    engine = engine if engine is not None else _model_engine(cache_dir)
    old = receipt["result"]["provenance"]
    model = engine.model_info()
    for key in ("revision", "weights_sha256", "dtype"):
        if old["model"].get(key) != model.get(key):
            raise ValueError(f"Replay model provenance mismatch: {key}")
    fresh = execute(receipt["operation"], params, engine=engine)
    for key in ("dataset_sha256", "direction_sha256", "engine_source_sha256"):
        if old.get(key) != fresh["result"]["provenance"].get(key):
            raise ValueError(f"Replay computation provenance mismatch: {key}")
    if _measurements(receipt["result"]) != _measurements(fresh["result"]):
        raise RuntimeError("Replay measurements differ from the receipt.")
    fresh["replay"] = {"source_receipt_sha256": receipt["sha256"], "measurements_exact": True}
    fresh["sha256"] = receipt_digest(fresh)
    return fresh
