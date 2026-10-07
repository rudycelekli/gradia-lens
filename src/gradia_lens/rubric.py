"""Discovery-only derivation and transparent reranking of recorded experiments.

No model is imported or evaluated here. Confirmation observations remain tied
to the original selected setting; reranking cannot turn another setting into a
confirmed result. Timing is optional observed metadata, never estimated.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

MAX_CANDIDATES = 300


def _finite(value: Any, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite measured number.")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite measured number.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite measured number.")
    if nonnegative and result < 0:
        raise ValueError(f"{label} must be nonnegative.")
    return result


def _weights(target: float, damage: float, latency: float) -> tuple[float, float, float]:
    values = (
        _finite(target, "Rubric weight", nonnegative=True),
        _finite(damage, "Rubric weight", nonnegative=True),
        _finite(latency, "Rubric weight", nonnegative=True),
    )
    if any(value > 10 for value in values):
        raise ValueError("Rubric weights must be at most 10.")
    return values


def rank_candidates(
    candidates: list[dict], *, target_weight: float = 1, damage_weight: float = 2, latency_weight: float = 0
) -> list[dict]:
    """Return copied observations with utility; ties retain captured order."""
    target, damage, latency = _weights(target_weight, damage_weight, latency_weight)
    scored = []
    for candidate in candidates:
        gain = _finite(candidate["target_gain"], "Target gain")
        loss = _finite(candidate["regression_damage"], "Regression damage", nonnegative=True)
        duration = candidate.get("evaluation_elapsed_seconds")
        if duration is not None:
            duration = _finite(duration, "Evaluation elapsed seconds", nonnegative=True)
        if latency and duration is None:
            raise ValueError("Latency penalty requires recorded duration for every candidate.")
        utility = _finite(target * gain - damage * loss - latency * (duration or 0), "Derived rubric utility")
        scored.append({**candidate, "utility": utility})
    return sorted(scored, key=lambda candidate: -candidate["utility"])


def pareto_ids(candidates: list[dict], *, include_latency: bool = False) -> list[str]:
    """Maximize gain and minimize damage, optionally also measured duration."""
    coordinates = []
    for candidate in candidates:
        point: tuple[float, ...] = (
            _finite(candidate["target_gain"], "Target gain"),
            -_finite(candidate["regression_damage"], "Regression damage", nonnegative=True),
        )
        if include_latency:
            point += (
                -_finite(candidate.get("evaluation_elapsed_seconds"), "Evaluation seconds", nonnegative=True),
            )
        coordinates.append(point)
    frontier = []
    for index, point in enumerate(coordinates):
        if not any(
            all(other[axis] >= value for axis, value in enumerate(point))
            and any(other[axis] > value for axis, value in enumerate(point))
            for other in coordinates
        ):
            frontier.append(candidates[index]["id"])
    return frontier


def discovery_metrics(observation: dict, *, summary_is_discovery: bool = False) -> tuple[dict, list[dict]]:
    """Derive from tagged discovery rows; never read confirmation outcome values."""
    rows = observation.get("rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Recorded probe rows must be objects.")
    discovery = [row for row in rows if row.get("partition") == "discovery"]
    if discovery:
        identifiers = [row["id"] for row in discovery if "id" in row]
        if any(not isinstance(identifier, str) for identifier in identifiers) or len(set(identifiers)) != len(
            identifiers
        ):
            raise ValueError("Captured discovery probe IDs must be unique strings.")
        targets = [row for row in discovery if row.get("split", row.get("role")) == "target"]
        regressions = [row for row in discovery if row.get("split", row.get("role")) in ("near", "unrelated")]
        if len(targets) + len(regressions) != len(discovery):
            raise ValueError("Discovery cases need target, near, or unrelated roles.")
        if not targets or not regressions:
            raise ValueError("Discovery comparison needs target and regression cases.")
        gain = _finite(
            sum(_finite(row.get("margin_gain"), "Target margin gain") for row in targets) / len(targets),
            "Derived target mean",
        )
        losses = []
        for row in regressions:
            delta = row.get("expected_logprob_delta")
            if delta is not None:
                losses.append(max(0.0, -_finite(delta, "Expected log probability change")))
            else:
                losses.append(_finite(row.get("regression_damage"), "Regression damage", nonnegative=True))
        return {
            "target_gain": gain,
            "regression_damage": _finite(
                sum(losses) / len(losses), "Derived regression mean", nonnegative=True
            ),
            "target_count": len(targets),
            "regression_count": len(regressions),
            "metric_source": "derived_from_tagged_discovery_rows",
        }, discovery
    if rows and any(row.get("partition") == "confirmation" for row in rows):
        raise ValueError("Confirmation-only observations cannot supply discovery rankings.")
    summary = observation.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("Recorded summary must be an object.")
    metrics = summary.get("discovery")
    if metrics is None and summary_is_discovery:
        metrics = summary
    if not isinstance(metrics, dict):
        raise ValueError("No declared discovery metrics were captured.")
    return {
        "target_gain": _finite(metrics.get("target_gain"), "Discovery target gain"),
        "regression_damage": _finite(
            metrics.get("regression_damage"), "Discovery regression damage", nonnegative=True
        ),
        "target_count": metrics.get("target_count"),
        "regression_count": metrics.get("regression_count"),
        "metric_source": "recorded_discovery_summary_without_individual_rows",
    }, []


def _unwrap(record: dict) -> dict:
    value = record.get("result", record)
    if not isinstance(value, dict):
        raise ValueError("Recorded result must be an object.")
    return value


def _origins(record: dict, depth: int = 0) -> list[tuple[dict, list[dict], bool]]:
    if depth > 4:
        raise ValueError("Campaign nesting exceeds the four-level limit.")
    data = _unwrap(record)
    origins = []
    if "points" in data:
        points = data["points"]
        if not isinstance(points, list) or any(not isinstance(point, dict) for point in points):
            raise ValueError("Search candidates must be objects.")
        origins.append((data, points, False))
    elif "sweep" in data:
        points = data["sweep"]
        if not isinstance(points, list) or any(not isinstance(point, dict) for point in points):
            raise ValueError("Sweep candidates must be objects.")
        origins.append((data, points, True))
    elif "rows" in data and isinstance(data.get("summary"), dict):
        origins.append((data, [data], False))
    # A selected evaluation attached to a search/sweep is an audit, not a
    # second acquisition candidate. Keep it at the parent for explicit binding.
    has_campaign = (
        "points" in data
        or "sweep" in data
        or any(
            isinstance(data.get(name), dict) and "points" in _unwrap(data[name])
            for name in ("search", "boundary")
        )
    )
    for name in ("search", "boundary") if has_campaign else ("search", "boundary", "evaluation"):
        item = data.get(name)
        if isinstance(item, dict) and item is not data:
            origins.extend(_origins(item, depth + 1))
    for name in ("searches", "experiments", "campaigns"):
        items = data.get(name, [])
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            raise ValueError(f"Campaign {name} must be a list of objects.")
        for item in items:
            origins.extend(_origins(item, depth + 1))
        if sum(len(points) for _, points, _ in origins) > MAX_CANDIDATES:
            raise ValueError("Campaign exceeds the 300-candidate viewer limit.")
    return origins


def _identity(point: dict, parent: dict, identifier: str) -> dict:
    provenance = point.get("provenance", parent.get("provenance", {}))
    if not isinstance(provenance, dict):
        raise ValueError("Recorded provenance must be an object.")
    model = provenance.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("Recorded model identity must be an object.")
    identity = {
        "experiment_id": identifier,
        "model_id": model.get("id"),
        "model_revision": model.get("revision"),
        "weights_sha256": model.get("weights_sha256"),
        "dtype": model.get("dtype"),
        "dataset_sha256": provenance.get("dataset_sha256"),
        "engine_source_sha256": provenance.get("engine_source_sha256"),
    }
    if any(value is not None and not isinstance(value, str) for value in identity.values()):
        raise ValueError("Compatibility fingerprints must be strings when present.")
    return identity


def _confirmation(evaluation: dict, candidate: dict, parent: dict, identity: dict) -> dict | None:
    """Attach only an audit with a matching, explicitly captured context."""
    if evaluation.get("confirmation_included") is False:
        return None
    if any(evaluation.get(key) != candidate[key] for key in ("layer", "strength")):
        return None
    if evaluation.get("experiment_id") != identity["experiment_id"]:
        return None
    context_identity = {key: value for key, value in identity.items() if key != "discovery_cases_sha256"}
    if _identity(evaluation, parent, identity["experiment_id"]) != context_identity:
        return None
    direction = candidate.get("direction_sha256")
    if direction is not None and evaluation.get("provenance", {}).get("direction_sha256") != direction:
        return None
    rows = evaluation.get("rows", [])
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Confirmation rows must be objects.")
    audit_rows = [row for row in rows if row.get("partition") == "confirmation"]
    summary = evaluation.get("summary", {})
    if not isinstance(summary, dict):
        raise ValueError("Confirmation summary must be an object.")
    if not audit_rows:
        totals = summary.get("confirmation")
        if not isinstance(totals, dict):
            return None
        target_count = totals.get("target_count", 0)
        regression_count = totals.get("regression_count", 0)
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (target_count, regression_count)
            )
            or target_count + regression_count == 0
        ):
            return None
    return {
        "rows": audit_rows,
        "summary": summary.get("confirmation"),
        "layer": evaluation["layer"],
        "strength": evaluation["strength"],
        "experiment_id": identity["experiment_id"],
    }


def analyze_candidates(
    record: dict, *, target_weight: float = 1, damage_weight: float = 2, latency_weight: float = 0
) -> dict:
    """Normalize compatible recorded candidates for a standalone interactive viewer."""
    if not isinstance(record, dict):
        raise ValueError("A campaign must be a recorded JSON object.")
    groups: dict[str, dict] = {}
    origins = _origins(record)
    if sum(len(points) for _, points, _ in origins) > MAX_CANDIDATES:
        raise ValueError("Campaign exceeds the 300-candidate viewer limit.")
    for origin_index, (parent, points, summary_only) in enumerate(origins):
        catalog = parent.get("catalog", {})
        if not isinstance(catalog, dict):
            raise ValueError("Recorded catalog must be an object.")
        identifier = parent.get("experiment_id", catalog.get("id", "unidentified"))
        if not isinstance(identifier, str):
            raise ValueError("Experiment identity must be a string.")
        for index, point in enumerate(points):
            layer, strength = point.get("layer"), point.get("strength")
            if isinstance(layer, bool) or not isinstance(layer, int) or not 0 <= layer <= 11:
                raise ValueError("Captured candidate layer must be an integer between 0 and 11.")
            strength = _finite(strength, "Captured strength")
            if not -2 <= strength <= 2:
                raise ValueError("Captured candidate strength must be between -2 and 2.")
            metrics, rows = discovery_metrics(point, summary_is_discovery=summary_only)
            identity = _identity(point, parent, identifier)
            # Matching dataset claims alone do not make differing case subsets
            # comparable. Bind the captured prompts and expectations as well.
            definitions = [
                {
                    key: row.get(key)
                    for key in ("id", "prompt", "split", "role", "desired_tokens", "undesired_tokens")
                }
                for row in rows
            ]
            identity["discovery_cases_sha256"] = (
                hashlib.sha256(json.dumps(definitions, sort_keys=True).encode()).hexdigest() if rows else None
            )
            # Missing fingerprints never license cross-record comparability.
            missing = any(value is None for value in identity.values())
            key = json.dumps(
                {**identity, "unidentified_origin": origin_index if missing else None}, sort_keys=True
            )
            group_id = hashlib.sha256(key.encode()).hexdigest()[:16]
            group = groups.setdefault(
                group_id,
                {
                    "id": group_id,
                    "experiment_id": identifier,
                    "identity": identity,
                    "compatibility_complete": not missing,
                    "candidates": [],
                    "confirmation": [],
                },
            )
            duration = point.get("evaluation_elapsed_seconds")
            if duration is None:
                # A full evaluation can include a different confirmation batch.
                # Its total time must not be substituted for discovery time.
                point_rows = point.get("rows", [])
                if (
                    rows
                    and isinstance(point_rows, list)
                    and all(row.get("partition") == "discovery" for row in point_rows)
                ):
                    duration = point.get("elapsed_seconds")
            if duration is not None:
                duration = _finite(duration, "Captured evaluation duration", nonnegative=True)
            candidate_id = f"{group_id}-{origin_index}-{index}"
            direction = point.get("provenance", {}).get("direction_sha256")
            if direction is not None and not isinstance(direction, str):
                raise ValueError("Captured direction fingerprint must be a string.")
            candidate = {
                "id": candidate_id,
                "layer": layer,
                "strength": strength,
                **metrics,
                "evaluation_elapsed_seconds": duration,
                "evaluation_timing_scope": "discovery_evaluation_batch"
                if duration is not None
                else "not_recorded",
                "direction_sha256": direction,
                "rows": rows,
                "captured_score": point.get("score", point.get("objective")),
                "captured_reason": point.get("reason"),
            }
            group["candidates"].append(candidate)
            selection_record = parent.get("selection", {})
            if not isinstance(selection_record, dict):
                raise ValueError("Recorded selection must be an object.")
            selection = parent.get("selected", selection_record.get("selected"))
            confirmation = parent.get(
                "selected_evaluation", parent.get("confirmation", parent.get("evaluation"))
            )
            # A standalone evaluation can carry its own frozen confirmation.
            if point is parent and "rows" in parent and "summary" in parent:
                confirmation, selection = parent, {"layer": layer, "strength": strength}
            if isinstance(selection, dict) and isinstance(confirmation, dict):
                if selection.get("layer") == layer and selection.get("strength") == strength:
                    audit = _confirmation(confirmation, candidate, parent, identity)
                    if audit is not None:
                        group["confirmation"].append({"candidate_id": candidate_id, "evaluation": audit})
    for group in groups.values():
        candidates = group["candidates"]
        group["all_candidates_timed"] = all(
            candidate["evaluation_elapsed_seconds"] is not None for candidate in candidates
        )
        group["ranked_candidates"] = rank_candidates(
            candidates,
            target_weight=target_weight,
            damage_weight=damage_weight,
            latency_weight=latency_weight,
        )
        group["pareto_ids"] = pareto_ids(candidates, include_latency=latency_weight > 0)
    return {
        "groups": list(groups.values()),
        "weights": {
            "target_gain": target_weight,
            "regression_damage": damage_weight,
            "latency_seconds": latency_weight,
        },
        "confirmation_used_for_ranking": False,
        "limitations": [
            "Utilities and frontier are derived from captured discovery proxies, not new model evaluations.",
            "Confirmation belongs only to the originally audited setting; choosing another setting needs fresh confirmation.",
            "Recorded durations include differing cache/fitting conditions and are not a controlled latency benchmark.",
            "Candidates with missing fingerprints are compared only within their original supplied record.",
        ],
    }
