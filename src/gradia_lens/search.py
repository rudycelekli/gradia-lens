"""Budgeted search. Acquisition only reads discovery metrics.

This is a transparent heuristic, not a learned world model. The same finite grid
can be searched uniformly or randomly to test the adaptive policy fairly.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Candidate:
    layer: int
    strength: float


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        raise ValueError("Non-finite measured metric")
    return result


def metrics(evaluation: dict[str, Any]) -> dict[str, float]:
    """Normalize the engine summary without using confirmation rows."""
    summary = evaluation.get("summary", {})
    discovery = summary.get("discovery", summary)
    gain = _number(discovery.get("target_gain", discovery.get("mean_target_delta", 0)))
    damage = max(0.0, _number(discovery.get("regression_damage", 0)))
    return {
        "target_gain": gain,
        "regression_damage": damage,
        "score": gain - 2.0 * damage,
    }


def pareto_indices(points: Sequence[dict[str, Any]]) -> list[int]:
    """Higher target gain and lower measured regression damage are preferred."""
    result = []
    for index, point in enumerate(points):
        dominated = any(
            other["target_gain"] >= point["target_gain"]
            and other["regression_damage"] <= point["regression_damage"]
            and (
                other["target_gain"] > point["target_gain"]
                or other["regression_damage"] < point["regression_damage"]
            )
            for other in points
        )
        if not dominated:
            result.append(index)
    return result


class BoundarySearch:
    def __init__(
        self,
        layers: Sequence[int],
        strengths: Sequence[float],
        strategy: str = "adaptive",
        seed: int = 17,
    ):
        if strategy not in ("adaptive", "uniform", "random"):
            raise ValueError("Unknown search strategy")
        if not layers or not strengths:
            raise ValueError("Search grid must not be empty")
        self.layers = sorted(set(layers))
        self.strengths = sorted(set(float(value) for value in strengths))
        if any(not math.isfinite(value) for value in self.strengths):
            raise ValueError("Strengths must be finite")
        self.grid = [Candidate(layer, strength) for layer in self.layers for strength in self.strengths]
        self.strategy = strategy
        self.seed = seed
        self.observed: list[dict[str, Any]] = []
        self.remaining = set(self.grid)
        self._random_order = list(self.grid)
        random.Random(seed).shuffle(self._random_order)

    def distance(self, left: Candidate, right: Candidate) -> float:
        layer_scale = max(1, self.layers[-1] - self.layers[0])
        strength_scale = max(1.0, self.strengths[-1] - self.strengths[0])
        return math.hypot(
            (left.layer - right.layer) / layer_scale,
            (left.strength - right.strength) / strength_scale,
        )

    def _location(self, point: dict[str, Any]) -> Candidate:
        return Candidate(point["layer"], point["strength"])

    def _acquisition(self, candidate: Candidate) -> tuple[float, str]:
        nearest = sorted(
            [(self.distance(candidate, self._location(point)), point) for point in self.observed],
            key=lambda pair: pair[0],
        )[:4]
        uncertainty = nearest[0][0]
        total_weight = sum(1.0 / (distance + 0.05) for distance, _ in nearest)
        prediction = sum(point["score"] / (distance + 0.05) for distance, point in nearest) / total_weight
        scale = max(0.25, max(abs(point["score"]) for point in self.observed))
        boundary = 0.0
        for i, (distance_a, point_a) in enumerate(nearest):
            for distance_b, point_b in nearest[i + 1 :]:
                flips_gain = point_a["target_gain"] * point_b["target_gain"] < 0
                flips_damage = (point_a["regression_damage"] > 0.1) != (point_b["regression_damage"] > 0.1)
                if flips_gain or flips_damage:
                    boundary = max(boundary, 0.35 / (1.0 + distance_a + distance_b))
        explore = len(self.observed) % 3 == 0
        value = uncertainty if explore else prediction / scale * 0.25 + uncertainty * 0.65 + boundary
        reason = (
            "Cover an unmeasured region"
            if explore
            else ("Resolve a nearby observed tradeoff" if boundary else "Test a promising uncertain setting")
        )
        return value, reason

    def next(self) -> tuple[Candidate, str] | None:
        if not self.remaining:
            return None
        if not self.observed:
            candidate = min(self.remaining, key=lambda c: (abs(c.strength), c.layer))
            return candidate, "Measure the zero or smallest intervention control"
        if self.strategy == "random":
            return (
                next(c for c in self._random_order if c in self.remaining),
                "Seeded random control",
            )
        if self.strategy == "uniform":
            candidate = max(
                sorted(self.remaining, key=lambda c: (c.layer, c.strength)),
                key=lambda c: min(self.distance(c, self._location(p)) for p in self.observed),
            )
            return candidate, "Uniform space-filling control"
        candidates = sorted(self.remaining, key=lambda c: (c.layer, c.strength))
        candidate = max(candidates, key=lambda c: self._acquisition(c)[0])
        return candidate, self._acquisition(candidate)[1]

    def observe(self, candidate: Candidate, evaluation: dict[str, Any], reason: str) -> dict[str, Any]:
        if candidate not in self.remaining:
            raise ValueError("Candidate already evaluated or not in grid")
        point = {
            "layer": candidate.layer,
            "strength": candidate.strength,
            **metrics(evaluation),
            "reason": reason,
            "summary": evaluation.get("summary", {}),
            "rows": evaluation.get("rows", evaluation.get("cases", [])),
            "provenance": evaluation.get("provenance", {}),
        }
        self.remaining.remove(candidate)
        self.observed.append(point)
        return point

    def run(
        self,
        evaluate: Callable[[int, float], dict[str, Any]],
        budget: int,
        on_point: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[dict[str, Any]]:
        if budget < 1 or budget > len(self.grid):
            raise ValueError("Budget outside candidate grid")
        for _ in range(budget):
            proposal = self.next()
            if proposal is None:
                break
            candidate, reason = proposal
            point = self.observe(candidate, evaluate(candidate.layer, candidate.strength), reason)
            if on_point:
                on_point(point)
        return self.observed


def experiment_fingerprint(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()
