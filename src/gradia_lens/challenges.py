"""Mechanical wording candidates and ranking of measured probe observations.

These are label-inheriting challenges, not verified paraphrases or new facts.
Only discovery examples seed proposals; known confirmation strings participate
only in duplicate exclusion. No model outcomes are synthesized by this module.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections.abc import Iterable

from .experiments import get_experiment

MAX_PROBES = 24
ROLES = ("target", "near", "unrelated")
PREFIXES = (
    "For this example, ",
    "Consider this statement: ",
    "The following sentence describes an example: ",
    "Here is a short passage: ",
    "In the scenario below, ",
    "Read this sentence: ",
)
LABEL_LIMITATION = (
    "Expected continuations are inherited from a discovery example. "
    "The variant's semantics and label have not been independently verified."
)


def _normalized_prompt(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def propose_probes(experiment_id: str, budget: int = 12, seed: int = 17) -> list[dict]:
    """Return at most `budget` reproducible, unmeasured wording variants.

    The role-balanced order prevents the first few candidates from testing only
    target examples. Prefixes can change context, so every proposal needs review.
    """
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= MAX_PROBES:
        raise ValueError("Probe budget must be an integer between 1 and 24")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2147483647:
        raise ValueError("Seed must be an integer between 0 and 2147483647")
    experiment = get_experiment(experiment_id)
    rng = random.Random(seed)
    groups = {
        role: [case for case in experiment.cases if case.partition == "discovery" and case.split == role]
        for role in ROLES
    }
    for group in groups.values():
        rng.shuffle(group)
    ordered = []
    for index in range(max((len(group) for group in groups.values()), default=0)):
        for role in ROLES:
            if index < len(groups[role]):
                ordered.append(groups[role][index])

    # Confirmation labels are never read. Their prompt strings only prevent
    # accidentally proposing an existing holdout item as a new challenge.
    seen = {_normalized_prompt(case.prompt) for case in experiment.cases}
    seen.update(_normalized_prompt(prompt) for pair in experiment.training_pairs for prompt in pair)
    prefixes = {}
    for case in ordered:
        variants = list(PREFIXES)
        rng.shuffle(variants)
        prefixes[case.id] = variants
    proposals = []
    for variant_index in range(len(PREFIXES)):
        for case in ordered:
            prompt = prefixes[case.id][variant_index] + case.prompt
            normalized = _normalized_prompt(prompt)
            if normalized in seen:
                continue
            seen.add(normalized)
            identity = "\0".join((experiment_id, case.id, prompt)).encode()
            proposals.append(
                {
                    "id": "challenge-" + hashlib.sha256(identity).hexdigest()[:20],
                    "prompt": prompt,
                    "desired_tokens": list(case.desired),
                    "undesired_tokens": list(case.undesired),
                    "role": case.split,
                    "origin": "mechanical_wording_variant",
                    "seed_id": case.id,
                    "seed_partition": "discovery",
                    "proposal_seed": seed,
                    "review_required": True,
                    "label_limitation": LABEL_LIMITATION,
                }
            )
            if len(proposals) == budget:
                return proposals
    return proposals


def _measured(row: dict, name: str) -> float | None:
    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _margin(row: dict, side: str) -> float | None:
    value = row.get(side)
    return _measured(value, "margin") if isinstance(value, dict) else None


def rank_findings(rows: Iterable[dict]) -> list[dict]:
    """Rank actual observations for review without claiming semantic correctness.

    Non-target desired-probability losses come first, then target preference
    failures/deterioration. Severity is a measured nats proxy within its basis,
    not a calibrated semantic harm score. Missing observations sort last with
    severity=None rather than a fabricated outcome.
    """
    findings = []
    for position, original in enumerate(rows):
        row = copy.deepcopy(original)
        role = row.get("role", row.get("split"))
        kind = "unmeasured"
        severity = None
        basis = None
        group = 3
        if role in ("near", "unrelated"):
            delta = _measured(row, "expected_logprob_delta")
            damage = _measured(row, "regression_damage")
            measured_loss = -delta if delta is not None else damage
            if measured_loss is not None:
                severity = max(0.0, measured_loss)
                basis = "desired_logprob_loss"
                kind = "regression_loss" if severity > 0 else "no_measured_loss"
                group = 0 if severity > 0 else 2
        elif role == "target":
            change = _measured(row, "margin_gain")
            after = _margin(row, "steered")
            before = _margin(row, "baseline")
            if change is None and before is not None and after is not None:
                change = after - before
            if after is not None or change is not None:
                preference_failure = max(0.0, -after) if after is not None else 0.0
                deterioration = max(0.0, -change) if change is not None else 0.0
                severity = max(preference_failure, deterioration)
                basis = (
                    "negative_target_margin"
                    if preference_failure >= deterioration and after is not None
                    else "target_margin_loss"
                )
                if after is not None and after < 0:
                    kind = (
                        "target_preference_reversal"
                        if before is not None and before >= 0
                        else "target_preference_failure"
                    )
                else:
                    kind = "target_margin_loss" if deterioration > 0 else "no_measured_loss"
                group = 1 if severity > 0 else 2
        row.update(
            {
                "severity": severity,
                "severity_basis": basis,
                "finding_kind": kind,
                "review_required": True,
                "suggested_review": (
                    "Validate the inherited continuation label, then inspect this measured lexical change."
                    if severity is not None and severity > 0
                    else "Validate the inherited label; this row does not establish a verified failure."
                ),
            }
        )
        findings.append((group, -(severity if severity is not None else 0.0), position, row))
    findings.sort(key=lambda finding: finding[:3])
    return [finding[3] for finding in findings]
