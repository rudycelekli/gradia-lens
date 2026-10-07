"""Recorded ranking invariants, independent of Torch and any model execution."""

from __future__ import annotations

import copy

import pytest

from gradia_lens.api import _measurements, receipt_digest
from gradia_lens.rubric import analyze_candidates, discovery_metrics, pareto_ids, rank_candidates
from gradia_lens.search import BoundarySearch, Candidate


def provenance():
    return {
        "model": {"id": "synthetic", "revision": "rev", "weights_sha256": "weights", "dtype": "float32"},
        "dataset_sha256": "cases",
        "engine_source_sha256": "engine",
    }


def observation(layer=8, strength=0.3, gain=1.0, damage=0.1, elapsed=2.0):
    return {
        "experiment_id": "synthetic",
        "layer": layer,
        "strength": strength,
        "rows": [
            {
                "id": "t",
                "partition": "discovery",
                "split": "target",
                "prompt": "target",
                "margin_gain": gain,
                "expected_logprob_delta": gain,
            },
            {
                "id": "r",
                "partition": "discovery",
                "split": "near",
                "prompt": "near",
                "margin_gain": -damage,
                "expected_logprob_delta": -damage,
            },
        ],
        "summary": {"discovery": {"target_gain": 999, "regression_damage": 999}},
        "provenance": provenance(),
        "elapsed_seconds": elapsed,
    }


def search_record():
    points = [observation(gain=1, damage=0.4), observation(layer=9, gain=0.7, damage=0.05)]
    audit = copy.deepcopy(points[0])
    audit["rows"].append(
        {
            "id": "audit",
            "partition": "confirmation",
            "split": "target",
            "prompt": "sealed",
            "margin_gain": -999,
            "expected_logprob_delta": -999,
        }
    )
    audit["summary"]["confirmation"] = {"target_gain": -999, "regression_damage": 999}
    return {
        "experiment_id": "synthetic",
        "points": points,
        "selected": {"layer": 8, "strength": 0.3},
        "selected_evaluation": audit,
        "provenance": provenance(),
    }


def test_discovery_rows_override_summary_and_confirmation_never_affects_selection():
    source = search_record()
    result = analyze_candidates(source)
    group = result["groups"][0]
    assert [(p["target_gain"], p["regression_damage"]) for p in group["candidates"]] == [
        (1, 0.4),
        (0.7, 0.05),
    ]
    assert group["ranked_candidates"][0]["layer"] == 9
    assert group["confirmation"][0]["candidate_id"] == group["candidates"][0]["id"]
    changed = copy.deepcopy(source)
    changed["selected_evaluation"]["rows"][-1]["margin_gain"] = 1e9
    changed["selected_evaluation"]["summary"]["confirmation"]["target_gain"] = 1e9
    reranked = analyze_candidates(changed)["groups"][0]
    assert reranked["ranked_candidates"] == group["ranked_candidates"]
    assert reranked["pareto_ids"] == group["pareto_ids"]
    assert not result["confirmation_used_for_ranking"]


def test_rubric_weights_change_choice_without_inheriting_original_confirmation():
    group = analyze_candidates(search_record(), damage_weight=0)["groups"][0]
    assert group["ranked_candidates"][0]["layer"] == 8
    group = analyze_candidates(search_record(), damage_weight=2)["groups"][0]
    assert group["ranked_candidates"][0]["layer"] == 9
    assert not any(
        audit["candidate_id"] == group["ranked_candidates"][0]["id"] for audit in group["confirmation"]
    )


def test_pareto_tradeoff_ties_and_measured_latency_dimension():
    points = [
        {"id": "fast", "target_gain": 1, "regression_damage": 0.2, "evaluation_elapsed_seconds": 1},
        {"id": "slow", "target_gain": 1, "regression_damage": 0.2, "evaluation_elapsed_seconds": 4},
        {"id": "safe", "target_gain": 0.5, "regression_damage": 0, "evaluation_elapsed_seconds": 2},
        {"id": "dominated", "target_gain": 0.1, "regression_damage": 0.5, "evaluation_elapsed_seconds": 5},
    ]
    assert pareto_ids(points) == ["fast", "slow", "safe"]
    assert pareto_ids(points, include_latency=True) == ["fast", "safe"]
    assert rank_candidates(points, latency_weight=0.5)[0]["id"] == "fast"


@pytest.mark.parametrize("value", [True, "2", -1, float("nan"), float("inf"), 10**1000])
def test_latency_never_estimates_or_accepts_invalid_values(value):
    policy = BoundarySearch([8], [0.3])
    entry = observation(elapsed=value)
    with pytest.raises(ValueError):
        policy.observe(Candidate(8, 0.3), entry, "test")
    assert not policy.observed and len(policy.remaining) == 1


def test_missing_timing_is_visible_and_nonzero_latency_weight_rejected():
    record = search_record()
    record["points"][0].pop("elapsed_seconds")
    group = analyze_candidates(record)["groups"][0]
    assert group["candidates"][0]["evaluation_elapsed_seconds"] is None
    assert not group["all_candidates_timed"]
    with pytest.raises(ValueError, match="every candidate"):
        analyze_candidates(record, latency_weight=1)


def test_timing_changes_are_digest_bound_but_excluded_from_numerical_replay():
    first = {"result": {"points": [{"evaluation_elapsed_seconds": 1, "target_gain": 0.5}]}}
    second = copy.deepcopy(first)
    second["result"]["points"][0]["evaluation_elapsed_seconds"] = 4
    assert receipt_digest(first) != receipt_digest(second)
    assert _measurements(first) == _measurements(second)
    second["result"]["points"][0]["target_gain"] = 0.6
    assert _measurements(first) != _measurements(second)


def test_distinct_fingerprints_and_unknown_origins_never_share_a_frontier():
    first, second = search_record(), search_record()
    second["provenance"]["dataset_sha256"] = "different"
    for point in second["points"]:
        point["provenance"]["dataset_sha256"] = "different"
    assert len(analyze_candidates({"searches": [first, second]})["groups"]) == 2
    for source in (first, second):
        source.pop("provenance")
        for point in source["points"]:
            point.pop("provenance")
    groups = analyze_candidates({"searches": [first, second]})["groups"]
    assert len(groups) == 2 and all(not group["compatibility_complete"] for group in groups)


@pytest.mark.parametrize("key,value", [("layer", 9), ("strength", 0.4), ("experiment_id", "other")])
def test_mismatched_confirmation_context_cannot_be_borrowed(key, value):
    record = search_record()
    record["selected_evaluation"][key] = value
    assert not analyze_candidates(record)["groups"][0]["confirmation"]


def test_standalone_evaluation_and_legacy_summary_sweep_are_explicitly_supported():
    evaluation = search_record()["selected_evaluation"]
    group = analyze_candidates(evaluation)["groups"][0]
    assert len(group["candidates"]) == 1 and group["confirmation"]
    assert group["candidates"][0]["evaluation_elapsed_seconds"] is None
    legacy = {
        "experiment_id": "synthetic",
        "sweep": [{"layer": 8, "strength": 0.3, "summary": {"target_gain": 0.5, "regression_damage": 0.1}}],
    }
    assert analyze_candidates(legacy)["groups"][0]["candidates"][0]["target_gain"] == 0.5
    with pytest.raises(ValueError, match="No declared discovery"):
        discovery_metrics({"summary": {"target_gain": 999, "regression_damage": 0}})


def test_differently_worded_case_subsets_cannot_be_ranked_on_one_frontier():
    record = search_record()
    record["points"][1]["rows"][0]["prompt"] = "a different target subset"
    assert len(analyze_candidates(record)["groups"]) == 2


def test_confirmation_only_and_invalid_untrusted_sections_fail_closed():
    with pytest.raises(ValueError, match="Confirmation-only"):
        discovery_metrics({"rows": [{"partition": "confirmation"}], "summary": {}})
    for name in ("catalog", "selection"):
        value = search_record()
        value[name] = None
        with pytest.raises(ValueError):
            analyze_candidates(value)
    value = search_record()
    value["points"] = [observation() for _ in range(301)]
    with pytest.raises(ValueError, match="300"):
        analyze_candidates(value)


def test_empty_confirmation_summary_does_not_claim_an_audit():
    evaluation = observation()
    evaluation["summary"]["confirmation"] = {
        "target_gain": 0,
        "regression_damage": 0,
        "target_count": 0,
        "regression_count": 0,
    }
    assert not analyze_candidates(evaluation)["groups"][0]["confirmation"]
    evaluation["confirmation_included"] = False
    evaluation["summary"]["confirmation"]["target_count"] = 99
    assert not analyze_candidates(evaluation)["groups"][0]["confirmation"]


def test_search_policy_rejects_confirmation_before_consuming_a_candidate():
    policy = BoundarySearch([8], [0.3])
    evaluation = search_record()["selected_evaluation"]
    with pytest.raises(ValueError, match="cannot consume confirmation"):
        policy.observe(Candidate(8, 0.3), evaluation, "sealed control")
    assert not policy.observed and len(policy.remaining) == 1


def test_confirmation_direction_is_bound_without_splitting_cross_layer_frontier():
    record = search_record()
    record["points"][0]["provenance"]["direction_sha256"] = "layer8-direction"
    record["points"][1]["provenance"]["direction_sha256"] = "layer9-direction"
    record["selected_evaluation"]["provenance"]["direction_sha256"] = "layer8-direction"
    groups = analyze_candidates(record)["groups"]
    assert len(groups) == 1 and groups[0]["confirmation"]
    record["selected_evaluation"]["provenance"]["direction_sha256"] = "another-direction"
    assert not analyze_candidates(record)["groups"][0]["confirmation"]


def test_native_boundary_export_keeps_all_settings_and_exact_confirmation():
    boundary = search_record()
    boundary["confirmation"] = boundary.pop("selected_evaluation")
    export = {
        "schema": "gradia-model-research-v1",
        "boundary": boundary,
        "evaluation": copy.deepcopy(boundary["confirmation"]),
    }
    group = analyze_candidates(export)["groups"][0]
    assert len(group["candidates"]) == 2
    assert len(group["confirmation"]) == 1
    assert group["all_candidates_timed"]


def test_finite_inputs_cannot_produce_nonfinite_derived_means_or_utility():
    with pytest.raises(ValueError, match="Derived rubric utility"):
        rank_candidates(
            [{"id": "overflow", "target_gain": 1e308, "regression_damage": 1e308}],
            target_weight=10,
            damage_weight=10,
        )
    value = observation(gain=1e308)
    value["rows"].append(dict(value["rows"][0], id="another-target"))
    with pytest.raises(ValueError, match="Derived target mean"):
        discovery_metrics(value)
    value = observation(damage=1e308)
    value["rows"].append(dict(value["rows"][1], id="another-regression"))
    with pytest.raises(ValueError, match="Derived regression mean"):
        discovery_metrics(value)
