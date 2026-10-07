"""Public projection integrity and recorded-example utility, without inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gradia_lens import get_experiment
from gradia_lens.rubric import analyze_candidates
from gradia_lens.search import experiment_fingerprint

FIXTURE = Path(__file__).parents[1] / "examples" / "records" / "positive-tone-campaign.json"


def test_curated_campaign_digest_context_and_public_boundary_are_consistent():
    raw = FIXTURE.read_text()
    campaign = json.loads(raw)
    digest = campaign.pop("projection_sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(campaign, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )
    assert campaign["recording_status"] == "real_local_model_measurements_on_hand_written_public_probes"
    assert "/Users/" not in raw and "git_sha" not in raw
    assert "not synthetic" in campaign["probe_design"]
    search = campaign["searches"][0]
    assert search["fingerprint"] == experiment_fingerprint(search["config"])
    assert search["provenance"]["dataset_sha256"] == get_experiment("positive-tone").digest()


def test_public_campaign_retains_real_zero_controls_and_original_confirmation_binding():
    campaign = json.loads(FIXTURE.read_text())
    group = analyze_candidates(campaign)["groups"][0]
    assert group["compatibility_complete"] and group["all_candidates_timed"]
    assert len(group["candidates"]) == 6
    for point in group["candidates"]:
        assert point["evaluation_elapsed_seconds"] >= 0
        assert len(point["rows"]) == 7
        if point["strength"] == 0:
            assert point["target_gain"] == point["regression_damage"] == 0
            assert all(row["margin_gain"] == row["expected_logprob_delta"] == 0 for row in point["rows"])
    assert len(group["confirmation"]) == 1
    audited = group["confirmation"][0]["candidate_id"]
    audited_setting = next(point for point in group["candidates"] if point["id"] == audited)
    assert (audited_setting["layer"], audited_setting["strength"]) == (10, 0.35)


def test_example_rubric_change_exposes_an_actual_gain_damage_tradeoff():
    campaign = json.loads(FIXTURE.read_text())
    low_penalty = analyze_candidates(campaign, damage_weight=0)["groups"][0]
    high_penalty = analyze_candidates(campaign, damage_weight=4)["groups"][0]
    first, second = low_penalty["ranked_candidates"][0], high_penalty["ranked_candidates"][0]
    assert (first["layer"], first["strength"]) == (10, 0.35)
    assert (second["layer"], second["strength"]) == (8, 0.15)
    assert first["target_gain"] > second["target_gain"]
    assert first["regression_damage"] > second["regression_damage"]
    assert all(audit["candidate_id"] != second["id"] for audit in high_penalty["confirmation"])
