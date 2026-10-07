"""Test restoration and dataset boundaries, plus opt-in real checkpoint integration."""

import math
import os
import threading
import unittest
from unittest.mock import patch

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from gradia_lens.engine import Engine
from gradia_lens.experiments import EXPERIMENTS


class RestorationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.engine = Engine.__new__(Engine)
        self.engine.device = torch.device("cpu")
        self.engine._lock = threading.RLock()
        self.engine.model = GPT2LMHeadModel(
            GPT2Config(
                n_layer=2,
                n_head=2,
                n_embd=16,
                vocab_size=30,
                resid_pdrop=0,
                embd_pdrop=0,
                attn_pdrop=0,
            )
        ).eval()
        self.engine._directions = {"tiny": {0: {"vector": torch.arange(16).float() / 16}}}
        self.ids = torch.tensor([[1, 2, 3, 4]])

    def logits(self):
        with torch.inference_mode():
            return self.engine.model(input_ids=self.ids).logits.clone()

    def test_intervention_changes_last_token_then_restores_exactly(self):
        original = self.logits()
        weights = self.engine.weight_digest()
        with self.engine.intervene("tiny", 0, 0.6):
            altered = self.logits()
            self.assertEqual(self.engine.hook_count(), 1)
        self.assertTrue(torch.equal(original[:, :-1], altered[:, :-1]))
        self.assertFalse(torch.equal(original[:, -1], altered[:, -1]))
        self.assertTrue(torch.equal(original, self.logits()))
        self.assertEqual(self.engine.hook_count(), 0)
        self.assertEqual(weights, self.engine.weight_digest())

    def test_hook_removed_on_caller_exception(self):
        original = self.logits()
        with self.assertRaisesRegex(RuntimeError, "injected failure"):
            with self.engine.intervene("tiny", 0, 0.6):
                raise RuntimeError("injected failure")
        self.assertEqual(self.engine.hook_count(), 0)
        self.assertTrue(torch.equal(original, self.logits()))

    def test_zero_strength_installs_no_hook_and_has_identical_logits(self):
        original = self.logits()
        with self.engine.intervene("tiny", 0, 0):
            self.assertEqual(self.engine.hook_count(), 0)
            self.assertTrue(torch.equal(original, self.logits()))

    def test_rejects_out_of_bounds_and_nonfinite_controls(self):
        for layer, strength in (
            (-1, 0.2),
            (2, 0.2),
            (0, 3),
            (0, math.nan),
            (0, math.inf),
        ):
            with self.assertRaises(ValueError):
                self.engine._validate(layer, strength)

    def test_attention_head_suppression_restores_and_cleans_exception(self):
        original = self.logits()
        with self.engine.suppress_head(0, 0, 1):
            self.assertEqual(self.engine.hook_count(), 1)
            changed = self.logits()
        self.assertFalse(torch.equal(original[:, -1], changed[:, -1]))
        self.assertTrue(torch.equal(original, self.logits()))
        with self.assertRaises(RuntimeError):
            with self.engine.suppress_head(0, 0, 1):
                raise RuntimeError("injected failure")
        self.assertEqual(self.engine.hook_count(), 0)


class DatasetTests(unittest.TestCase):
    def test_disjoint_training_discovery_and_confirmation_prompts(self):
        for experiment in EXPERIMENTS.values():
            train = {prompt for pair in experiment.training_pairs for prompt in pair}
            discovery = {case.prompt for case in experiment.cases if case.partition == "discovery"}
            confirmation = {case.prompt for case in experiment.cases if case.partition == "confirmation"}
            self.assertFalse(train & discovery)
            self.assertFalse(train & confirmation)
            self.assertFalse(discovery & confirmation)
            for split in ("target", "near", "unrelated"):
                self.assertTrue(
                    any(case.split == split and case.partition == "confirmation" for case in experiment.cases)
                )


@unittest.skipUnless(
    os.environ.get("GRADIA_LENS_REAL_MODEL_TEST") == "1",
    "Set GRADIA_LENS_REAL_MODEL_TEST=1 for GPT-2 integration",
)
class RealCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def test_real_checkpoint_all_three_directions_and_restoration(self):
        before = self.engine.weight_digest()
        for experiment in EXPERIMENTS.values():
            result = self.engine.run(experiment.id, 0.35, max_new_tokens=8)
            self.assertTrue(result["restoration"]["exact_logits"])
            self.assertTrue(result["restoration"]["exact_tokens"])
            self.assertEqual(result["restoration"]["max_logit_difference"], 0)
            self.assertNotEqual(result["baseline"]["margin"], result["steered"]["margin"])
            discovered = self.engine.evaluate(experiment.id, 0.35, include_confirmation=False)
            self.assertTrue(all(row["partition"] == "discovery" for row in discovered["rows"]))
        self.assertEqual(before, self.engine.weight_digest())
        self.assertEqual(self.engine.hook_count(), 0)

    def test_matching_regression_prompt_uses_its_expected_lexical_probe(self):
        experiment = EXPERIMENTS["positive-tone"]
        case = next(case for case in experiment.cases if case.split == "near")
        result = self.engine.run(experiment.id, 0.15, prompt=case.prompt, max_new_tokens=2)
        self.assertEqual(result["probe"]["id"], case.id)
        self.assertEqual(result["probe"]["desired_tokens"], list(case.desired))
        self.assertTrue(result["restoration"]["exact_logits"])

    def generated_probes(self):
        return [
            {
                "id": "generated-cafe",
                "prompt": "The meal was spoiled and the cups were dirty. The cafe was",
                "desired_tokens": [" bad", " awful"],
                "undesired_tokens": [" good", " great"],
                "role": "near",
                "origin": "test mutation",
                "seed_id": "tone-near-1",
                "review_required": False,
            },
            {
                "id": "generated-fact",
                "prompt": "In geography class we learned that the capital of France is",
                "desired_tokens": [" Paris"],
                "undesired_tokens": [" Rome", " London"],
                "role": "unrelated",
                "origin": "test paraphrase",
                "seed_id": "tone-unrelated-1",
                "review_required": True,
            },
        ]

    def test_distinct_generated_probes_are_measured_and_restore_without_confirmation(
        self,
    ):
        probes = self.generated_probes()
        before = self.engine.weight_digest()
        result = self.engine.measure_probes("positive-tone", 10, 0.35, probes + [probes[0]])
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(len(result["probe_specs"]), 2)
        self.assertTrue(all(row["review_required"] for row in result["rows"]))
        self.assertTrue(all(row["partition"] == "generated" for row in result["rows"]))
        self.assertTrue(all(row["baseline"] != row["steered"] for row in result["rows"]))
        self.assertGreater(result["rows"][0]["regression_damage"], 0.25)
        self.assertTrue(result["restoration"]["exact_logits"])
        self.assertEqual(result["restoration"]["max_logit_difference"], 0)
        self.assertEqual(result["restoration"]["hooks_after"], 0)
        self.assertEqual(before, self.engine.weight_digest())
        zero = self.engine.measure_probes("positive-tone", 10, 0, probes)
        self.assertTrue(all(row["baseline"] == row["steered"] for row in zero["rows"]))
        self.assertEqual(
            result["provenance"]["extra_probes_sha256"],
            zero["provenance"]["extra_probes_sha256"],
        )

    def test_generated_probe_validation_protects_sealed_cases_and_token_groups(self):
        probe = self.generated_probes()[0]
        invalid = [
            {**probe, "desired_tokens": []},
            {**probe, "undesired_tokens": probe["desired_tokens"]},
            {**probe, "seed_id": "tone-near-4"},
            {**probe, "prompt": EXPERIMENTS["positive-tone"].cases[-1].prompt},
            {**probe, "prompt": " word" * 257},
        ]
        for specification in invalid:
            with self.assertRaises(ValueError):
                self.engine.measure_probes("positive-tone", 10, 0.35, [specification])
        self.assertEqual(self.engine.hook_count(), 0)

    def test_computation_trace_has_actual_attention_shapes_and_zero_control(self):
        prompt = EXPERIMENTS["positive-tone"].default_prompt
        result = self.engine.inspect_computation("positive-tone", prompt, 8, 0.15)
        count = len(result["tokens"])
        self.assertEqual(len(result["layers"]), 12)
        for layer in result["layers"]:
            self.assertEqual(len(layer["heads"]), 12)
            self.assertEqual(len(layer["heads"][0]["baseline_attention"]), count)
            for row in layer["heads"][0]["baseline_attention"]:
                self.assertEqual(len(row), count)
                self.assertAlmostEqual(sum(row), 1, places=5)
            self.assertLess(layer["head_conservation"]["baseline_relative_max_error"], 1e-5)
            self.assertLess(layer["head_conservation"]["steered_relative_max_error"], 1e-5)
            self.assertEqual(len(layer["mlp"]["top_units_by_token"]), count)
        selected = next(probe for probe in result["mediation"] if probe["probe_layer"] == 8)
        self.assertTrue(selected["restores_baseline_logits_exact"])
        self.assertAlmostEqual(selected["fraction_reverted"], 1, places=7)
        self.assertTrue(result["restoration"]["exact_logits"])
        control = self.engine.inspect_computation("positive-tone", prompt, 8, 0)
        self.assertEqual(control["output"]["margin_gain"], 0)
        self.assertTrue(
            all(token["delta_rms"] == 0 for layer in control["layers"] for token in layer["tokens"])
        )
        self.assertTrue(all(probe["fraction_reverted"] is None for probe in control["mediation"]))
        self.assertEqual(self.engine.hook_count(), 0)

    def test_real_parameter_preview_changes_weights_and_restores_even_on_failure(self):
        prompt = EXPERIMENTS["positive-tone"].default_prompt
        result = self.engine.run_weight_experiment("positive-tone", prompt, 8, 0, 0, max_new_tokens=2)
        parameter = result["parameter"]
        self.assertNotEqual(parameter["weight_sha256_before"], parameter["weight_sha256_during"])
        self.assertEqual(parameter["weight_sha256_before"], parameter["weight_sha256_after"])
        self.assertTrue(result["restoration"]["exact_logits"])
        self.assertTrue(result["restoration"]["exact_tokens"])
        control = self.engine.run_weight_experiment("positive-tone", prompt, 8, 0, 1, max_new_tokens=2)
        self.assertEqual(control["baseline"], control["steered"])
        original_logits = self.engine._logits

        def fail_changed_forward(prompts):
            if self.engine.weight_digest() != self.engine.model_info()["weights_sha256"]:
                raise RuntimeError("injected changed-weight failure")
            return original_logits(prompts)

        with patch.object(self.engine, "_logits", side_effect=fail_changed_forward):
            with self.assertRaisesRegex(RuntimeError, "injected changed-weight failure"):
                self.engine.run_weight_experiment("positive-tone", prompt, 8, 0, 0, max_new_tokens=2)
        self.assertEqual(self.engine.weight_digest(), self.engine.model_info()["weights_sha256"])
        self.assertEqual(self.engine.hook_count(), 0)


if __name__ == "__main__":
    unittest.main()
