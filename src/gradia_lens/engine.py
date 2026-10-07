"""Real GPT-2 residual activation experiments with reversible temporary hooks.

The intervention adds a contrast-fitted direction at the last sequence position
after one transformer block, on every forward pass. Parameters are frozen except during explicitly bounded weight previews, which restore the original bytes in a finally block.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import threading
import time
from contextlib import contextmanager
from pathlib import Path

# Prevent a second model-sized download cache and fork warnings in provenance reads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import __version__
from .experiments import catalog, get_experiment

MODEL_ID = "openai-community/gpt2"
MODEL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
SEED = 17


class Engine:
    """A serialized, deterministic CPU inference engine; no paid services."""

    def __init__(self, device: str = "cpu", cache_dir: str | None = None):
        if device != "cpu":
            raise ValueError("This verified prototype supports deterministic CPU execution only.")
        cache_dir = (
            cache_dir
            or os.environ.get("GRADIA_LENS_CACHE_DIR")
            or os.environ.get("GRADIA_MODEL_RESEARCH_CACHE_DIR")
        )
        self.device = torch.device(device)
        torch.set_num_threads(max(1, min(4, int(os.environ.get("ATLAS_CPU_THREADS", "4")))))
        torch.manual_seed(SEED)
        self._lock = threading.RLock()
        model_cache = (
            Path(cache_dir)
            if cache_dir
            else Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))) / "gradia-lens" / "hub"
        )
        model_cache.mkdir(parents=True, exist_ok=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID, revision=MODEL_REVISION, cache_dir=str(model_cache)
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                cache_dir=str(model_cache),
                use_safetensors=True,
                attn_implementation="eager",
            )
            .to(self.device)
            .eval()
        )
        self.model.requires_grad_(False)
        self._directions: dict = {}
        self._baseline_logits: dict[str, torch.Tensor] = {}
        self._candidate_ids: dict[str, list[int]] = {}
        self._weight_digest = self.weight_digest()
        # Installed code cannot infer build identity from an unrelated checkout.
        # Exact implementation hashes identify this computation edition.
        self._git_sha: str | None = None

    def weight_digest(self) -> str:
        """Hash parameter names, shapes, dtypes, and exact bytes in stable order."""
        digest = hashlib.sha256()
        for name, parameter in self.model.named_parameters():
            digest.update(name.encode())
            digest.update(str(tuple(parameter.shape)).encode())
            digest.update(str(parameter.dtype).encode())
            digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def model_info(self) -> dict:
        return {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "parameters": sum(parameter.numel() for parameter in self.model.parameters()),
            "layers": self.model.config.n_layer,
            "hidden_size": self.model.config.n_embd,
            "device": str(self.device),
            "dtype": str(next(self.model.parameters()).dtype),
            "weights_sha256": self._weight_digest,
            "seed": SEED,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "cpu_threads": torch.get_num_threads(),
        }

    def catalog(self) -> list[dict]:
        return catalog()

    def hook_count(self) -> int:
        return sum(
            len(module._forward_hooks) + len(module._forward_pre_hooks) for module in self.model.modules()
        )

    def _validate(self, layer: int, strength: float):
        if (
            isinstance(layer, bool)
            or not isinstance(layer, int)
            or not 0 <= layer < self.model.config.n_layer
        ):
            raise ValueError(f"Layer must be an integer between 0 and {self.model.config.n_layer - 1}.")
        if (
            isinstance(strength, bool)
            or not isinstance(strength, (int, float))
            or not math.isfinite(strength)
        ):
            raise ValueError("Strength must be a finite number.")
        if not -2.0 <= strength <= 2.0:
            raise ValueError("Strength must be between -2 and 2.")

    def _inputs(self, prompts: list[str]) -> dict[str, torch.Tensor]:
        if any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
            raise ValueError("Prompt must contain text.")
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        if inputs["input_ids"].shape[1] > 256:
            raise ValueError("Prompt exceeds the prototype's 256-token limit.")
        inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
        return {key: value.to(self.device) for key, value in inputs.items()}

    def _logits(self, prompts: list[str]) -> torch.Tensor:
        with torch.inference_mode():
            return self.model(**self._inputs(prompts), use_cache=False).logits[:, -1].detach().cpu()

    def learn_direction(self, experiment_id: str, layer: int) -> dict:
        self._validate(layer, 0.0)
        with self._lock:
            if experiment_id not in self._directions:
                experiment = get_experiment(experiment_id)
                positive = [pair[0] for pair in experiment.training_pairs]
                negative = [pair[1] for pair in experiment.training_pairs]
                captured = {}
                handles = []

                def capture(index):
                    def hook(_module, _args, output):
                        hidden = output[0] if isinstance(output, tuple) else output
                        captured[index] = hidden[:, -1].detach().cpu().clone()

                    return hook

                try:
                    for index, block in enumerate(self.model.transformer.h):
                        handles.append(block.register_forward_hook(capture(index)))
                    self._logits(positive + negative)
                finally:
                    for handle in handles:
                        handle.remove()

                records = {}
                n = len(positive)
                for index, activations in captured.items():
                    positive_mean = activations[:n].mean(0)
                    negative_mean = activations[n:].mean(0)
                    contrast = positive_mean - negative_mean
                    contrast_norm = float(contrast.norm())
                    if contrast_norm == 0:
                        raise RuntimeError("Contrast has zero norm; direction is undefined.")
                    reference_norm = float(activations.norm(dim=1).mean())
                    direction = contrast / contrast_norm * reference_norm
                    records[index] = {
                        "vector": direction,
                        "fingerprint": hashlib.sha256(direction.numpy().tobytes()).hexdigest(),
                        "layer": index,
                        "activation_rms": float(activations.square().mean().sqrt()),
                        "contrast_norm": contrast_norm,
                        "direction_norm": float(direction.norm()),
                        "contrast_cosine": float(
                            torch.nn.functional.cosine_similarity(
                                positive_mean.unsqueeze(0), negative_mean.unsqueeze(0)
                            )
                        ),
                        "reference_norm": reference_norm,
                    }
                self._directions[experiment_id] = records
            return self._directions[experiment_id][layer]

    def inspect(self, experiment_id: str) -> dict:
        with self._lock:
            experiment = get_experiment(experiment_id)
            self.learn_direction(experiment_id, experiment.default_layer)
            return {
                "experiment_id": experiment_id,
                "layers": [
                    {key: value for key, value in record.items() if key != "vector"}
                    for record in self._directions[experiment_id].values()
                ],
                "measurement": "Training-context last-token post-block residual activations; CPU float32.",
                "provenance": self._provenance(experiment_id, experiment.default_layer, 0.0),
            }

    @contextmanager
    def intervene(self, experiment_id: str, layer: int, strength: float):
        """The lock and finally removal also cover exceptions inside the caller."""
        self._validate(layer, strength)
        with self._lock:
            vector = self.learn_direction(experiment_id, layer)["vector"].to(self.device)
            handle = None

            def hook(_module, _args, output):
                hidden = output[0] if isinstance(output, tuple) else output
                replacement = hidden.clone()
                replacement[:, -1, :] += float(strength) * vector.to(hidden.dtype)
                return (replacement,) + output[1:] if isinstance(output, tuple) else replacement

            try:
                if strength != 0:
                    handle = self.model.transformer.h[layer].register_forward_hook(hook)
                yield
            finally:
                if handle is not None:
                    handle.remove()

    def _provenance(self, experiment_id: str, layer: int, strength: float) -> dict:
        experiment = get_experiment(experiment_id)
        direction = self.learn_direction(experiment_id, layer)
        source_digest = hashlib.sha256()
        for source in (Path(__file__), Path(__file__).with_name("experiments.py")):
            source_digest.update(source.read_bytes())
        return {
            "model": self.model_info(),
            "dataset_sha256": experiment.digest(),
            "direction_sha256": direction["fingerprint"],
            "engine_source_sha256": source_digest.hexdigest(),
            "package": {"name": "gradia-lens", "version": __version__},
            "git_sha": self._git_sha,
            "method": "contrast_mean_residual_addition",
            "hook": f"transformer.h.{layer}:post_block:last_sequence_position",
            "layer": layer,
            "strength": float(strength),
            "seed": SEED,
            "normalization": "unit contrast scaled to mean training residual L2 norm",
            "application": "last sequence position on every forward pass, including each generated token",
            "training_pairs": len(experiment.training_pairs),
            "intervention_l2": abs(float(strength)) * direction["direction_norm"],
            "decoding": {"do_sample": False, "algorithm": "greedy", "use_cache": False},
            "metric": "logsumexp of candidate continuation log probabilities; nats",
            "thresholds": {
                "target_improved_margin_gain_nats": 0.05,
                "regression_degraded_logprob_drop_nats": 0.25,
            },
            "model_parameters_frozen": True,
            "dataset_origin": "Versioned hand-written lexical probes; not an external benchmark.",
        }

    def _ids(self, text: str) -> list[int]:
        if text not in self._candidate_ids:
            self._candidate_ids[text] = self.tokenizer.encode(text, add_special_tokens=False)
        return self._candidate_ids[text]

    def _scores(self, prompt: str, logits: torch.Tensor, desired, undesired) -> dict:
        logprobs = logits.log_softmax(-1)

        def score(text):
            ids = self._ids(text)
            total = float(logprobs[ids[0]])
            # Multi-token candidates use explicit teacher-forced continuation forwards.
            for index in range(1, len(ids)):
                prefix = prompt + self.tokenizer.decode(ids[:index])
                additional = self._logits([prefix])[0].log_softmax(-1)
                total += float(additional[ids[index]])
            return total

        def aggregate(tokens):
            values = [score(token) for token in tokens]
            return float(torch.tensor(values, dtype=torch.float64).logsumexp(0))

        desired_logprob = aggregate(desired)
        undesired_logprob = aggregate(undesired)
        return {
            "desired_logprob": desired_logprob,
            "undesired_logprob": undesired_logprob,
            "margin": desired_logprob - undesired_logprob,
        }

    def _generate(self, prompt: str, max_new_tokens: int) -> dict:
        ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        initial_size = ids.shape[1]
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = self.model(input_ids=ids, use_cache=False).logits[0, -1]
                next_token = logits.argmax().reshape(1, 1)
                ids = torch.cat((ids, next_token), dim=-1)
                if int(next_token.item()) == self.tokenizer.eos_token_id:
                    break
        tokens = ids[0, initial_size:].cpu().tolist()
        return {
            "text": self.tokenizer.decode(ids[0], skip_special_tokens=True),
            "continuation": self.tokenizer.decode(tokens, skip_special_tokens=True),
            "tokens": tokens,
        }

    def run(
        self,
        experiment_id: str,
        strength: float,
        layer: int | None = None,
        prompt: str | None = None,
        max_new_tokens: int = 32,
    ) -> dict:
        with self._lock:
            started = time.perf_counter()
            experiment = get_experiment(experiment_id)
            layer = experiment.default_layer if layer is None else layer
            prompt = experiment.default_prompt if prompt is None else prompt
            self._validate(layer, strength)
            if (
                isinstance(max_new_tokens, bool)
                or not isinstance(max_new_tokens, int)
                or not 1 <= max_new_tokens <= 64
            ):
                raise ValueError("max_new_tokens must be an integer between 1 and 64.")
            probe = next(
                (case for case in experiment.cases if case.prompt == prompt),
                next(case for case in experiment.cases if case.split == "target"),
            )
            original_hooks = self.hook_count()
            baseline_logits = self._logits([prompt])[0]
            baseline = self._generate(prompt, max_new_tokens)
            baseline.update(self._scores(prompt, baseline_logits, probe.desired, probe.undesired))
            baseline["top_tokens"] = self._top_tokens(baseline_logits)
            with self.intervene(experiment_id, layer, strength):
                steered_logits = self._logits([prompt])[0]
                steered = self._generate(prompt, max_new_tokens)
                steered.update(self._scores(prompt, steered_logits, probe.desired, probe.undesired))
                steered["top_tokens"] = self._top_tokens(steered_logits)
            restored_logits = self._logits([prompt])[0]
            restored = self._generate(prompt, max_new_tokens)
            restoration = {
                "exact_logits": bool(torch.equal(baseline_logits, restored_logits)),
                "exact_tokens": baseline["tokens"] == restored["tokens"],
                "hooks_after": self.hook_count(),
                "hooks_before": original_hooks,
                "max_logit_difference": float((baseline_logits - restored_logits).abs().max()),
                "weights_sha256": self._weight_digest,
            }
            if (
                not restoration["exact_logits"]
                or not restoration["exact_tokens"]
                or self.hook_count() != original_hooks
            ):
                raise RuntimeError("Restoration invariant failed.")
            provenance = self._provenance(experiment_id, layer, strength)
            provenance["decoding"]["max_new_tokens"] = max_new_tokens
            return {
                "experiment_id": experiment_id,
                "prompt": prompt,
                "layer": layer,
                "strength": float(strength),
                "max_new_tokens": max_new_tokens,
                "probe": {
                    "id": probe.id,
                    "split": probe.split,
                    "partition": probe.partition,
                    "label": probe.label,
                    "desired_tokens": list(probe.desired),
                    "undesired_tokens": list(probe.undesired),
                },
                "baseline": baseline,
                "steered": steered,
                "restoration": restoration,
                "provenance": provenance,
                "elapsed_seconds": time.perf_counter() - started,
            }

    def _top_tokens(self, logits: torch.Tensor) -> list[dict]:
        values, indices = logits.log_softmax(-1).topk(8)
        return [
            {"token": self.tokenizer.decode([index]), "logprob": float(value)}
            for index, value in zip(indices.tolist(), values.tolist())  # noqa: B905 — optional Python 3.9 worker
        ]

    @staticmethod
    def _summary(rows: list[dict]) -> dict:
        targets = [row for row in rows if row["split"] == "target"]
        regressions = [row for row in rows if row["split"] != "target"]

        def mean(entries, key):
            return sum(row[key] for row in entries) / len(entries) if entries else 0.0

        return {
            "target_gain": mean(targets, "margin_gain"),
            "regression_damage": mean(regressions, "regression_damage"),
            "near_damage": mean(
                [row for row in regressions if row["split"] == "near"],
                "regression_damage",
            ),
            "unrelated_damage": mean(
                [row for row in regressions if row["split"] == "unrelated"],
                "regression_damage",
            ),
            "target_improved": sum(row["margin_gain"] > 0.05 for row in targets),
            "target_count": len(targets),
            "regression_degraded": sum(row["degraded"] for row in regressions),
            "regression_count": len(regressions),
        }

    def evaluate(
        self,
        experiment_id: str,
        strength: float,
        layer: int | None = None,
        include_generations: bool = False,
        include_confirmation: bool = True,
    ) -> dict:
        with self._lock:
            started = time.perf_counter()
            experiment = get_experiment(experiment_id)
            layer = experiment.default_layer if layer is None else layer
            self._validate(layer, strength)
            cases = [
                case for case in experiment.cases if include_confirmation or case.partition == "discovery"
            ]
            prompts = [case.prompt for case in cases]
            # Cache only unintervened baseline tensors, keyed by dataset identity.
            cache_key = experiment.digest() + (":all" if include_confirmation else ":discovery")
            if cache_key not in self._baseline_logits:
                self._baseline_logits[cache_key] = self._logits(prompts)
            original_logits = self._baseline_logits[cache_key]
            baseline_scores = [
                self._scores(case.prompt, original_logits[index], case.desired, case.undesired)
                for index, case in enumerate(cases)
            ]
            with self.intervene(experiment_id, layer, strength):
                changed_logits = self._logits(prompts)
                changed_scores = [
                    self._scores(case.prompt, changed_logits[index], case.desired, case.undesired)
                    for index, case in enumerate(cases)
                ]
            rows = []
            for case, baseline, steered in zip(cases, baseline_scores, changed_scores):  # noqa: B905 — optional Python 3.9 worker
                expected_delta = steered["desired_logprob"] - baseline["desired_logprob"]
                damage = max(0.0, -expected_delta) if case.split != "target" else 0.0
                rows.append(
                    {
                        "id": case.id,
                        "split": case.split,
                        "partition": case.partition,
                        "prompt": case.prompt,
                        "label": case.label,
                        "desired_tokens": list(case.desired),
                        "undesired_tokens": list(case.undesired),
                        "baseline": baseline,
                        "steered": steered,
                        "margin_gain": steered["margin"] - baseline["margin"],
                        "expected_logprob_delta": expected_delta,
                        "regression_damage": damage,
                        "degraded": damage > 0.25,
                    }
                )
            discovery = self._summary([row for row in rows if row["partition"] == "discovery"])
            confirmation = self._summary([row for row in rows if row["partition"] == "confirmation"])
            result = {
                "experiment_id": experiment_id,
                "layer": layer,
                "strength": float(strength),
                "confirmation_included": include_confirmation,
                "rows": rows,
                "summary": {
                    **discovery,
                    "discovery": discovery,
                    "confirmation": confirmation,
                },
                "provenance": self._provenance(experiment_id, layer, strength),
                "elapsed_seconds": time.perf_counter() - started,
            }
            if include_generations:
                result["comparison"] = self.run(experiment_id, strength, layer)
            return result

    def _normalize_probes(self, experiment_id: str, probes: list[dict]) -> list[dict]:
        """Validate generated proxy specifications and protect sealed seed cases."""
        experiment = get_experiment(experiment_id)
        if not isinstance(probes, list) or not 1 <= len(probes) <= 24:
            raise ValueError("Provide between 1 and 24 probe specifications.")
        sealed_ids = {case.id for case in experiment.cases if case.partition == "confirmation"}
        sealed_prompts = {case.prompt for case in experiment.cases if case.partition == "confirmation"}
        normalized = []
        identifiers = set()
        seen_prompts: dict[str, tuple[str, tuple[str, ...], tuple[str, ...]]] = {}
        for probe in probes:
            if not isinstance(probe, dict):
                raise ValueError("Each probe must be an object.")
            for key in ("id", "prompt", "role", "origin", "seed_id"):
                if not isinstance(probe.get(key), str) or not probe[key].strip():
                    raise ValueError(f"Probe {key} must contain text.")
            if probe["role"] not in ("target", "near", "unrelated", "regression"):
                raise ValueError("Probe role must be target, near, unrelated, or regression.")
            if probe["seed_id"] in sealed_ids or probe["prompt"] in sealed_prompts:
                raise ValueError("Confirmation cases cannot seed generated failure discovery.")
            groups = {}
            for key in ("desired_tokens", "undesired_tokens"):
                tokens = probe.get(key)
                if not isinstance(tokens, list) or not 1 <= len(tokens) <= 12:
                    raise ValueError(f"Probe {key} requires between 1 and 12 continuations.")
                if any(not isinstance(token, str) or not token.strip() for token in tokens):
                    raise ValueError(f"Probe {key} must contain nonempty text strings.")
                groups[key] = list(dict.fromkeys(tokens))
                if any(not self._ids(token) or len(self._ids(token)) > 8 for token in groups[key]):
                    raise ValueError("Each probe continuation must have between 1 and 8 tokens.")
            if set(groups["desired_tokens"]) & set(groups["undesired_tokens"]):
                raise ValueError("Desired and undesired probe continuations must be disjoint.")
            prompt_size = len(self.tokenizer.encode(probe["prompt"], add_special_tokens=False))
            continuation_size = max(len(self._ids(token)) for tokens in groups.values() for token in tokens)
            if prompt_size + continuation_size > 256:
                raise ValueError("Probe prompt plus continuation exceeds 256 tokens.")
            canonical = {key: probe[key] for key in ("id", "prompt", "role", "origin", "seed_id")}
            canonical.update(groups)
            canonical["review_required"] = True
            identity = (
                canonical["role"],
                tuple(canonical["desired_tokens"]),
                tuple(canonical["undesired_tokens"]),
            )
            if canonical["prompt"] in seen_prompts:
                if seen_prompts[canonical["prompt"]] != identity:
                    raise ValueError("Duplicate probe prompts have conflicting roles or continuations.")
                continue
            if canonical["id"] in identifiers:
                raise ValueError("Distinct probe prompts require unique identifiers.")
            identifiers.add(canonical["id"])
            seen_prompts[canonical["prompt"]] = identity
            normalized.append(canonical)
        return normalized

    def measure_probes(self, experiment_id: str, layer: int, strength: float, probes: list[dict]) -> dict:
        """Measure proposed failure probes; lexical expectations need human review.

        New probes never enter direction fitting or the fixed confirmation audit.
        The returned canonical specifications allow exact challenge replay.
        """
        with self._lock:
            started = time.perf_counter()
            self._validate(layer, strength)
            specifications = self._normalize_probes(experiment_id, probes)
            original_hooks = self.hook_count()
            if original_hooks:
                raise RuntimeError("Generated probe measurement requires an unmodified model.")
            prompts = [probe["prompt"] for probe in specifications]
            baseline_logits = self._logits(prompts)
            baseline_scores = [
                self._scores(
                    probe["prompt"],
                    baseline_logits[index],
                    probe["desired_tokens"],
                    probe["undesired_tokens"],
                )
                for index, probe in enumerate(specifications)
            ]
            with self.intervene(experiment_id, layer, strength):
                steered_logits = self._logits(prompts)
                steered_scores = [
                    self._scores(
                        probe["prompt"],
                        steered_logits[index],
                        probe["desired_tokens"],
                        probe["undesired_tokens"],
                    )
                    for index, probe in enumerate(specifications)
                ]
            restored_logits = self._logits(prompts)
            restoration = {
                "exact_logits": bool(torch.equal(baseline_logits, restored_logits)),
                "hooks_before": original_hooks,
                "hooks_after": self.hook_count(),
                "max_logit_difference": float((baseline_logits - restored_logits).abs().max()),
                "weights_sha256": self._weight_digest,
            }
            if not restoration["exact_logits"] or self.hook_count() != original_hooks:
                raise RuntimeError("Generated probe restoration invariant failed.")
            rows = []
            for probe, baseline, steered in zip(specifications, baseline_scores, steered_scores):  # noqa: B905 — optional Python 3.9 worker
                expected_delta = steered["desired_logprob"] - baseline["desired_logprob"]
                damage = max(0.0, -expected_delta) if probe["role"] != "target" else 0.0
                rows.append(
                    {
                        **probe,
                        "split": probe["role"],
                        "partition": "generated",
                        "label": "Unreviewed generated lexical probe",
                        "baseline": baseline,
                        "steered": steered,
                        "margin_gain": steered["margin"] - baseline["margin"],
                        "expected_logprob_delta": expected_delta,
                        "regression_damage": damage,
                        "degraded": damage > 0.25,
                        "measurement_kind": "unreviewed_lexical_proxy",
                    }
                )
            provenance = self._provenance(experiment_id, layer, strength)
            provenance["extra_probes_sha256"] = hashlib.sha256(
                json.dumps(
                    specifications,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            provenance["extra_probes_count"] = len(specifications)
            return {
                "experiment_id": experiment_id,
                "layer": layer,
                "strength": float(strength),
                "probe_specs": specifications,
                "rows": rows,
                "summary": self._summary(rows),
                "restoration": restoration,
                "provenance": provenance,
                "review_required": True,
                "measurement_kind": "unreviewed_lexical_proxy",
                "interpretation_limit": "Generated expectations require human review; a likelihood drop is a candidate failure, not an independently verified behavioral error.",
                "confirmation_used_for_generation": False,
                "elapsed_seconds": time.perf_counter() - started,
            }

    def _probe_for_prompt(self, experiment_id, prompt):
        experiment = get_experiment(experiment_id)
        return next(
            (case for case in experiment.cases if case.prompt == prompt),
            next(case for case in experiment.cases if case.split == "target"),
        )

    def _first_token_scores(self, logits, desired, undesired):
        """A next-token readout, including for candidates that have multiple BPE tokens."""
        logprobs = logits.log_softmax(-1)
        desired_ids = list(dict.fromkeys(self._ids(text)[0] for text in desired))
        undesired_ids = list(dict.fromkeys(self._ids(text)[0] for text in undesired))
        positive = float(logprobs[desired_ids].logsumexp(0))
        negative = float(logprobs[undesired_ids].logsumexp(0))
        return {
            "desired_logprob": positive,
            "undesired_logprob": negative,
            "margin": positive - negative,
        }

    def _capture_computation(self, prompt):
        inputs = self._inputs([prompt])
        captured = {
            "residuals": {},
            "mlp": {},
            "projection_inputs": {},
            "projection_outputs": {},
            "mlp_projection_inputs": {},
        }
        handles = []

        def capture(group, index, pre=False):
            def hook(_module, args, output=None):
                tensor = args[0] if pre else (output[0] if isinstance(output, tuple) else output)
                captured[group][index] = tensor[0].detach().cpu().clone()

            return hook

        try:
            for index, block in enumerate(self.model.transformer.h):
                handles.append(block.register_forward_hook(capture("residuals", index)))
                handles.append(block.mlp.register_forward_hook(capture("mlp", index)))
                handles.append(
                    block.mlp.c_proj.register_forward_pre_hook(capture("mlp_projection_inputs", index, True))
                )
                handles.append(
                    block.attn.c_proj.register_forward_pre_hook(capture("projection_inputs", index, True))
                )
                handles.append(block.attn.c_proj.register_forward_hook(capture("projection_outputs", index)))
            with torch.inference_mode():
                output = self.model(**inputs, use_cache=False, output_attentions=True)
            captured["logits"] = output.logits[0, -1].detach().cpu().clone()
            captured["attention"] = [tensor[0].detach().cpu().clone() for tensor in output.attentions]
            captured["input_ids"] = inputs["input_ids"][0].cpu().tolist()
            return captured
        finally:
            for handle in handles:
                handle.remove()

    @contextmanager
    def _restore_residual(self, probe_layer, baseline_residual):
        def patch(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            restored = hidden.clone()
            restored[:, -1] = baseline_residual.to(hidden.device, hidden.dtype)
            return (restored,) + output[1:] if isinstance(output, tuple) else restored

        handle = self.model.transformer.h[probe_layer].register_forward_hook(patch)
        try:
            yield
        finally:
            handle.remove()

    def inspect_computation(
        self,
        experiment_id: str,
        prompt: str,
        layer: int,
        strength: float,
        max_tokens: int = 48,
    ) -> dict:
        """Measure a partial computation trace and whole-residual restoration probes.

        Attention weights describe mixing, not causal importance. Intermediate
        readouts apply the final norm/unembedding to unfinished residual states.
        Whole-residual patches demonstrate reversibility at coarse layer boundaries.
        """
        with self._lock:
            started = time.perf_counter()
            self._validate(layer, strength)
            if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 48:
                raise ValueError("Trace token budget must be between 1 and 48.")
            inputs = self._inputs([prompt])
            if inputs["input_ids"].shape[1] > max_tokens:
                raise ValueError(f"Trace prompt exceeds the {max_tokens}-token budget.")
            if self.hook_count():
                raise RuntimeError("Computation inspection requires an unmodified model.")
            probe = self._probe_for_prompt(experiment_id, prompt)
            baseline = self._capture_computation(prompt)
            # Install steering before capture so the selected post-block state is observed after addition.
            with self.intervene(experiment_id, layer, strength):
                steered = self._capture_computation(prompt)
            restored = self._capture_computation(prompt)
            restoration = {
                "exact_logits": bool(torch.equal(baseline["logits"], restored["logits"])),
                "hooks_after": self.hook_count(),
                "max_logit_difference": float((baseline["logits"] - restored["logits"]).abs().max()),
            }
            if not restoration["exact_logits"] or restoration["hooks_after"]:
                raise RuntimeError("Computation trace restoration invariant failed.")
            original_score = self._first_token_scores(baseline["logits"], probe.desired, probe.undesired)
            changed_score = self._first_token_scores(steered["logits"], probe.desired, probe.undesired)
            gain = changed_score["margin"] - original_score["margin"]
            layer_records = []
            for index, block in enumerate(self.model.transformer.h):
                original = baseline["residuals"][index]
                changed = steered["residuals"][index]
                difference = changed - original
                top_indices = difference.abs().topk(min(8, difference.shape[-1]), dim=-1).indices
                token_records = []
                for token_index in range(original.shape[0]):
                    token_records.append(
                        {
                            "index": token_index,
                            "baseline_rms": float(original[token_index].square().mean().sqrt()),
                            "steered_rms": float(changed[token_index].square().mean().sqrt()),
                            "delta_rms": float(difference[token_index].square().mean().sqrt()),
                            "residual_cosine": float(
                                torch.nn.functional.cosine_similarity(
                                    original[token_index].unsqueeze(0),
                                    changed[token_index].unsqueeze(0),
                                )
                            ),
                            "top_changed_dimensions": [
                                {
                                    "index": coordinate,
                                    "baseline": float(original[token_index, coordinate]),
                                    "steered": float(changed[token_index, coordinate]),
                                    "delta": float(difference[token_index, coordinate]),
                                }
                                for coordinate in top_indices[token_index].tolist()
                            ],
                        }
                    )
                number_heads = block.attn.num_heads
                width = baseline["projection_inputs"][index].shape[-1] // number_heads
                weight = block.attn.c_proj.weight.detach().cpu().reshape(number_heads, width, -1)
                head_vectors = []
                conservation = {}
                for name, snapshot in (("baseline", baseline), ("steered", steered)):
                    contexts = snapshot["projection_inputs"][index].reshape(-1, number_heads, width)
                    contributions = torch.einsum("shd,hdo->sho", contexts, weight)
                    reconstructed = contributions.sum(dim=1) + block.attn.c_proj.bias.detach().cpu()
                    conservation[f"{name}_max_abs_error"] = float(
                        (reconstructed - snapshot["projection_outputs"][index]).abs().max()
                    )
                    head_vectors.append(contributions)
                    conservation[f"{name}_relative_max_error"] = conservation[f"{name}_max_abs_error"] / max(
                        1e-12, float(snapshot["projection_outputs"][index].abs().max())
                    )
                original_heads, changed_heads = head_vectors
                head_records = []
                for head in range(number_heads):
                    original_attention = baseline["attention"][index][head]
                    changed_attention = steered["attention"][index][head]

                    def entropy(values):
                        return float(-(values * values.clamp_min(1e-12).log()).sum(-1).mean())

                    first = original_heads[:, head]
                    second = changed_heads[:, head]
                    head_records.append(
                        {
                            "head": head,
                            "baseline_contribution_rms": float(first.square().mean().sqrt()),
                            "steered_contribution_rms": float(second.square().mean().sqrt()),
                            "delta_contribution_rms": float((second - first).square().mean().sqrt()),
                            "baseline_contribution_rms_by_token": first.square().mean(-1).sqrt().tolist(),
                            "steered_contribution_rms_by_token": second.square().mean(-1).sqrt().tolist(),
                            "delta_contribution_rms_by_token": (second - first)
                            .square()
                            .mean(-1)
                            .sqrt()
                            .tolist(),
                            "baseline_attention": original_attention.tolist(),
                            "steered_attention": changed_attention.tolist(),
                            "mean_attention_l1_delta": float(
                                (changed_attention - original_attention).abs().sum(-1).mean()
                            ),
                            "baseline_entropy": entropy(original_attention),
                            "steered_entropy": entropy(changed_attention),
                        }
                    )
                readouts = []
                with torch.inference_mode():
                    for state in (original, changed):
                        logits = self.model.lm_head(self.model.transformer.ln_f(state.to(self.device))).cpu()
                        readouts.append(
                            [
                                self._first_token_scores(row, probe.desired, probe.undesired)["margin"]
                                for row in logits
                            ]
                        )
                original_mlp = baseline["mlp"][index]
                changed_mlp = steered["mlp"][index]
                original_units = baseline["mlp_projection_inputs"][index]
                changed_units = steered["mlp_projection_inputs"][index]
                row_rms = block.mlp.c_proj.weight.detach().cpu().square().mean(-1).sqrt()
                unit_indices = (changed_units.abs() * row_rms).topk(8, dim=-1).indices
                mlp_units = []
                for token_index in range(original_units.shape[0]):
                    mlp_units.append(
                        [
                            {
                                "index": unit,
                                "baseline_activation": float(original_units[token_index, unit]),
                                "steered_activation": float(changed_units[token_index, unit]),
                                "delta_activation": float(
                                    changed_units[token_index, unit] - original_units[token_index, unit]
                                ),
                                "baseline_contribution_rms": float(
                                    original_units[token_index, unit].abs() * row_rms[unit]
                                ),
                                "steered_contribution_rms": float(
                                    changed_units[token_index, unit].abs() * row_rms[unit]
                                ),
                                "delta_contribution_rms": float(
                                    (
                                        changed_units[token_index, unit] - original_units[token_index, unit]
                                    ).abs()
                                    * row_rms[unit]
                                ),
                            }
                            for unit in unit_indices[token_index].tolist()
                        ]
                    )
                layer_records.append(
                    {
                        "layer": index,
                        "tokens": token_records,
                        "heads": head_records,
                        "mlp": {
                            "top_units_by_token": mlp_units,
                            "baseline_rms_by_token": original_mlp.square().mean(-1).sqrt().tolist(),
                            "steered_rms_by_token": changed_mlp.square().mean(-1).sqrt().tolist(),
                            "delta_rms_by_token": (changed_mlp - original_mlp)
                            .square()
                            .mean(-1)
                            .sqrt()
                            .tolist(),
                        },
                        "head_conservation": conservation,
                        "intermediate_readout": {
                            "method": "final_norm_unembedding_first_token_logit_lens",
                            "baseline_margin": readouts[0][-1],
                            "steered_margin": readouts[1][-1],
                            "delta_margin": readouts[1][-1] - readouts[0][-1],
                            "baseline_margin_by_token": readouts[0],
                            "steered_margin_by_token": readouts[1],
                        },
                    }
                )
            available = list(range(layer, self.model.config.n_layer))
            selected = (
                available
                if len(available) <= 8
                else [available[int(round(value))] for value in np.linspace(0, len(available) - 1, 8)]
            )
            mediation = []
            for probe_layer in selected:
                with self.intervene(experiment_id, layer, strength):
                    with self._restore_residual(probe_layer, baseline["residuals"][probe_layer][-1]):
                        patched_logits = self._logits([prompt])[0]
                patched = self._first_token_scores(patched_logits, probe.desired, probe.undesired)
                mediation.append(
                    {
                        "probe_layer": probe_layer,
                        "patched_margin": patched["margin"],
                        "margin_gain": patched["margin"] - original_score["margin"],
                        "fraction_reverted": (
                            (changed_score["margin"] - patched["margin"]) / gain if abs(gain) > 1e-8 else None
                        ),
                        "restores_baseline_logits_exact": bool(
                            torch.equal(baseline["logits"], patched_logits)
                        ),
                        "intervention": "restore whole baseline last-token post-block residual after steering",
                    }
                )
            provenance = self._provenance(experiment_id, layer, strength)
            provenance["trace"] = {
                "token_budget": max_tokens,
                "token_count": len(baseline["input_ids"]),
                "causal_probe_count": len(mediation),
                "forward_passes": 3 + len(mediation),
                "forward_pass_scope": "baseline, steered, restored and causal patches; excludes cold direction fitting",
                "output_metric": "first-token candidate-group log probability margin; nats",
                "head_contributions": "each head context multiplied by its actual output-projection weight slice; shared bias excluded",
                "attention_entropy_units": "nats",
                "coordinate_semantics": "raw residual coordinates; no semantic labels",
            }
            provenance["metric"] = provenance["trace"]["output_metric"]
            return {
                "experiment_id": experiment_id,
                "prompt": prompt,
                "layer": layer,
                "strength": float(strength),
                "tokens": [
                    {
                        "index": index,
                        "id": token,
                        "text": self.tokenizer.decode([token]),
                    }
                    for index, token in enumerate(baseline["input_ids"])
                ],
                "layers": layer_records,
                "output": {
                    "baseline": {
                        **original_score,
                        "top_tokens": self._top_tokens(baseline["logits"]),
                    },
                    "steered": {
                        **changed_score,
                        "top_tokens": self._top_tokens(steered["logits"]),
                    },
                    "margin_gain": gain,
                    "logits_delta_l2": float((steered["logits"] - baseline["logits"]).norm()),
                },
                "mediation": mediation,
                "restoration": restoration,
                "provenance": provenance,
                "interpretation_limit": "A partial measured computation map. Attention and contribution magnitudes are descriptive; whole-residual restoration probes do not identify a unique circuit or causal head.",
                "elapsed_seconds": time.perf_counter() - started,
            }

    @contextmanager
    def suppress_head(self, layer: int, head: int, fraction: float = 1.0):
        """Suppress one attention head's context contribution at the last position."""
        self._validate(layer, 0)
        number_heads = self.model.transformer.h[layer].attn.num_heads
        if isinstance(head, bool) or not isinstance(head, int) or not 0 <= head < number_heads:
            raise ValueError(f"Head index must be between 0 and {number_heads - 1}.")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(fraction)
            or not 0 <= fraction <= 1
        ):
            raise ValueError("Ablation fraction must be between 0 and 1.")
        with self._lock:
            module = self.model.transformer.h[layer].attn.c_proj
            width = module.weight.shape[0] // number_heads

            def patch(_module, args):
                replacement = args[0].clone()
                replacement[:, -1, head * width : (head + 1) * width] *= 1 - float(fraction)
                return (replacement,) + args[1:]

            handle = module.register_forward_pre_hook(patch) if fraction else None
            try:
                yield
            finally:
                if handle is not None:
                    handle.remove()

    def _physical_provenance(self, experiment_id, layer, method):
        provenance = self._provenance(experiment_id, layer, 0)
        for key in (
            "direction_sha256",
            "normalization",
            "training_pairs",
            "intervention_l2",
        ):
            provenance.pop(key, None)
        provenance["method"] = method
        return provenance

    def run_head_ablation(
        self,
        experiment_id: str,
        layer: int,
        head: int,
        prompt: str | None = None,
        fraction: float = 1.0,
        max_new_tokens: int = 16,
    ) -> dict:
        with self._lock:
            started = time.perf_counter()
            prompt = get_experiment(experiment_id).default_prompt if prompt is None else prompt
            self._validate(layer, 0)
            if (
                isinstance(max_new_tokens, bool)
                or not isinstance(max_new_tokens, int)
                or not 1 <= max_new_tokens <= 64
            ):
                raise ValueError("max_new_tokens must be between 1 and 64.")
            if self.hook_count():
                raise RuntimeError("Head ablation requires an unmodified model.")
            probe = self._probe_for_prompt(experiment_id, prompt)
            provenance = self._physical_provenance(
                experiment_id, layer, "temporary_attention_head_output_ablation"
            )
            original_logits = self._logits([prompt])[0]
            baseline = self._generate(prompt, max_new_tokens)
            baseline.update(self._scores(prompt, original_logits, probe.desired, probe.undesired))
            baseline["top_tokens"] = self._top_tokens(original_logits)
            with self.suppress_head(layer, head, fraction):
                changed_logits = self._logits([prompt])[0]
                steered = self._generate(prompt, max_new_tokens)
                steered.update(self._scores(prompt, changed_logits, probe.desired, probe.undesired))
                steered["top_tokens"] = self._top_tokens(changed_logits)
            restored_logits = self._logits([prompt])[0]
            restored = self._generate(prompt, max_new_tokens)
            restoration = {
                "exact_logits": bool(torch.equal(original_logits, restored_logits)),
                "exact_tokens": baseline["tokens"] == restored["tokens"],
                "hooks_after": self.hook_count(),
                "max_logit_difference": float((original_logits - restored_logits).abs().max()),
            }
            if (
                not restoration["exact_logits"]
                or not restoration["exact_tokens"]
                or restoration["hooks_after"]
            ):
                raise RuntimeError("Head ablation restoration invariant failed.")
            provenance.update(
                hook=f"transformer.h.{layer}.attn.c_proj:pre:last_position:head:{head}",
                head=head,
                fraction=float(fraction),
                application="Scale one head's pre-projection context slice at last position on every forward pass",
            )
            provenance["decoding"]["max_new_tokens"] = max_new_tokens
            return {
                "experiment_id": experiment_id,
                "prompt": prompt,
                "layer": layer,
                "head": head,
                "fraction": float(fraction),
                "max_new_tokens": max_new_tokens,
                "baseline": baseline,
                "steered": steered,
                "restoration": restoration,
                "probe": {
                    "id": probe.id,
                    "label": probe.label,
                    "desired_tokens": list(probe.desired),
                    "undesired_tokens": list(probe.undesired),
                },
                "provenance": provenance,
                "elapsed_seconds": time.perf_counter() - started,
            }

    def run_weight_experiment(
        self,
        experiment_id: str,
        prompt: str,
        layer: int,
        neuron: int,
        gain: float,
        max_new_tokens: int = 16,
    ) -> dict:
        """Preview one MLP projection parameter row; exact snapshot restoration only."""
        with self._lock:
            started = time.perf_counter()
            self._validate(layer, 0)
            if (
                isinstance(max_new_tokens, bool)
                or not isinstance(max_new_tokens, int)
                or not 1 <= max_new_tokens <= 64
            ):
                raise ValueError("max_new_tokens must be between 1 and 64.")
            weight = self.model.transformer.h[layer].mlp.c_proj.weight
            if isinstance(neuron, bool) or not isinstance(neuron, int) or not 0 <= neuron < weight.shape[0]:
                raise ValueError(f"MLP row index must be between 0 and {weight.shape[0] - 1}.")
            if (
                isinstance(gain, bool)
                or not isinstance(gain, (int, float))
                or not math.isfinite(gain)
                or not 0 <= gain <= 2
            ):
                raise ValueError("Weight gain must be between 0 and 2.")
            if self.hook_count():
                raise RuntimeError("Parameter preview requires an unmodified model.")
            before_digest = self.weight_digest()
            if before_digest != self._weight_digest:
                raise RuntimeError("Parameter preview requires the original checkpoint weights.")
            probe = self._probe_for_prompt(experiment_id, prompt)
            provenance = self._physical_provenance(
                experiment_id, layer, "temporary_mlp_projection_parameter_row_scaling"
            )
            original_logits = self._logits([prompt])[0]
            baseline = self._generate(prompt, max_new_tokens)
            baseline.update(self._scores(prompt, original_logits, probe.desired, probe.undesired))
            baseline["top_tokens"] = self._top_tokens(original_logits)
            original_row = weight[neuron].detach().clone()
            try:
                with torch.no_grad():
                    weight[neuron].copy_(original_row * float(gain))
                during_digest = self.weight_digest()
                changed_logits = self._logits([prompt])[0]
                steered = self._generate(prompt, max_new_tokens)
                steered.update(self._scores(prompt, changed_logits, probe.desired, probe.undesired))
                steered["top_tokens"] = self._top_tokens(changed_logits)
            finally:
                with torch.no_grad():
                    weight[neuron].copy_(original_row)
                after_digest = self.weight_digest()
                if after_digest != before_digest:
                    raise RuntimeError("Parameter snapshot restoration failed.")
            restored_logits = self._logits([prompt])[0]
            restored = self._generate(prompt, max_new_tokens)
            restoration = {
                "exact_logits": bool(torch.equal(original_logits, restored_logits)),
                "exact_tokens": baseline["tokens"] == restored["tokens"],
                "hooks_after": self.hook_count(),
                "weights_unchanged": before_digest == after_digest,
                "max_logit_difference": float((original_logits - restored_logits).abs().max()),
            }
            if (
                not restoration["exact_logits"]
                or not restoration["exact_tokens"]
                or restoration["hooks_after"]
            ):
                raise RuntimeError("Parameter preview restoration invariant failed.")
            parameter = {
                "name": f"transformer.h.{layer}.mlp.c_proj.weight",
                "neuron": neuron,
                "gain": float(gain),
                "shape": list(weight.shape),
                "weight_sha256_before": before_digest,
                "weight_sha256_during": during_digest,
                "weight_sha256_after": after_digest,
                "original_row_rms": float(original_row.square().mean().sqrt()),
                "original_row_l2": float(original_row.norm()),
            }
            provenance.update(
                hook=None,
                parameter=parameter,
                model_parameters_frozen=False,
                gradients_enabled=False,
                application="Scale one stored MLP output projection row for all positions; restore its exact snapshot after preview",
            )
            provenance["decoding"]["max_new_tokens"] = max_new_tokens
            return {
                "experiment_id": experiment_id,
                "prompt": prompt,
                "layer": layer,
                "neuron": neuron,
                "gain": float(gain),
                "max_new_tokens": max_new_tokens,
                "baseline": baseline,
                "steered": steered,
                "restoration": restoration,
                "parameter": parameter,
                "provenance": provenance,
                "probe": {
                    "id": probe.id,
                    "label": probe.label,
                    "desired_tokens": list(probe.desired),
                    "undesired_tokens": list(probe.undesired),
                },
                "interpretation_limit": "A temporary parameter influence experiment, not fine-tuning or a validated learned-fact edit.",
                "elapsed_seconds": time.perf_counter() - started,
            }
