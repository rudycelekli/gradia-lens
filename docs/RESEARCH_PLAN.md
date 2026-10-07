# Gradia Lens research plan

Gradia Lens, also called Intervention Atlas, studies how people investigate a model by changing a declared part of its computation, measuring the intended effect and neighboring failures, and restoring the original state.

This document is a prospective protocol draft. It is not a completed benchmark, a publicly registered trial, or evidence of a generally superior optimizer. Before collecting confirmatory outcomes, freeze the datasets, implementation, metrics, budgets, and stopping rule in a timestamped public record. Amendments must state whether outcomes were already visible.

## Foundation and contribution to test

Inspection and intervention have substantial prior art. [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) exposes activation analysis and patching. [Neuronpedia](https://docs.neuronpedia.org/steering) provides visual feature exploration and steering. [EasyEdit](https://github.com/zjunlp/EasyEdit) covers editing, steering, and evaluation, while [ROME](https://github.com/kmeng01/rome) implements factual parameter editing. [AdaTest](https://www.microsoft.com/en-us/research/publication/adaptive-testing-and-debugging-of-nlp-models/) combines proposed tests with human validation.

Our contribution to test is a portable investigation that links a precise intervention, partial computation measurements, target and regression evidence, and restoration in a replayable record. A useful design could help an investigator move from a changed answer to a reviewed failure boundary and a well-chosen next experiment. Its benefit requires comparative evidence; neither a slider nor a diagram establishes novelty.

The current alpha uses one pinned GPT-2 checkpoint, contrast-fitted last-position residual steering, temporary head suppression, and temporary MLP projection-row scaling. It also provides finite search, lexical evaluation, mechanically proposed wording variants, and an offline viewer of captured measurements. See [RESEARCH.md](RESEARCH.md) for the implemented methods.

Attention is observed routing, and intermediate vocabulary projections are descriptive readouts. Causal patches support a particular component, contrast, and metric; coarse downstream residual restoration does not identify a unique circuit. Parameter-row scaling measures influence rather than a single learned fact or trained adaptation. The viewer selects recorded values and performs no inference.

## Study A: matched-budget intervention search

### Question

Does adaptive acquisition find independently confirmed useful settings more often than space-filling or seeded random search at matched observation budgets? The first study concerns GPT-2 lexical probes, not general correctness, factual editing, or model safety. Existing demonstration cases and previously viewed confirmation outcomes are excluded from confirmatory outcomes.

### Freeze before collection

Use `openai-community/gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`, CPU float32, greedy generation, model seed 17, and the qualified dependency/source edition. Preserve the existing eight fitting pairs for each of positive-tone, space-association, and future-tense. Collect new discovery and confirmation prompts/candidate expectations with no duplicate fitting string or case.

For each experiment, freeze 20 discovery and 20 fresh confirmation cases: six target, seven nearby, and seven unrelated. At least two reviewers independently assess candidate expectations and resolve disagreements before inspecting intervention outcomes. Record exclusions and their reasons before running a policy. Reviewing labels does not turn a lexical proxy into semantic truth.

Record hashes for fitting/discovery/confirmation data, acquisition code, dependencies, model identity, and this protocol. Publish the frozen protocol/data definition before collection where permitted; sensitive data must never be used in public examples.

### Campaigns and budgets

Use layers 8, 9, 10, and 11 and strengths 0, 0.05, 0.15, 0.25, 0.35, and 0.50: 24 unique settings. Compare adaptive, space-filling (`uniform`), and seeded random implementations at budgets 4, 8, and 12. Use policy seeds 0 through 19 without selecting favorable seeds afterward. A seed is an independent policy replicate only when it changes that policy's ordering; identical deterministic histories count as one history.

Each campaign observes only its own discovery results. Select the greatest discovery utility with the qualified implementation's frozen tie rule. Utility is mean target margin gain minus twice mean non-target desired-group log-probability damage. Audit exactly one selected setting on fresh confirmation after selection. No confirmation value enters acquisition or tie breaking. Campaigns do not share measured histories, outcome caches, or fitted surrogates.

Report observation budgets and actual forward-pass counts. Confirmation and setup costs are explicit and equal in purpose across policies. Record cold-start and warm-inference timings separately, and include hashing, fitting, preprocessing, and accelerator setup in full campaign time. Mark an unmeasured cost unknown instead of estimating an efficiency win.

### Primary outcome

A selected setting qualifies as useful only if fresh confirmation shows mean target margin gain at least 0.50 nats and mean non-target desired-group damage at most 0.25 nats. These thresholds define this study's narrow proxy, not semantic correctness or safe deployment.

Report paired adaptive-minus-control differences in qualified-setting rate, per-experiment outcomes, all selected settings, and all negative results. Use intervals that respect shared data and repeated histories; prompts from one family and repeated deterministic policies are not independent transfer evidence. With only three families, wider population inference remains exploratory.

Secondary outcomes are discovery utility, confirmation target gain, mean and worst non-target damage, count of cases with damage above 1 nat, time to a human-reviewed adverse effect, full wall time, forward passes, memory, and review effort. A failed primary outcome cannot be replaced after the run by a favorable secondary outcome.

The current heuristic has no demonstrated general advantage. Include results where a simpler policy wins. Add new behavior families and open checkpoints before claiming transfer.

## Study B: visual investigation

Compare a linked computation/intervention/failure view with an outputs-and-metrics view using the same measured data and controls. Freeze new investigation tasks and a scoring rubric. Counterbalance task and interface order, and blind scoring where practical.

Candidate outcomes are reproducible side-effect identification, supported versus unsupported causal statements, next-test selection, time, and review burden. Recruitment, sample size, exclusions, and a primary outcome must be finalized before collection. Qualitative feedback can explain the result but cannot substitute for the primary endpoint. No human-study advantage has been measured.

## Later research phases

1. Add finer head/MLP/token/path interventions with reverse controls, null controls, component-specific restoration, and tests across new wording.
2. Generate semantic challenges with independent human label review. Inherited mechanical expectations remain unverified until reviewed.
3. Study RuVector failure retrieval or learned decision surrogates as separate prospective arms. Freeze training data, abstention/calibration rules, and setup costs; predictions stay distinct from target-model measurements.
4. Qualify additional model architectures and numerical environments before broadening supported scope.
5. Treat persistent factual editing and LoRA/general training as separate workflows with objectives, data partitions, checkpoints/adapters, regression evaluation, and explicit rollback.

Every suggested intervention still requires model measurement. Retrieval and surrogate selection do not supply unlimited paths or free confirmation.

## Reproducibility and governance

Public examples must use synthetic or intentionally public inputs. Keep private prompts, customer receipts, checkpoints, raw internal logs, credentials, and caches out of issues and contributions. The included [positive-tone recording](../examples/records/positive-tone.json) is a curated synthetic demonstration; it is already visible and cannot serve as a new holdout.

A result should identify model revision, source edition, input tokenization, dtype/device, fitting and evaluation partitions, intervention location/position policy, direction or parameter provenance, metric, seed, costs, and restoration outcome. Export candidate-level outcomes rather than only the selected success. A digest binds bytes; it does not prove authorship or truthful measurement.

Changes to measurement semantics or benchmark claims require independent review and a frozen validation record. Publish null and adverse results, preserve historical receipts, and version changed schemas. Community contributions should include a bounded reproducer and appropriate zero/cleanup/regression checks. Support for this alpha is best effort, with no promised service level.

The Apache-2.0 license applies to the published package, including local experimentation, export, replay, and the offline viewer. Model weights retain their upstream terms. Hosted compute and private team collaboration are possible future services; they are not prerequisites for local use, implemented offerings, pricing commitments, or changes to the permissions granted for published source.
