# Paper track: joint acquisition of interventions and counterexamples

Status: prospective algorithm and study design. The released alpha is a reproducible research instrument; this document does not report an implemented joint acquisition algorithm, a completed benchmark, a world-first result, or an accepted paper. A useful contribution must survive strong controls and independent replication.

## Scientific question

Can choosing **which internal intervention and which challenge example to measure together** find independently confirmed useful settings and their failure boundaries with fewer actual target-model forwards than choosing settings or tests separately?

The object is a finite response surface over declared intervention locations/strengths and reviewed prompt families. It is not every possible computation path. A numerical change, attractive graph, or editable utility rubric is insufficient evidence for this claim.

## Prior art and the remaining hypothesis

| Primary source | What already exists | Required distinction to test |
| --- | --- | --- |
| [AdaTest](https://www.microsoft.com/en-us/research/publication/adaptive-testing-and-debugging-of-nlp-models/) | LM-proposed tests with human validation and an iterative debugging workflow. | Couple test acquisition to internal intervention decisions; compare against the same test-generation/review resources. |
| [ACDC](https://arxiv.org/abs/2304.14997) and its [reference code](https://github.com/ArthurConmy/Automatic-Circuit-Discovery) | Automated circuit discovery for a specified behavior, dataset and metric. | Discover where a proposed behavioral intervention fails across examples, rather than claim automated circuit discovery itself is new. |
| [AtP*](https://arxiv.org/abs/2403.00745) | Efficient approximation of activation-patching effects, with analysis of missed components. | Compare any approximation against measured intervention outcomes and charge gradient/proxy computation; retrieval is not free causal evidence. |
| [Activation-patching guidance](https://arxiv.org/html/2404.15255v1) | Distinct exploratory/confirmatory uses, controlled corruption choices, and limitations of component/path evidence. | Test a boundary hypothesis with appropriate contrasts and finer controls instead of interpreting a broad residual reset as a unique mediator. |
| [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html) | Model-specific interpretable replacement artifacts and validated attribution graphs. | Keep raw tensor telemetry separate from learned feature attribution; evaluate a decision procedure rather than rename existing graphs. |

TransformerLens, Neuronpedia and EasyEdit also establish substantial inspection, steering and editing tooling, as described in [RESEARCH.md](RESEARCH.md). This bounded source review does not establish absence of similar joint acquisition work. A broader current literature and code review, including active experimental design and newer steering papers, is a prerequisite to a novelty claim.

**Hypothesis H1:** at a frozen actual-forward budget, a joint intervention/example policy increases the rate of useful settings confirmed on fresh cases relative to strong intervention-only, test-only and uncoupled controls. **Hypothesis H2:** it predicts reviewed regression boundaries more accurately on a sealed audit matrix. The algorithm may fail either test. Its value is empirical, not guaranteed by coupling two search spaces.

## Proposed algorithm, not current functionality

Let `I` be a finite intervention set and `X` a pool of reviewed discovery probes. For each pair `(i, x)`, record the baseline and intervened candidate scores, intended margin change or regression loss, intervention identity, and restoration checks. Prompt families and roles determine frozen aggregation weights. A classifier or surrogate prediction must never be presented as a recorded measurement.

1. Begin with the same stratified initialization for every policy. Include zero-strength controls and representative target, nearby and unrelated cases. Fit directions only from the separate fitting partition.
2. Fit a calibrated uncertainty model over the partially observed `I × X` surface. A first prototype can use a frozen ensemble with intervention coordinates, prompt-family descriptors and an optional product-kernel model as a separately named arm. Hyperparameters and encoders are chosen on development families, never final confirmation outcomes.
3. Estimate uncertainty about each intervention satisfying the target-gain/regression constraints and about which probe families cross the adverse-effect threshold. Compute expected uncertainty reduction for a proposed pair, using a frozen posterior-sampling approximation, divided by predicted query cost. Cost estimates only choose actions; recorded forwards determine the enforced budget.
4. Acquire the highest-scoring legal `(i, x)` pair. Execute it on the target model; check cleanup/restoration; append the actual result and actual cost. Failed runs count toward cost and do not become fabricated zero measurements. Abstain from a recommendation if coverage or calibration is insufficient.
5. Repeat until the frozen budget or stopping rule. Select one intervention using discovery evidence and a declared tie rule. Freeze predictions about its effects and the broader boundary before opening confirmation.
6. Run the same independent confirmation procedure for every policy. Subsequent tuning starts a new exploratory campaign and needs new confirmation data.

The acquisition target is uncertainty about a **constrained behavioral boundary**, not uncertainty everywhere. Its components and coefficients must be frozen before the comparison. This is a concrete implementation proposal, with no claim of convergence, calibrated confidence or superiority yet. Mechanically generated wording with inherited expectations remains unverified until independently reviewed.

### An executable first acquisition objective

Define `F_i` as the binary event that setting `i` meets the frozen mean target-gain threshold and maximum permitted mean regression damage over the complete, reviewed discovery pool. Define `B_i,x` as the binary event that a non-target probe `x` exceeds the frozen adverse-effect threshold at setting `i`. These are posterior events about unmeasured cells, not observations. The discovery-pool events are decision aids; only the separate fresh audit establishes the primary confirmed outcome.

For observed data `D`, use the following declared uncertainty functional:

`U(D) = mean_i H[P(F_i | D)] + lambda × mean_(i,x in non-target pool) H[P(B_i,x | D)]`

Here `H` is binary entropy, and each mean gives equal weight within its declared finite set; family-balanced weights can replace these means only if frozen before the study. For each legal, unobserved pair `q`, estimate:

`A(q) = [U(D) − E_(y ~ predictive(q | D)) U(D ∪ {(q, y)})] / [predicted_cost(q) + epsilon]`

Draw the same fixed number of posterior predictive samples for each action, update the same frozen surrogate for each hypothetical result, and use a predeclared stable tie order. Clip only Monte Carlo negative uncertainty reductions to zero under a rule fixed for all arms; do not clip measured gains or damage. Query the highest scoring pair whose actual worst-case forward/token allowance fits the remaining budget. The number of samples, `lambda`, `epsilon`, surrogate family and retraining schedule require development-only choices and must be preregistered before Stage 2. Surrogate work itself is reported as cost.

The same-surrogate uncoupled control alternates intervention acquisition on a fixed stratified probe batch with probe acquisition at a fixed current setting; it never optimizes a free pair jointly. It receives the same initialization, predictive samples and measured resources. Compare those arms before attributing improvement to coupling. A greedy highest-predicted-gain policy is an additional useful control because entropy reduction can spend queries without improving the selected setting. Publish the acquisition score, prediction and eventual measured outcome separately for every query.


The objective assumes that the frozen predictive model represents uncertainty usefully; an ensemble disagreement score alone does not prove calibration. Require predeclared calibration/coverage checks on independent development families, then freeze an out-of-distribution warning rule. If the surrogate cannot fit, fails those checks or encounters unsupported probe representations, switch to the predeclared seeded space-filling fallback and withhold confidence or boundary certification. If the campaign has insufficient measured coverage at the stopping point, abstain from a recommended intervention. Missing predictions are never zero-risk observations. Log the fallback and its cost; do not exclude those campaigns from the primary endpoint.

Prospective pseudocode, distinct from the shipped implementation:

```text
D = execute_shared_stratified_initialization()
while a legal unobserved pair fits the remaining actual-cost allowance:
    posterior = fit_frozen_surrogate(D)
    if predeclared_uncertainty_checks_fail(posterior):
        q = next_seeded_space_filling_pair()
        mark_fallback_and_withhold_confidence()
    else:
        for q in legal_pairs:
            expected_U = mean(U(update(posterior, q, y))
                              for y in frozen_count_predictive_samples(q))
            score[q] = max(0, U(posterior) - expected_U) / (cost[q] + epsilon)
        q = stable_argmax(score)
    result, actual_cost = measure_and_verify_restoration(q)
    append_actual_result_or_failure(D, q, result, actual_cost)
freeze_one_discovery_selection_or_abstain_and_boundary_predictions(D)
run_separate_fresh_confirmation_and_sealed_matrix_audit()
```

Do not interpret a sampled hypothetical update as a model measurement. Query failures consume their executed budget, and the primary confirmed-success endpoint counts abstentions as unsuccessful selections under a rule frozen in advance.

A later study may generate new semantic probes during the campaign. The first algorithm study should use the same frozen, reviewed pool for every arm so benefits cannot be explained by unequal generators or label quality. Review time and proposal-model usage are separate resources with matched caps.

## Budget instrumentation is a blocking study requirement

Current candidate-level timing covers an entire discovery evaluation batch, possibly including cold fitting. It is not serving latency or a throughput measurement. The current tool has not established a complete target-model forward ledger. Equal candidate counts or equal seven-case evaluation counts are **not** equal forward-pass budgets.

Before H1 is tested, instrument every target-model forward used by fitting, baselines, teacher-forced continuation scoring, generation, patching, verification and restoration. Record batch size, sequence length, processed token positions, mode/device/dtype, cache hits, and cold/warm scope. Instrument and separately report backward/gradient work, encoder/proposal model work, surrogate fitting, hashing, loading, memory and human review. Neither cached logical evaluations nor a GPU batch should be counted as if it were a fresh single-example forward.

Enforce matched limits on actual forward invocations **and** processed token positions. Use a separate fixed allowance for the final confirmation audit and causal controls, report that cost explicitly, and charge the same purpose to every arm. Shared baseline caching must be either disabled for all arms or defined symmetrically with actual executed and logical costs both reported. Predictive cache reuse across campaigns is prohibited.

Choose budget values in a cost-only development pilot, then freeze them before collecting primary outcomes. Stop before an action exceeds its declared bound; report unused budget. Publish the ledger and all candidate failures. This instrumentation does not currently exist merely because a duration field is available.

## Staged experiments and baselines

**Stage 0: instrumentation and adversarial qualification.** Verify the ledger against small known workloads, exceptions and cache paths. Demonstrate null-control parity, restoration after failures, source provenance and no confirmation access. Small synthetic surfaces may check acquisition logic, but cannot demonstrate behavior on a real model.

**Stage 1: finite real-model pilot.** Keep the pinned GPT-2 edition and finite layer/strength grid. Collect fresh, human-reviewed examples with independent fitting, discovery and confirmation strings. Use target, nearby and unrelated roles. Existing published demo/confirmation examples are development data. This stage may reject the algorithm and remains exploratory; freeze amendments with outcomes-visible status.

**Stage 2: preregistered comparison.** Freeze new behavior families, data/label hashes, candidate grids, algorithms, seeds, budgets, aggregation weights, statistical analysis and stopping rule. Include at least two qualified open-model architectures before making transfer claims. Sample-size and family counts require a development-only power/sensitivity analysis; no confirmatory sample size is settled by this plan. Two reviewers resolve expectations before seeing intervention outputs; blinded semantic judgments remain separate from lexical scores.

At minimum compare:

- Seeded random and space-filling pair acquisition, with the same pools, initialization and resource limits.
- Intervention-only optimization using a fixed discovery-test batch.
- Test-only adaptive acquisition at a predeclared intervention, plus a predeclared selection stage.
- Alternating, uncoupled intervention/test acquisition with the same surrogate and resources.
- Joint boundary acquisition, and ablations removing boundary targeting, interaction terms, uncertainty, or cost normalization.
- A human-reviewed adaptive-testing control using the same proposed-test pool and review budget. Do not present its performance as a literal AdaTest reproduction unless the reference method is faithfully implemented.

For component-localization claims, add exact patching and an appropriately qualified AtP*/ACDC-style control separately; those methods solve related but different objectives. A failure-retrieval accelerator is another prospective arm. Include indexing/embedding setup and target-model verification cost rather than compare a lookup with a full model run.

## Outcomes, causal evidence and confounding

The narrow primary H1 outcome is the rate of selected settings meeting a frozen fresh-confirmation target-gain threshold while remaining under a frozen non-target damage threshold. The alpha's lexical proxies can define a first pilot; a semantic usefulness claim requires independent judgments and agreement reporting. Freeze thresholds on development data. Report target gain, mean/worst damage, per-family results, adverse-case counts, restoration failures and abstentions, including unsuccessful campaigns.

For H2, predict regression probabilities on a sealed, independently reviewed intervention/probe audit matrix. Measure calibration and boundary classification with proper scores and frozen thresholds. Compute this audit after predictions are frozen, never expose it to acquisition, and report its model/human cost. A matrix exhaustively measured on one small grid is not every possible failure boundary.

Separate policy seeds from independent data families. Repeated deterministic histories count once. Prespecify joint versus the same-surrogate uncoupled policy as the primary contrast; control multiplicity for additional superiority comparisons. Use paired comparisons on the same sealed inputs and intervals that account for clustering by family/model; choosing the best seed, family or metric after seeing results invalidates the primary claim. Report compute-adjusted curves and the prespecified endpoint; do not replace a failed endpoint with a favorable chart.

For any causal interpretation, require same-prompt baseline/intervention/undo, zero controls, norm-matched random or orthogonal directions, sign/strength sweeps, and independent wording. Test component-specific noising and denoising with appropriate prompt contrasts; test interactions and backup behavior where relevant. Finer path controls are needed to distinguish direct from mediated effects. A whole downstream residual reset can trivially remove the upstream intervention and does not establish a minimal circuit. Attention weight, graph position and unembedding readout alone do not establish causality.

Audit these alternative explanations: unequal probe generation/review, shared cached outcomes, semantic label errors, prompt-family memorization, token-length cost differences, surrogate miscalibration, out-of-distribution ablation, greedy-decoding threshold effects, changed dataset/source editions, selective completion, and reporting only successful controls. Publish negative and overshoot effects.

## What would falsify or narrow the claim

- The primary comparison fails the preregistered success criterion. A sufficiently precise upper bound below the declared smallest useful effect rules out the proposed practical benefit; an imprecise interval leaves the question unresolved.
- The apparent benefit disappears with identical test pools, labels, initialization, caches and uncertainty models.
- Better discovery scores fail fresh confirmation, or are explained by worse nearby/unrelated behavior.
- Boundary probabilities are miscalibrated, miss important failures, or turn unreviewed labels into apparent counterexamples.
- Improvements do not transfer beyond the original families/model. Then the claim stays local.
- A faster proposal step loses its gain once setup, inference, verification and review costs are included.
- Causal hypotheses fail finer controls. Behavioral boundary discovery may still be useful, but a circuit claim must be withdrawn.

A null result is a publishable research artifact if the method, costs, data and adverse results are reproducible. Venue acceptance and originality are judgments for independent reviewers, not launch promises.

## Required artifacts before a paper-level claim

Publish frozen protocol/amendments, safe data and label-review procedure, algorithm/code edition, complete cost ledgers, all observed and predicted pair records with explicit field separation, selected-setting and sealed-audit outcomes, ablations, environment qualification, restoration controls and an independent replication package. Public examples contain no private prompts, organization receipts, credentials or checkpoints. Keep historical released receipts intact; develop this algorithm in a separately reviewed source edition.

The next engineering step is forward-cost instrumentation and a measured pair-query adapter, followed by a development-only pilot. Editable rubric ranking and a portable graph are useful research interfaces; they do not implement or validate H1/H2 by themselves.
