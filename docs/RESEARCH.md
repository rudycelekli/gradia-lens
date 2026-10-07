# Research method and open questions

The alpha supports `openai-community/gpt2` at immutable revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`: 124,439,808 parameters, 12 blocks,
12 attention heads per block, residual width 768, and MLP width 3072. The
[upstream model card](https://huggingface.co/openai-community/gpt2) describes the
original model. Weights are downloaded separately under their upstream terms.

## Fitting and evaluating interventions

Each experiment has eight contrast pairs for direction fitting. The engine
captures last-position post-block residuals for both groups, subtracts group
means, normalizes the contrast, and scales it to the average training residual
L2 norm. The intervention adds strength times that direction after one selected
block at the last position on each forward, including each generation step.
It uses CPU float32, greedy decoding, seed 17, and no key/value cache.

Fitting strings are disjoint from discovery and confirmation probes. Each
experiment includes seven discovery and six confirmation cases, with target,
nearby, and unrelated roles in both partitions. Definitions and expected
candidate continuations are in `src/gradia_lens/experiments.py`; these labels
were manually constructed and are not an independently collected benchmark.

For each candidate continuation the engine measures its complete teacher-forced
log probability. Group scores use log-sum-exp. Target gain is the change in
desired-minus-undesired group margin. Regression damage is the positive part of
the desired group's log-probability drop. These metrics report lexical support,
not semantic truth. A .25-nat regression threshold and .05-nat target margin
threshold are declared in provenance.

Bounded search maximizes discovery target gain minus twice mean regression
damage. Adaptive acquisition combines nearby observed utility, distance-based
uncertainty, and observed sign/threshold changes; uniform and seeded random
controls use the same grid. Confirmation is evaluated after selecting one
setting and is never passed to acquisition.

## Partial computation evidence

Attention weights are observed routing coefficients. Head contribution vectors
are calculated from each head's concatenated context slice and its output
projection rows, excluding shared output bias. Their sum plus bias is checked
against attention output within float32 error. MLP unit summaries report
observed post-activation contribution norms and signed activation changes.
Raw residual coordinates have numeric indices, not invented semantic labels.

Intermediate residual vocabulary readouts apply the final layer norm and
unembedding to incomplete states. They are descriptive logit-lens estimates.
Restoration probes replace whole downstream last-position residuals with their
stored baseline values. Because the steering affects only the last position,
such a patch can remove all upstream differences. That establishes reversal
for this contrast, not a minimal or unique mediator.

Temporary head suppression changes one head's final-position context slice
before output projection. Temporary parameter preview scales one declared MLP
projection row, restores the exact snapshot even on failure, and checks full
parameter hashes and baseline parity. Neither operation learns new weights.

These mechanisms build on existing research. [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
provides activation analysis and patching; [activation patching guidance](https://arxiv.org/html/2404.15255v1)
discusses contrast and metric choices. [Neuronpedia](https://docs.neuronpedia.org/steering)
and [EasyEdit](https://github.com/zjunlp/EasyEdit) already provide steering or
editing workflows. [Circuit Tracing](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)
uses learned model-specific replacement artifacts; this alpha has no trained
SAE/transcoder and does not claim an attribution graph of that kind.

## Prespecified next study

This is a study plan, not a completed result or a registered trial. Before
running it, freeze the model, fitting examples, discovery data, fresh
confirmation set, candidate grid, proposal policy, budgets, and stopping rule.

1. Compare adaptive, space-filling, and seeded random acquisition at equal
   target-model evaluation budgets across all three experiments and additional
   independently collected families. Report each seed and all failures.
2. Measure time to a human-confirmed adverse effect, useful-setting discovery,
   and reviewer time. Treat inherited generated labels as unverified until a
   reviewer checks them. Add a human-reviewed adaptive-testing control.
3. Report cold model loading, direction fitting, forwards, wall time, memory,
   and preprocessing costs. Count accelerator setup and verification inference.
4. Freeze one setting per campaign for fresh confirmation. Further tuning after
   confirmation requires another held-out set.
5. Add finer head/MLP/token/path interventions only with reverse controls and
   per-component restoration tests. Report null, negative, and overshoot effects.

The current heuristic has no demonstrated general advantage. RuVector failure
retrieval, typed decision surrogates, larger models, learned feature artifacts,
ROME editing, and LoRA training are future research options. None is required
to run this alpha, and none supplies free or unlimited model evaluations.
