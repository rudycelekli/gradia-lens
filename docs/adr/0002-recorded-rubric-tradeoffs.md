---
id: ADR-0002
title: Make recorded intervention tradeoffs inspectable offline
status: Proposed
date: 2026-10-06
tags: [research, interpretability, reproducibility, evaluation]
depends_on: [ADR-0001]
---

# ADR-0002: Make recorded intervention tradeoffs inspectable offline

Status: proposed next alpha, pending public pull-request review.

**Depends-on**: ADR-0001

## Context

A computation trace describes one setting. Choosing useful interventions also
requires examining multiple discovery measurements, regression effects, and
runtime. A polished graph alone does not demonstrate that an intervention
works beyond the original prompt or that an optimizer is more effective.

The standalone package must remain useful without a Gradia account, model
installation, network access, or access to the private Gradia repository.
Historical records can lack timings or individual cases; the renderer must not
fill those gaps with invented measurements or lend another setting's audit.

## Decision

Add standard-library derivation helpers and a self-contained HTML viewer for
recorded search, evaluation, and campaign data. Derive target margin gain and
regression damage from explicitly tagged discovery rows when available. Use
only explicitly declared discovery summaries as a historical fallback.

Group settings by captured model/revision/parameter/dtype, dataset, engine, and
discovery-case fingerprints. If identity fields are missing, compare within the
original supplied record only. Rank with nonnegative weights between zero and
ten: gain weight times mean target margin gain, minus damage weight times mean
regression damage, minus time weight times whole discovery evaluation seconds.
The frontier maximizes gain and minimizes damage, plus duration when its weight
is active. Exact coordinate ties remain nondominated and ranking ties retain
captured order. These computations are transparent and deterministic.

Confirmation does not enter either calculation. Display an audit only when its
recorded experiment, layer, strength, and provenance match the original setting.
Changing the rubric does not make a different setting confirmed. Pinned probe
comparisons require matching case ID, prompt, role and expected vocabulary.
Existing computation and generated output panels retain their own context.

New search points record actual whole discovery evaluation batch durations.
Missing historical values stay null. Time axes and penalties require every
candidate in a group to have timing. Do not substitute confirmation-inclusive
batch totals. Declare cache/fitting conditions as uncontrolled; the observations
support exploratory tradeoffs, not a comparative speed claim. Top-level receipt
timings separately describe model loading and serialized operation scope.
Receipt integrity binds timings; exact numerical replay excludes variable time.

## Alternatives

- Live inference in the viewer adds dependencies and makes slider exploration
  expensive and nondeterministic. Keep inference in the bounded CLI/API.
- Rank on confirmation improves apparent choice at the cost of invalidating the
  audit. Keep discovery acquisition and confirmation display separate.
- Impute missing timings gives a complete chart but fabricates performance
  evidence. Disable time comparisons for incomplete groups.
- Compare all supplied records on one chart ignores different models and case
  subsets. Use explicit compatibility groups and conservative missing identity.
- A fixed utility hides the researcher's tradeoff. Expose the formula and allow
  bounded local weights without claiming the new ranking is an unseen result.

## Consequences

The viewer supports reproducible, inspectable choice among measured settings,
including regression and timing tradeoffs. It does not discover additional
interventions, predict untested regions, certify correctness, establish causal
edges, or offer unlimited exploration. Timing-based reranking can favor warm
cache states; a future performance study needs controlled repetitions.

## Validation

Python controls check row-derived metrics, confirmation perturbation invariance,
frontier ties, rubric changes, timing validation, missing timing, incompatible
fingerprints/case subsets, audit context, and receipt timing integrity/replay.
Node tests execute the shipped JavaScript, compare its utility/frontier with
Python derivation, move selection and pinned references, and check audit/timing
transitions. Existing partial-trace, zero-effect, script-injection, model import,
restoration, and exact replay controls remain required. A real pinned-model
search receipt and detached-wheel replay qualify the changed timing path.

Qualification covers 84 implementation and curated-fixture cases: 81 implementation cases, including
real-checkpoint and shipped-JavaScript controls, plus three curated-fixture
checks. The public adaptation additionally passes 14 archive-boundary cases,
including omission of promised examples, research documents and images.
The independent implementation reviewer initially ran 35 rubric/viewer cases
without skips. Final targeted qualification covers 43 rubric/viewer/fixture
cases, including numeric overflow refusal and recovery, campaign-first display,
and named keyboard-accessible utility controls.

The final detached wheel executes a real six-setting GPT-2 search and reproduces
its non-timing measurements exactly on replay. The installed implementation's
ten Python modules match the canonical source. Both operations restore the
original parameters and leave zero hooks. The curated campaign preserves the
original numerical measurements while removing local receipt identities.
The shared engine/experiment source SHA-256 remains
`6d1e2d3eb92a60a8908b8c09297e9c747993a66c9e2377cc13e9ddf4a76c755a`.

These checks qualify the implementation and evidence handling. They do not
validate comparative search efficiency, the proposed joint-acquisition
algorithm in PAPER_TRACK.md, broad semantic quality or a complete causal graph.
