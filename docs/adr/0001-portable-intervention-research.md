---
id: ADR-0001
title: Portable intervention research with evidence-bound controls
status: Accepted
date: 2026-10-06
tags: [research, open-source, interpretability, reproducibility]
---

# ADR-0001: Portable intervention research with evidence-bound controls

## Context

Gradia Lens, also called Intervention Atlas, needs a useful open-source foundation that researchers can run and inspect without a hosted account. A visual change in an answer is weak evidence unless the investigator can test neighboring cases, inspect what was actually changed, restore the original computation, and reproduce the experiment.

Activation steering, patching, model editing, and adaptive test discovery already exist. The research opportunity is a connected investigation workflow and a testable improvement in discovery, rather than a claim that the underlying mechanisms are newly invented. The initial source edition is `gradia-lens` 0.1.0a1.

## Decision

Publish a standalone Apache-2.0 Python tool with a zero-dependency base and an explicit optional inference stack. Support one immutable GPT-2 checkpoint on the qualified CPU path. Keep catalog/help/dependency inspection and captured-data viewing free of model loading, network calls, accounts, or premium authentication.

Expose declared reversible operations: contrastive last-position residual steering, attention-head suppression, and temporary MLP projection-row scaling. Keep runtime steering, temporary parameter influence, persistent factual editing, and learned fine-tuning conceptually separate. The alpha implements the first two categories of control, not persistent editing or training.

Link actual output and partial token/block/head/MLP measurements to lexical target/regression tests, finite discovery-only setting search, and review-required wording challenges. Select one setting before confirmation. A person tuning after reading confirmation needs fresh holdout data.

Bind portable receipts to model/source/dataset/direction provenance and byte integrity. Verify parameter restoration and hook cleanup; reject an externally altered model before emitting a receipt that would claim the original checkpoint identity. Exact replay is qualified for the same implementation and tested CPU computation.

Provide a self-contained offline viewer of recorded values. Use inert JSON and text-only rendering for untrusted input, with no external dependencies or requests. Keep independently measured parameter previews separate from residual-steering traces. Missing measurements must remain missing; zero differences remain zero. Structural edges and attention observations are not presented as a complete causal circuit.

Use synthetic public examples and a clean source history. Publish intended code, tests, method/plan documents, license/attribution, and the curated recording. Exclude model weights, private receipts, internal logs, caches, databases, and unrelated application code.

## Alternatives considered

- A hosted-only interface: easier to centralize compute, but prevents independent local use and makes reproduction depend on a service.
- An interface without side-effect evaluation or source-bound receipts: visually compelling, but does not support a complete investigation.
- Persistent editing or fine-tuning first: scientifically useful, but adds objectives, data/checkpoint management, and a different restoration contract.
- A broad multi-model or unlimited-search promise: attractive scope, but unsupported by the present qualification and finite inference budget.
- Claiming search superiority from a successful demonstration: does not provide an independent matched-budget comparison.

## Consequences

The small pinned model makes a concrete research prototype possible, but the built-in lexical probes do not establish general correctness, factual repair, or model safety. Attention and intermediate readouts are partial descriptions. Whole downstream residual patches can restore a contrast without isolating a unique mediator. Parameter rows are numeric targets, not uniquely identified knowledge.

The portable tool can work independently of a native Gradia integration. Future hosted compute or team collaboration may complement local use without changing permissions already granted for published source. Such services are proposals rather than current premium offerings.

Every supported new model, device, edit method, or training method needs its own qualification and restoration evidence. Source changes require a new computation edition for replay. Schema evolution and scientific claims need independent review. A receipt hash is not an author signature.

## Validation

The reviewed alpha's nine Python implementation modules have exact byte parity with the qualified wheel and detached installs. The engine/experiment source digest is `6d1e2d3eb92a60a8908b8c09297e9c747993a66c9e2377cc13e9ddf4a76c755a`; published manifests identify the remaining source/artifact fingerprints.

Seven actual installed-wheel operations and seven exact measurement replays passed with the surrounding application not importable. Direct CLI comparison, trace, and parameter-preview execution/replay passed. Real head and parameter-row previews changed generated tokens and restored the original computation; full parameter bytes changed during the row preview and matched afterward. A real changed-row control verified rejection of preexisting checkpoint drift.

Package qualification covered 48 cases: 47 suite cases plus a separately executed Node 22 missing-head/MLP DOM regression. Independent reviewer checks verified archive contents, canonical bytes, all wheel RECORD paths/sizes/hashes, receipt/source integrity, complete recorded replay measurement parity, and seven portable-viewer tests without model/application imports.

The curated synthetic recording's 21,918 retained measurement values match the recorded source experiments with no projection differences. The comparison and trace share layer 10/strength 0.35; the layer-8 physical preview is explicitly independent. Organization receipt identifiers are omitted. Its SHA-256 is `870c2b1d4d2d6bdd2f7e529513c75768d5a6511e625bb64b8ace98a2250a49a2`.

These checks qualify the initial implementation and curated demonstration, not a general performance advantage. The [prospective research plan](../RESEARCH_PLAN.md) specifies the next matched-budget comparison, fresh holdouts, negative-result reporting, and a separate visual investigation study. It is not a completed or registered benchmark.

## Follow-up

Freeze datasets and acquisition code before the next confirmatory study. Preserve the released alpha while developing new viewer or scoring work in a separately reviewed change. Complete new model/edit/training support only with explicit provenance, regression evidence, and restoration contracts.
