# Changelog

## 0.1.0a2 — proposed next alpha

- Portable, offline search/evaluation/campaign tradeoff viewer with editable
  discovery utility weights, measured-setting scatter, Pareto frontier, bounded
  zoom/pan, and matching individual-probe comparisons against a pinned reference.
- Strict confirmation binding to the originally audited setting; confirmation
  outcomes never contribute to ranking or frontier calculation.
- Compatibility groups include model, dataset, engine, and captured discovery
  case definitions, with conservative handling of absent fingerprints.
- Actual whole discovery-batch times in new search observations; missing timings
  remain unmeasured, and confirmation-inclusive totals are not substituted.
- Scoped model-load/receipt timings bound by receipt integrity, with variable
  wall time excluded from exact numerical replay.

This iteration is prepared for a reviewed public branch. It is not a registry
publication and does not establish an optimizer or speed advantage.

## 0.1.0a1 — prepared alpha

- Standalone zero-dependency base package and optional pinned CPU model stack.
- Three versioned contrast-fitting and lexical evaluation experiments.
- Reversible residual steering, attention head suppression, and parameter-row
  preview with restoration checks.
- Actual attention/residual/MLP telemetry and whole-residual restoration probes.
- Bounded discovery-only search and measured generated wording challenges.
- CLI, Python API, SHA-256 receipts, and exact measurement replay.
- Canonical engine shared with the native Gradia workbench.

This entry records prepared functionality, not PyPI publication or deployment.
