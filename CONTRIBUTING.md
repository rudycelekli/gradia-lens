# Contributing

Clone this public repository or unpack its source archive; no Gradia account
or access to the surrounding application repository is required. Install `python -m pip install -e '.[dev]'`
for metadata and CLI tests, or `'.[model,dev]'` for real checkpoint work.

Run `python -m pytest tests`. Optional checkpoint tests require
`GRADIA_LENS_REAL_MODEL_TEST=1` and a pinned model cache configured through
`GRADIA_LENS_CACHE_DIR`; `HF_HUB_OFFLINE=1` prevents network access. To build
archives, run `python -m build --outdir dist` from this package directory.

Keep package import, catalog, doctor, and help free of optional model imports or
network access. Preserve fitting/discovery/confirmation separation, bounded
controls, exception-safe restoration, and canonical implementation hashes.
Never represent predicted or generated outcomes as measured results. A change
to model support or numerical methods requires real checkpoint tests and new
receipts; do not rewrite historical receipts to match new code.

Propose focused patches with the problem, resulting behavior, validation, and
limitations. Include zero controls, exception cleanup, and held-out evaluation
where they address the change. Do not include private prompts, model caches,
database contents, credentials, or raw customer receipts in patches.

The public project page is https://www.gradiahq.com/lens. Issues and pull requests are welcome at
https://github.com/rudycelekli/gradia-lens. PyPI publication remains pending. Contributor license is Apache-2.0.
