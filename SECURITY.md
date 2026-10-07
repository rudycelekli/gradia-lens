# Security and private data

Lens executes local model inference and downloads a single pinned GPT-2
safetensors checkpoint through Transformers/Hugging Face. It does not load
remote Python model code, accept arbitrary module names, invoke a hosted model
API, or require credentials. Optional dependencies remain upstream software and
must be installed from trusted release artifacts.

Receipts include input prompts, outputs, and computation measurements. Store
them with appropriate permissions. Avoid private or sensitive prompts in public
examples and reports. Hashes provide byte integrity, not encryption, identity,
or authorization. A receipt from another person is untrusted data; replay only
accepts declared bounded Lens operations, never executable code.

Replay fails if the receipt digest, canonical sources, supported checkpoint,
parameter digest, or measured results differ. It cannot prove who produced a
receipt or that its claims were originally measured. Exact CPU restoration is a
tested local invariant, not a general model safety guarantee.

Report vulnerabilities privately through the contact route on
https://www.gradiahq.com. Do not attach secrets or private receipts to a public
report. Only the prepared 0.1.0a1 edition is covered by this alpha's local
validation; no long-term security support or registry release is implied.
