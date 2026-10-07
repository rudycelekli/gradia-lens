"""Credential-free local research CLI; help, catalog, and doctor load no model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path

from . import __version__, catalog
from .api import MAX_RECEIPT_BYTES, execute, load_receipt, receipt_digest, replay, save_receipt, source_hashes

MODEL_DEPENDENCIES = {
    "torch": "2.8.0",
    "transformers": "4.57.3",
    "numpy": "1.26.4",
    "huggingface-hub": "0.36.0",
    "tokenizers": "0.22.1",
    "safetensors": "0.7.0",
}


def doctor() -> dict:
    """Inspect installed metadata only. Never import optional modules or download."""
    dependencies = {}
    for name, expected in MODEL_DEPENDENCIES.items():
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        dependencies[name] = {"expected": expected, "installed": version, "matches": version == expected}
    return {
        "package": {"name": "gradia-lens", "version": __version__},
        "python": platform.python_version(),
        "dependencies": dependencies,
        "model_dependencies_match": all(item["matches"] for item in dependencies.values()),
        "supported_device": "cpu",
        "model": {"id": "openai-community/gpt2", "revision": "607a30d783dfa663caf39e06633721c8d4cfcd7e"},
        "checkpoint_loaded": False,
        "network_used": False,
        "sources": source_hashes(),
        "publication": "Prepared local alpha; registry publication is a separate reviewed action.",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gradia-lens",
        description="Change a real model component. Measure behavior and side effects. Restore it exactly.",
    )
    parser.add_argument("--version", action="version", version=f"Gradia Lens {__version__}")
    commands = parser.add_subparsers(dest="operation", required=True)
    commands.add_parser("catalog", help="List versioned experiments without loading a model")
    commands.add_parser("doctor", help="Inspect dependency metadata without loading or downloading a model")
    command = commands.add_parser(
        "view", help="Open recorded evidence as a self-contained offline HTML viewer"
    )
    command.add_argument("record", type=Path)
    command.add_argument("--output", type=Path, required=True)
    descriptions = {
        "compare": "Compare greedy baseline and residual-steered outputs",
        "trace": "Capture attention, residuals, MLP contributions, and restoration probes",
        "head": "Temporarily suppress one attention head's last-position output",
        "weight": "Temporarily scale one MLP projection parameter row, then restore",
        "evaluate": "Measure discovery probes and optional confirmation cases",
        "search": "Search a bounded grid; audit the selected setting on confirmation",
        "challenge": "Generate discovery wording variants and measure candidate failures",
    }
    for operation, description in descriptions.items():
        command = commands.add_parser(operation, help=description)
        command.add_argument("--experiment", dest="experiment_id", default="positive-tone")
        command.add_argument("--cache-dir", help="Pinned model cache; also accepts GRADIA_LENS_CACHE_DIR")
        command.add_argument("--offline", action="store_true", help="Use local checkpoint files only")
        command.add_argument("--output", type=Path, help="Write the complete JSON receipt atomically")
        if operation in ("compare", "trace", "head", "weight"):
            command.add_argument("--prompt")
        if operation != "search":
            command.add_argument("--layer", type=int)
        if operation in ("compare", "trace", "evaluate", "challenge"):
            command.add_argument("--strength", type=float, default=0.15)
        if operation in ("compare", "head", "weight"):
            command.add_argument("--max-new-tokens", type=int, default=16)
        if operation == "trace":
            command.add_argument("--max-tokens", type=int, default=48)
        if operation == "head":
            command.add_argument("--head", type=int, default=0)
            command.add_argument("--fraction", type=float, default=1.0)
        if operation == "weight":
            command.add_argument("--neuron", type=int, default=0)
            command.add_argument("--gain", type=float, default=1.0)
        if operation == "evaluate":
            command.add_argument("--include-confirmation", action="store_true")
        if operation in ("search", "challenge"):
            command.add_argument("--budget", type=int, default=6 if operation == "search" else 12)
            command.add_argument("--seed", type=int, default=17)
        if operation == "search":
            command.add_argument("--layers", type=int, nargs="+", default=[8, 10])
            command.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.15, 0.35])
            command.add_argument("--strategy", choices=("adaptive", "uniform", "random"), default="adaptive")
    command = commands.add_parser(
        "replay", help="Replay a receipt and fail if measured results or provenance differ"
    )
    command.add_argument("receipt", type=Path)
    command.add_argument("--cache-dir")
    command.add_argument("--offline", action="store_true")
    command.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "catalog":
            print(json.dumps(catalog(), indent=2, allow_nan=False))
            return 0
        if args.operation == "doctor":
            print(json.dumps(doctor(), indent=2, allow_nan=False))
            return 0
        if args.operation == "view":
            from .viewer import write_viewer

            if args.record.stat().st_size > MAX_RECEIPT_BYTES:
                raise ValueError("Viewer input exceeds the 64 MiB limit.")
            record = json.loads(args.record.read_text(encoding="utf-8"))
            if not isinstance(record, dict):
                raise ValueError("Viewer input must be a JSON object.")
            if record.get("schema") == "gradia.lens.receipt.v1":
                # Viewing historical measurements does not rerun their source
                # edition. Check byte integrity, retaining recorded provenance.
                if record.get("sha256") != receipt_digest(record):
                    raise ValueError("Receipt SHA-256 integrity mismatch.")
            destination = write_viewer(record, args.output)
            print(json.dumps({"viewer": str(destination), "network_used": False, "checkpoint_loaded": False}))
            return 0
        if args.offline:
            os.environ["HF_HUB_OFFLINE"] = "1"
        if args.operation == "replay":
            receipt = replay(load_receipt(args.receipt), cache_dir=args.cache_dir)
        else:
            params = {
                key: value
                for key, value in vars(args).items()
                if key not in {"operation", "cache_dir", "output", "offline"} and value is not None
            }
            receipt = execute(args.operation, params, cache_dir=args.cache_dir)
        if args.output:
            save_receipt(args.output, receipt)
            print(
                json.dumps(
                    {
                        "operation": receipt["operation"],
                        "receipt": str(args.output),
                        "sha256": receipt["sha256"],
                        "replay": receipt.get("replay"),
                    }
                )
            )
        else:
            print(json.dumps(receipt, indent=2, allow_nan=False))
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        print(json.dumps({"error": str(exc), "kind": type(exc).__name__}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
