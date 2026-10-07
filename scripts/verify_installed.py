"""Verify an installed public wheel. Optional real inference emits local receipts.

Run outside the source repository with the interpreter containing the installed
wheel. This script never adds a source directory to sys.path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--cache-dir")
    parser.add_argument("--output", type=Path, default=Path("lens-verification"))
    args = parser.parse_args()
    import gradia_lens
    from gradia_lens.api import execute, load_receipt, replay, save_receipt
    from gradia_lens.cli import doctor

    assert "torch" not in sys.modules and "transformers" not in sys.modules
    assert len(gradia_lens.catalog()) == 3
    report = {
        "installed_source": gradia_lens.__file__,
        "gradia_application_importable": importlib.util.find_spec("gradia") is not None,
        "base_import_lazy": True,
        "doctor": doctor(),
        "real_operations": [],
    }
    if report["gradia_application_importable"]:
        raise RuntimeError("Standalone verification requires an environment without the Gradia application.")
    if args.real:
        from gradia_lens import Engine

        engine = Engine(cache_dir=args.cache_dir)
        operations = [
            ("compare", {"layer": 10, "strength": 0.35, "max_new_tokens": 16}),
            ("trace", {"layer": 8, "strength": 0.15}),
            ("head", {"layer": 8, "head": 4, "fraction": 1, "max_new_tokens": 16}),
            ("weight", {"layer": 8, "neuron": 1253, "gain": 0, "max_new_tokens": 16}),
            ("evaluate", {"layer": 10, "strength": 0.35, "include_confirmation": True}),
            ("search", {"layers": [8, 10], "strengths": [0, 0.15, 0.35], "budget": 6}),
            ("challenge", {"layer": 10, "strength": 0.35, "budget": 12}),
        ]
        for operation, params in operations:
            receipt = execute(operation, params, engine=engine)
            path = args.output / f"{operation}.json"
            save_receipt(path, receipt)
            verified = replay(load_receipt(path), engine=engine)
            save_receipt(args.output / f"{operation}-replayed.json", verified)
            report["real_operations"].append(
                {"operation": operation, "receipt_sha256": receipt["sha256"], "measurements_exact": True}
            )
            print(f"verified real {operation} + exact replay", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "installed-verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
