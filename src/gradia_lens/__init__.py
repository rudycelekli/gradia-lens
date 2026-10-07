"""Gradia Lens: standalone, reversible model research.

Importing the package and listing experiments require only the standard library.
Accessing ``Engine`` explicitly loads optional model dependencies; construction
then loads the pinned checkpoint. There is no network request on package import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .experiments import catalog, get_experiment

__version__ = "0.1.0a1"
__all__ = ["Engine", "catalog", "get_experiment", "__version__"]

if TYPE_CHECKING:
    from .engine import Engine


def __getattr__(name: str) -> Any:
    if name == "Engine":
        from .engine import Engine

        return Engine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
