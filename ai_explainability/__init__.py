"""
AI Explainability — public import entry point.

This package provides two complementary ways to run an analysis:

1. **Config-driven / CLI** — unchanged from the original tool. Either call
   the ``ai-explainability`` console script or ``ai_explainability.run()``
   with argv, and the existing ``config.json`` file drives everything.

2. **Programmatic / in-memory** — call :func:`ai_explainability.explain`
   with a fitted model object and an in-memory ``pandas.DataFrame`` (or
   ``pyspark.sql.DataFrame``, ``numpy.ndarray``, etc). Returns a
   :class:`ExplanationResult` without touching disk.

The package stays intentionally cheap to import: the heavy runtime stack
(``shap``, ``torch``, ``pyspark``, ``nbconvert``) is only pulled in when a
function that actually needs it is called.
"""

from __future__ import annotations

from typing import Sequence

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "run",
    "explain",
    "ExplanationResult",
]


def run(argv: Sequence[str] | None = None) -> None:
    """Run the explainability CLI programmatically.

    Parameters
    ----------
    argv:
        Optional argument list (e.g. ``["--config", "config.json"]``). When
        ``None`` the process's own ``sys.argv`` is used, matching the behaviour
        of invoking ``python main.py`` or the ``ai-explainability`` console
        script.
    """
    import sys

    # Defer the heavy import chain (analysis → shap → torch/…) until call time
    from main import main as _main

    if argv is None:
        _main()
        return

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *argv]
        _main()
    finally:
        sys.argv = old_argv


def explain(*args, **kwargs):
    """Forwarder to :func:`ai_explainability.api.explain`.

    Defined here (rather than a direct re-export) so that ``import
    ai_explainability`` stays free of heavy imports. The actual API module
    is loaded on first call.
    """
    from .api import explain as _explain

    return _explain(*args, **kwargs)


def __getattr__(name: str):
    """Lazy attribute access for advanced users.

    Allows ``from ai_explainability import ANALYSIS_ROUTER`` without paying
    the import cost up-front. Also resolves ``ExplanationResult`` on demand.
    """
    _lazy_map = {
        "ANALYSIS_ROUTER": ("analysis", "ANALYSIS_ROUTER"),
        "run_tabular_analysis": ("analysis", "run_tabular_analysis"),
        "run_timeseries_analysis": ("analysis", "run_timeseries_analysis"),
    }
    if name in _lazy_map:
        mod_name, attr = _lazy_map[name]
        import importlib

        mod = importlib.import_module(mod_name)
        value = getattr(mod, attr)
        globals()[name] = value  # cache for subsequent accesses
        return value

    if name == "ExplanationResult":
        from .result import ExplanationResult as _ER

        globals()["ExplanationResult"] = _ER
        return _ER

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
