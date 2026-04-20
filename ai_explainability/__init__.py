"""
AI Explainability — public import entry point.

The repository is organised with several sibling top-level packages (``analysis``,
``output``) and a ``main`` CLI module at the root. This shim gives users a single,
canonical import path that matches the distribution name on PyPI / GitHub:

    >>> import ai_explainability
    >>> ai_explainability.__version__
    '0.1.0'
    >>> ai_explainability.run(["--config", "config.json"])

The shim is deliberately lightweight: it does **not** import ``analysis``, ``shap``,
``torch``, or any other heavy runtime dependency at import time. That means
``import ai_explainability`` stays fast and will not crash on a minimal install
(e.g. a Databricks cluster that hasn't yet installed the optional ``[torch]``
extra). The heavy modules are pulled in lazily when ``run()`` is actually called
or when a user explicitly reaches for a submodule via ``__getattr__``.

Existing code that uses the original top-level imports (``from analysis import
ANALYSIS_ROUTER``) continues to work unchanged.
"""

from __future__ import annotations

from typing import Sequence

__version__ = "0.1.0"

__all__ = ["__version__", "run"]


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


def __getattr__(name: str):
    """Lazy attribute access for advanced users.

    Allows ``from ai_explainability import ANALYSIS_ROUTER`` without paying the
    import cost up-front. Unknown attributes raise ``AttributeError`` as usual.
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
