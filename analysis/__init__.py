# Lazy orchestrator router — importing this package must not pull in torch
# (LSTM backend) or sklearn-heavy code paths.
#
# Exposes the same public surface as before (``run_tabular_analysis``,
# ``run_timeseries_analysis``, ``ANALYSIS_ROUTER``) but resolves each entry on
# first access so that users of the tabular path never pay for the timeseries
# dependency stack and vice-versa.

from importlib import import_module
from typing import Any

__all__ = [
    "run_tabular_analysis",
    "run_timeseries_analysis",
    "ANALYSIS_ROUTER",
]


def _load_tabular():
    return import_module("analysis.tabular").run_tabular_analysis


def _load_timeseries():
    return import_module("analysis.timeseries").run_timeseries_analysis


# ``ANALYSIS_ROUTER`` is a plain dict — so ``.get()``, ``.keys()``, ``in`` and
# iteration all behave exactly like the original eager dict — but each value is
# a thin wrapper that defers the actual submodule import until it is *called*.
# This keeps ``import ai_explainability`` cheap without breaking consumers like
# ``main.py`` that read ``ANALYSIS_ROUTER.keys()`` or call ``.get(...)``.
ANALYSIS_ROUTER = {
    "tabular": lambda *args, **kwargs: _load_tabular()(*args, **kwargs),
    "timeseries": lambda *args, **kwargs: _load_timeseries()(*args, **kwargs),
}


def __getattr__(name: str) -> Any:
    if name == "run_tabular_analysis":
        return _load_tabular()
    if name == "run_timeseries_analysis":
        return _load_timeseries()
    raise AttributeError(f"module 'analysis' has no attribute {name!r}")
