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


class _LazyRouter(dict):
    """Dict-like router that imports the target orchestrator on first access."""

    _loaders = {
        "tabular": _load_tabular,
        "timeseries": _load_timeseries,
    }

    def __getitem__(self, key: str):
        if key not in self and key in self._loaders:
            super().__setitem__(key, self._loaders[key]())
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        return key in self._loaders or super().__contains__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


ANALYSIS_ROUTER = _LazyRouter()


def __getattr__(name: str) -> Any:
    if name == "run_tabular_analysis":
        return _load_tabular()
    if name == "run_timeseries_analysis":
        return _load_timeseries()
    raise AttributeError(f"module 'analysis' has no attribute {name!r}")
