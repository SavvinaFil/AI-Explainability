"""
Correctness checks attached to every :class:`ExplanationResult`.

- **Unmeasurable features.** A feature that is constant over the background and
  the explained samples gets exactly zero SHAP by construction. That means
  "cannot be measured on this data", not "the model ignores it", so those
  features are listed and a warning is raised.
- **Additivity.** SHAP values should sum to ``f(x) - E[f]``. The per-output
  mean / max of ``|sum(SHAP) + E[f] - f(x)|`` is the cheapest way to catch a
  broken model wrapper or explainer.
"""

from __future__ import annotations

import warnings
from typing import Any, Mapping

import numpy as np
import pandas as pd

#: Default relative tolerance for the additivity warning (fraction of the mean
#: |f(x) - E[f]| of that output). Expected gradients is only additive in
#: expectation, so ``gradient`` gets a looser default that still flags a wrapper
#: whose output differs from what was explained.
DEFAULT_ADDITIVITY_TOLERANCE = 0.05
GRADIENT_ADDITIVITY_TOLERANCE = 0.5


def constant_features(X, background=None, feature_names=None) -> list[str]:
    """Return the names of features with zero variance over ``background ∪ X``.

    ``X`` / ``background`` are 2D ``(n, features)`` or 3D
    ``(n, look_back, features)``; for 3D a feature counts as constant only if it
    is constant over every sample *and* time step.
    """
    X = np.asarray(X)
    n_feat = X.shape[-1]
    rows = [X.reshape(-1, n_feat)]
    if background is not None:
        bg = np.asarray(background)
        if bg.ndim >= 2 and bg.shape[-1] == n_feat:
            rows.append(bg.reshape(-1, n_feat))
    stacked = np.concatenate(rows, axis=0)
    try:
        numeric = stacked.astype(float)
        const = np.flatnonzero(np.nanmax(numeric, axis=0) == np.nanmin(numeric, axis=0))
    except (TypeError, ValueError):
        # Non-numeric columns (e.g. categorical features of a tree model).
        const = np.flatnonzero(pd.DataFrame(stacked).nunique(dropna=False).to_numpy() <= 1)
    names = list(feature_names) if feature_names else [f"f{i}" for i in range(n_feat)]
    return [names[j] for j in const]


def additivity_error(shap_values, predictions, expected_value) -> dict[str, float] | None:
    """Return ``{"mean", "max", "relative_max"}`` of ``|Σ SHAP + E[f] − f(x)|`` for one output.

    ``expected_value`` is a scalar, or one value per sample when samples were
    explained in groups with different baselines. ``relative_max`` divides by
    the mean ``|f(x) − E[f]|`` so the tolerance is scale-free. Returns ``None``
    when the inputs can't be compared.
    """
    if predictions is None or expected_value is None:
        return None
    sv = np.asarray(shap_values, dtype=float)
    preds = np.asarray(predictions, dtype=float).ravel()
    if sv.shape[0] != preds.shape[0]:
        return None
    ev = np.asarray(expected_value, dtype=float).ravel()
    if ev.size not in (1, preds.size):
        return None
    total = sv.reshape(sv.shape[0], -1).sum(axis=1)
    err = np.abs(total + ev - preds)
    scale = float(np.mean(np.abs(preds - ev)))
    return {
        "mean": float(err.mean()),
        "max": float(err.max()),
        "relative_max": float(err.max() / scale) if scale > 0 else float(err.max()),
    }


def annotate(
    result,
    *,
    background=None,
    expected_values: Mapping[Any, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    tolerance: float | None = None,
    check_additivity: bool = True,
):
    """Fill the diagnostic fields of ``result`` and warn on problems. Returns ``result``."""
    import shap

    from . import __version__

    result.settings = {
        "ai_explainability_version": __version__,
        "shap_version": shap.__version__,
        **dict(settings or {}),
    }
    if tolerance is None:
        is_gradient = result.settings.get("explainer_type") == "gradient"
        tolerance = GRADIENT_ADDITIVITY_TOLERANCE if is_gradient else DEFAULT_ADDITIVITY_TOLERANCE
    tol = float(tolerance)
    result.settings["additivity_tolerance"] = tol

    if expected_values is not None:
        result.expected_value = {
            k: (float(np.ravel(v)[0]) if np.size(v) == 1 else np.asarray(v, dtype=float))
            for k, v in expected_values.items()
        }

    if result.raw_data_values is not None:
        frozen = set(result.settings.get("frozen_features") or [])
        result.unmeasurable_features = [
            f for f in constant_features(result.raw_data_values, background, result.feature_names)
            if f not in frozen
        ]
        if result.unmeasurable_features:
            shown = result.unmeasurable_features[:10]
            more = len(result.unmeasurable_features) - len(shown)
            warnings.warn(
                f"{len(result.unmeasurable_features)} feature(s) are constant over the "
                f"background and explained data, so their SHAP values are zero by "
                f"construction (not measurable here, not necessarily unimportant): "
                f"{shown}{f' and {more} more' if more else ''}. "
                "See result.unmeasurable_features.",
                stacklevel=3,
            )

    if check_additivity and expected_values:
        errors = {}
        for k, sv in result.shap_values.items():
            err = additivity_error(sv, result.predictions.get(k), expected_values.get(k))
            if err is not None:
                errors[k] = err
        result.additivity_error = errors
        bad = {k: e for k, e in errors.items() if e["relative_max"] > tol}
        if bad:
            worst = max(bad, key=lambda k: bad[k]["relative_max"])
            note = ""
            if result.settings.get("explainer_type") == "gradient":
                note = (" GradientExplainer (expected gradients) is only additive in "
                        "expectation; raise gradient_nsamples or use 'deep'/'kernel'.")
            warnings.warn(
                f"SHAP values deviate from f(x) - E[f] by more than {tol:.0%} for "
                f"{len(bad)} output(s) (worst: {worst!r}, max error "
                f"{bad[worst]['max']:.4g}). This usually points at a model wrapper "
                f"whose output differs from what was explained.{note} "
                "See result.additivity_error.",
                stacklevel=3,
            )
    return result
