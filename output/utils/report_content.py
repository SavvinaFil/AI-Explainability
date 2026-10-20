"""
Single source of truth for report CONTENT — the section texts and the plot
figures — shared by BOTH report backends so they have an identical layout:

- ``output/utils/report_gen.py``  -> the executed Jupyter (.ipynb) report;
- ``ai_explainability.ExplanationResult.save_report_html`` -> the kernel-free
  HTML report.

Each plot helper returns a Matplotlib ``Figure`` and never calls ``plt.show()``;
the notebook cells call ``plt.show()`` themselves, while the HTML report saves
the figure to a PNG. The beeswarm/bar helpers use ``shap.summary_plot`` (the
same call the notebook has always used) with a lightweight Matplotlib fallback
so a report can still render if ``shap`` is unavailable.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Section narrative (identical wording in both reports)                        #
# --------------------------------------------------------------------------- #
BEESWARM_TEXT = (
    "**What is this?** A distribution of SHAP values for every sample in the "
    "dataset.\n**What to focus on:**\n"
    "* **Horizontal Position:** Points to the right increase the model output; "
    "points to the left decrease it.\n"
    "* **Color:** Represents the feature value (**Red** is high, **Blue** is low).\n"
    "* **Insight:** If Red points are on the right, the feature has a positive "
    "correlation with the target."
)
BAR_TEXT = (
    "**What is this?** The mean absolute SHAP value for each feature.\n"
    "**What to focus on:** The length of the bar represents the global influence "
    "of a feature — how much it moves the prediction on average, regardless of "
    "direction."
)
TEMPORAL_TEXT = (
    "**What is this?** A look at which time-steps in the look-back window are "
    "most influential.\n**What to focus on:** Does the model care more about the "
    "immediate past (`t-0`, `t-1`) or older history?"
)
TOP5_TEXT = (
    "**What is this?** A high-precision look at the five most critical variables "
    "for this output.\n**What to focus on:** The gap between the 1st and 5th "
    "feature. If the 1st is much larger, the model is heavily reliant on a single "
    "variable."
)

# Section titles (shared).
BEESWARM_TITLE = "1. Feature Impact Distribution (Beeswarm)"
BAR_TITLE = "2. Global Feature Importance"
TEMPORAL_TITLE = "3. Temporal Relevance"
TOP5_TITLE = "4. Focused View: Top 5 Drivers"


# --------------------------------------------------------------------------- #
# Feature selection (shared) — trim which features appear in plots & results   #
# --------------------------------------------------------------------------- #
def select_feature_indices(shap_2d, names, top_features=None, select_features=None):
    """Return the column indices of the features to keep, in display order.

    - ``select_features``: an explicit list of feature names to keep (in that
      order). Names not present are ignored. Takes precedence.
    - ``top_features``: keep the N most important features by mean |SHAP|
      (descending). Computed **per output**, so different outputs can surface
      different drivers.
    - neither: keep all features (original order).

    This is used by BOTH reports and by ``ExplanationResult`` views so trimming
    is consistent everywhere.
    """
    names = list(names)
    if select_features:
        idx = [names.index(n) for n in select_features if n in names]
        return idx if idx else list(range(len(names)))
    if top_features:
        mean_abs = np.abs(np.asarray(shap_2d)).mean(axis=0)
        return [int(i) for i in np.argsort(mean_abs)[::-1][: int(top_features)]]
    return list(range(len(names)))


# --------------------------------------------------------------------------- #
# Figure helpers (shared). Return a Figure; do NOT call plt.show().           #
# --------------------------------------------------------------------------- #
def _new_fig(figsize):
    import matplotlib.pyplot as plt

    return plt.figure(figsize=figsize)


import contextlib


@contextlib.contextmanager
def _quiet_shap():
    """Suppress benign warnings emitted from inside ``shap.summary_plot``.

    Recent ``shap`` seeds the NumPy *global* RNG (for the beeswarm jitter),
    which newer NumPy flags with a ``FutureWarning``. It's harmless and comes
    from shap's internals, not our code, so we filter it around the call.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*NumPy global RNG was seeded.*")
        warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
        yield


def fig_beeswarm(shap_2d, data_2d, feature_names, title):
    """Beeswarm summary — shap.summary_plot, matplotlib fallback."""
    import matplotlib.pyplot as plt

    fig = _new_fig((10, 6))
    try:
        import shap

        with _quiet_shap():
            shap.summary_plot(
                np.asarray(shap_2d), data_2d, feature_names=list(feature_names),
                show=False, plot_size=None,
            )
    except Exception:
        _fallback_beeswarm(np.asarray(shap_2d), data_2d, list(feature_names))
    plt.title(title, fontsize=14, pad=20)
    return fig


def fig_bar(shap_2d, data_2d, feature_names, title):
    """Global importance bar — shap.summary_plot(plot_type='bar'), fallback."""
    import matplotlib.pyplot as plt

    fig = _new_fig((10, 6))
    try:
        import shap

        with _quiet_shap():
            shap.summary_plot(
                np.asarray(shap_2d), data_2d, feature_names=list(feature_names),
                plot_type="bar", show=False, color="#34495e", plot_size=None,
            )
    except Exception:
        _fallback_bar(np.asarray(shap_2d), list(feature_names))
    plt.title(title, fontsize=14, pad=20)
    plt.grid(axis="x", linestyle="--", alpha=0.6)
    return fig


def fig_temporal(shap_3d, look_back, title):
    """Temporal relevance (LSTM): mean|SHAP| per time step."""
    import matplotlib.pyplot as plt

    arr = np.asarray(shap_3d)
    importance_per_step = np.abs(arr).mean(axis=(0, 2)).flatten()
    time_labels = [f"t-{i}" for i in range(look_back - 1, -1, -1)]
    time_axis = list(range(len(time_labels)))
    fig = _new_fig((10, 4))
    plt.plot(time_axis, importance_per_step, marker="o", linewidth=2, color="#e67e22")
    plt.xticks(time_axis, time_labels)
    plt.title(title, fontsize=14)
    plt.xlabel("Timeline (History -> Present)")
    plt.fill_between(time_axis, importance_per_step, alpha=0.1, color="#e67e22")
    plt.grid(True, alpha=0.3)
    return fig


def fig_top5(shap_2d, feature_names, title):
    """Top-5 drivers by mean|SHAP|."""
    import matplotlib.pyplot as plt

    s_flat = np.asarray(shap_2d)
    names = list(feature_names)
    mean_shap = np.abs(s_flat).mean(axis=0)
    sorted_idx = np.argsort(mean_shap)[-5:]
    fig = _new_fig((10, 5))
    plt.barh([names[i] for i in sorted_idx], mean_shap[sorted_idx], color="#2c3e50")
    plt.title(title, fontsize=14)
    for i, v in enumerate(mean_shap[sorted_idx]):
        plt.text(v, i, f"  {v:.4f}", va="center", fontweight="bold")
    plt.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Matplotlib fallbacks (used only when shap can't render)                     #
# --------------------------------------------------------------------------- #
def _fallback_beeswarm(shap_2d, data_2d, names, top=15):
    import matplotlib.pyplot as plt

    mean_abs = np.abs(shap_2d).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][: min(top, len(names))]
    ax = plt.gca()
    sc = None
    for row, fi in enumerate(order[::-1]):
        x = shap_2d[:, fi]
        jitter = (np.random.RandomState(int(fi)).rand(len(x)) - 0.5) * 0.6
        y = np.full(len(x), row, dtype=float) + jitter
        if data_2d is not None and np.asarray(data_2d).shape[1] > fi:
            sc = ax.scatter(x, y, c=np.asarray(data_2d)[:, fi], cmap="coolwarm", s=10, alpha=0.7)
        else:
            ax.scatter(x, y, s=10, alpha=0.7, color="#3498db")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order[::-1]])
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_xlabel("SHAP value")
    if sc is not None:
        plt.colorbar(sc, ax=ax, label="feature value (red=high, blue=low)")


def _fallback_bar(shap_2d, names, top=15):
    import matplotlib.pyplot as plt

    mean_abs = np.abs(shap_2d).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][: min(top, len(names))]
    ax = plt.gca()
    ax.barh([names[i] for i in order][::-1], mean_abs[order][::-1], color="#34495e")
    ax.set_xlabel("mean(|SHAP|)")
