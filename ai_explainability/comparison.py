"""
Comparing explanations across agents and cohorts.

:func:`compare` turns several :class:`ExplanationResult` objects (e.g. two agents
on the same grid) into one group × (agent, cohort) table. Logit scales differ
between networks, often by orders of magnitude, so by default every column is
normalised to shares of its total importance.

:func:`plot_heatmap` draws such a table (or several, one panel each).
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def compare(
    results: Mapping[str, Any],
    *,
    groups: Any = None,
    normalize: str | None = "share",
    level: str = "group",
    outputs: Any = None,
) -> pd.DataFrame:
    """Importance table with rows = groups (or features), columns = (agent, cohort).

    Parameters
    ----------
    results:
        ``{agent_name: ExplanationResult}``. Each result's ``cohorts`` (if any)
        add columns next to its ``all`` column.
    groups:
        Feature groups applied to every result, overriding their own
        ``feature_groups`` (``{group: [features]}`` or ``feature -> group``).
    normalize:
        ``'share'`` (default): each column sums to 1. ``None``: raw mean |SHAP|.
    level:
        ``'group'`` or ``'feature'``.
    outputs:
        Output keys summed into the importance; default all outputs of each
        result. Keys that a result doesn't have are an error.
    """
    import dataclasses

    columns = {}
    for agent, res in results.items():
        if groups is not None:
            res = dataclasses.replace(res, feature_groups=groups)
        table = res.cohort_importance(level=level, outputs=outputs, normalize=normalize)
        for cohort in table.columns:
            columns[(str(agent), cohort)] = table[cohort]
    out = pd.DataFrame(columns)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["agent", "cohort"])
    out = out.fillna(0.0)
    return out.loc[out.mean(axis=1).sort_values(ascending=False).index]


def plot_heatmap(tables, *, title: str | None = None, cmap: str = "viridis", annotate: bool = True):
    """Heatmap of an importance table, or ``{panel_title: table}`` for one panel each.

    Rows are groups or features, columns are cohorts (or (agent, cohort)
    pairs). Returns the matplotlib figure.
    """
    import matplotlib.pyplot as plt

    if isinstance(tables, pd.DataFrame):
        tables = {title or "": tables}
    panels = list(tables.items())
    n_rows = max(len(t) for _, t in panels)
    width = sum(1.5 + 0.9 * t.shape[1] for _, t in panels)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(max(5.0, width + 3), 1.2 + 0.35 * n_rows), squeeze=False
    )
    for ax, (name, table) in zip(axes[0], panels):
        values = table.to_numpy(dtype=float)
        im = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_yticks(range(len(table.index)), [str(i) for i in table.index], fontsize=8)
        ax.set_xticks(
            range(table.shape[1]),
            [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in table.columns],
            rotation=45, ha="right", fontsize=8,
        )
        if annotate and values.size <= 400:
            vmax = np.nanmax(values) if np.isfinite(values).any() else 0
            for (r, c), v in np.ndenumerate(values):
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2g}", ha="center", va="center", fontsize=7,
                            color="white" if v < 0.6 * vmax else "black")
        ax.set_title(str(name), fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if title and len(panels) > 1:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
