"""
In-memory result container returned by :func:`ai_explainability.explain`.

The goal of this object is to keep every piece of information an explainer
produces reachable as a plain Python value, so a caller running inside a
Databricks / Jupyter notebook can ``display(result.to_dataframe())`` or feed
``result.shap_values`` straight into further analysis — without any file I/O.

Disk serialisation (Excel, Jupyter notebook report) is still available via
``save_excel`` / ``save_notebook`` but is now strictly opt-in.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass
class ExplanationResult:
    """Structured, framework-agnostic result of an explainability run.

    Attributes
    ----------
    shap_values:
        Mapping ``{target_index: ndarray}``. For tabular models the array is
        2D with shape ``(n_samples, n_features)``. For LSTM / time-series
        models the array is 3D with shape ``(n_samples, look_back, n_features)``.
    predictions:
        Mapping ``{target_index: ndarray}`` of the model predictions evaluated
        on ``raw_data``. May be empty for explainers that don't produce
        predictions (e.g. ARIMA coefficient inspection).
    raw_data:
        The inputs that were actually explained, as a :class:`pandas.DataFrame`
        when the analysis is tabular. For time-series the underlying ndarray
        is exposed via ``raw_data_values`` instead.
    raw_data_values:
        The numpy view of ``raw_data`` — populated for time-series (3D) where
        a DataFrame is not a natural representation.
    feature_names:
        Column names corresponding to the last axis of ``raw_data_values``.
    analysis:
        ``"tabular"`` or ``"timeseries"``.
    model_type:
        The ``model_type`` string from the configuration (e.g. ``"random_forest"``,
        ``"lstm"``, ``"arima"``).
    extras:
        Catch-all dict for model-specific artefacts that don't fit the generic
        shape, e.g. ARIMA coefficient statistics or the raw SHAP explanation
        object. Callers that need fine-grained access to explainer internals
        should look here.
    chosen_index:
        Set when ``explain(..., targets=...)`` was used. Mapping
        ``{head_name: ndarray (n_samples,)}`` of the output index each head
        selected per sample. ``shap_values`` / ``predictions`` are then keyed
        by head (or family) name and hold only the selected output.
    target_heads:
        ``{head_name: [output indices]}`` the targets were resolved to.
    target_families:
        ``{output_name: [head names]}``; each entry is one explained output
        (a single head, or several heads summed with ``targets_reduce='sum'``).
    feature_groups:
        ``{group: [feature names]}`` or a callable ``feature -> group``. Features
        not in a group are their own group. Dict observations default to one
        group per top-level key. See :attr:`group_shap_values`.
    row_labels:
        Optional callable ``(sample_idx, feature) -> label | None`` naming the
        real element a feature describes in that sample (e.g. which bus a
        table row is). See :meth:`label_importance`.
    cohorts:
        ``{cohort: boolean mask over samples}``. See :meth:`cohort_importance`.
    expected_value:
        ``{output: E[f]}`` — the SHAP baseline per output; an array with one
        value per sample when samples were explained in groups (``targets=`` /
        ``frozen_features=``).
    unmeasurable_features:
        Features constant over the background and explained data. Their SHAP
        is zero by construction: not measurable on this data, which is not the
        same as unimportant.
    additivity_error:
        ``{output: {"mean", "max", "relative_max"}}`` of
        ``|sum(SHAP) + E[f] - f(x)|``. Empty when the explainer's output space
        can't be compared with ``predictions`` (tree classifiers, which SHAP
        checks internally).
    settings:
        Everything needed to reproduce the run: explainer type, background
        source and size, seed, ``nsamples``, frozen features, library versions.
    """

    shap_values: dict[int, np.ndarray] = field(default_factory=dict)
    predictions: dict[int, np.ndarray] = field(default_factory=dict)
    raw_data: pd.DataFrame | None = None
    raw_data_values: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    analysis: str = ""
    model_type: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    # Feature trimming for views/reports (does NOT alter the full arrays in
    # ``shap_values`` — only what to_dataframe / save_excel / reports render).
    # ``select_features``: explicit list of feature names to keep.
    # ``top_features``: keep the N most important features, computed per output.
    select_features: list[str] | None = None
    top_features: int | None = None
    # Per-sample targets (explain(..., targets=...)) — see class docstring.
    chosen_index: dict[str, np.ndarray] | None = None
    target_heads: dict[str, list[int]] | None = None
    target_families: dict[str, list[str]] | None = None
    # Groups, labels and cohorts — see the class docstring.
    feature_groups: Any = None
    row_labels: Any = None
    cohorts: dict[str, np.ndarray] | None = None
    # Diagnostics — see the class docstring and ai_explainability.diagnostics.
    expected_value: dict[Any, Any] = field(default_factory=dict)
    unmeasurable_features: list[str] = field(default_factory=list)
    additivity_error: dict[Any, dict[str, float]] = field(default_factory=dict)
    settings: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Feature selection helper                                            #
    # ------------------------------------------------------------------ #
    def _selected_indices(self, shap_2d, names):
        """Column indices to keep for one output, honoring select/top_features."""
        from output.utils.report_content import select_feature_indices

        return select_feature_indices(
            shap_2d, names,
            top_features=self.top_features,
            select_features=self.select_features,
        )

    # ------------------------------------------------------------------ #
    # Views                                                              #
    # ------------------------------------------------------------------ #
    def to_dataframe(self) -> pd.DataFrame:
        """Flatten the result into a single :class:`pandas.DataFrame`.

        - For tabular: one row per input sample, columns are ``<feature>``,
          ``Model_Prediction_<target>``, ``SHAP_<feature>_t<target>``.
        - For 3D time-series: columns are ``Val_<feature>_t-<k>`` and
          ``SHAP_<feature>_t-<k>`` for ``k`` in ``0..look_back-1``.
        """
        if self.raw_data_values is None:
            raise ValueError("ExplanationResult has no raw_data_values to flatten.")

        arr = self.raw_data_values

        if arr.ndim == 3:
            return self._flatten_3d()

        # 2D tabular path
        features = self.feature_names or [
            f"f{i}" for i in range(arr.shape[1])
        ]

        # Per-output selected feature indices (honors select/top_features).
        selected = {
            idx: self._selected_indices(np.asarray(sa), features)
            for idx, sa in self.shap_values.items()
            if np.asarray(sa).ndim == 2
        }
        # Raw feature columns: union of all selected features (keeps the table
        # narrow when top_features/select_features trims the set).
        if selected:
            keep = sorted({j for idxs in selected.values() for j in idxs})
        else:
            keep = list(range(len(features)))
        base = pd.DataFrame(arr[:, keep], columns=[features[j] for j in keep])

        for idx, preds in self.predictions.items():
            base[f"Model_Prediction_{idx}"] = np.asarray(preds).ravel()[: len(base)]

        for idx, shap_arr in self.shap_values.items():
            sa = np.asarray(shap_arr)
            if sa.ndim == 2 and sa.shape[1] == len(features):
                for j in selected.get(idx, range(len(features))):
                    base[f"SHAP_{features[j]}_t{idx}"] = sa[:, j]
        return base

    def _flatten_3d(self) -> pd.DataFrame:
        arr = self.raw_data_values
        if arr is None or arr.ndim != 3:
            raise ValueError("_flatten_3d expects a 3D raw_data_values array")
        n, look_back, n_feat = arr.shape
        features = self.feature_names or [f"f{i}" for i in range(n_feat)]

        val_cols = [f"Val_{f}_t-{look_back - 1 - k}" for k in range(look_back) for f in features]
        val_df = pd.DataFrame(arr.reshape(n, -1), columns=val_cols)

        out = [val_df]
        for idx, shap_arr in self.shap_values.items():
            sa = np.asarray(shap_arr)
            if sa.ndim == 3:
                shap_cols = [
                    f"SHAP_{f}_t-{look_back - 1 - k}_target{idx}"
                    for k in range(look_back)
                    for f in features
                ]
                out.append(pd.DataFrame(sa.reshape(n, -1), columns=shap_cols))
        return pd.concat(out, axis=1)

    # ------------------------------------------------------------------ #
    # RL convenience: attribute the chosen (argmax) action                #
    # ------------------------------------------------------------------ #
    def chosen_output(self, among=None):
        """Return the SHAP attribution of the *selected* action per sample.

        For a discrete action head (e.g. transformer-tap positions represented
        as several output columns), the agent picks ``argmax`` over those
        outputs. This returns, for each state:

        - ``chosen_idx`` — the output index the agent selected (shape ``(n,)``);
        - ``chosen_shap`` — the SHAP values of that selected output
          (shape ``(n, n_features)``), i.e. "why did it pick *this* action here".

        Parameters
        ----------
        among:
            The output indices that make up the discrete head to argmax over.
            Defaults to every output present in ``predictions``. Pass a subset
            when the network mixes continuous and discrete heads (select only
            the discrete columns).
        """
        if self.chosen_index is not None:
            return self._stored_choice(among)

        keys = list(among) if among is not None else sorted(self.predictions)
        if not keys:
            raise ValueError("No outputs available to choose among.")
        missing = [k for k in keys if k not in self.predictions or k not in self.shap_values]
        if missing:
            raise ValueError(f"Outputs {missing} are not present in this result.")

        pred_stack = np.column_stack(
            [np.asarray(self.predictions[k]).ravel() for k in keys]
        )
        chosen_pos = pred_stack.argmax(axis=1)              # position within `keys`
        chosen_idx = np.array([keys[p] for p in chosen_pos])

        n = pred_stack.shape[0]
        n_feat = np.asarray(self.shap_values[keys[0]]).shape[1]
        chosen_shap = np.empty((n, n_feat), dtype=float)
        for i, pos in enumerate(chosen_pos):
            chosen_shap[i] = np.asarray(self.shap_values[keys[pos]])[i]
        return chosen_idx, chosen_shap

    def _stored_choice(self, among):
        """Return the choice recorded by a ``targets=`` run for the head matching ``among``."""
        heads = self.target_heads or {}
        if among is None:
            if len(heads) != 1:
                raise ValueError(
                    f"This result has several heads {list(heads)}; pass `among`."
                )
            name = next(iter(heads))
        else:
            wanted = sorted(int(c) for c in among)
            name = next((h for h, cols in heads.items() if sorted(cols) == wanted), None)
            if name is None:
                raise ValueError(
                    f"Outputs {wanted} do not match any head explained with targets= "
                    f"({heads})."
                )
        # A family may share a head's name yet hold the summed SHAP of several heads.
        own = (self.target_families or {name: [name]}).get(name) == [name]
        if not own or name not in self.shap_values:
            raise ValueError(
                f"Head {name!r} was summed into a family (targets_reduce='sum'); "
                "its own SHAP values were not computed."
            )
        return np.asarray(self.chosen_index[name]), np.asarray(self.shap_values[name])

    def chosen_actions(self, heads: Mapping[str, Any]):
        """Apply :meth:`chosen_output` to several discrete heads at once.

        For an agent with multiple discrete action heads (e.g. a categorical P,
        a categorical Q, and a categorical transformer tap), pass a mapping of
        head name -> the output indices (logit columns) that belong to it:

        >>> result.chosen_actions({
        ...     "P":   [0, 1, 2],      # P over 3 levels
        ...     "Q":   [3, 4, 5],      # Q over 3 levels
        ...     "tap": [6, 7, 8, 9],   # tap over 4 positions
        ... })

        Returns ``{head_name: (chosen_idx, chosen_shap)}``, where for each head
        ``chosen_idx`` is the selected category (as an output index) per state
        and ``chosen_shap`` is the SHAP of that selection — i.e. "why did the
        agent pick this P / Q / tap here". Works for any number of heads and any
        mix of head sizes; a continuous action is simply a head with one index.
        """
        return {name: self.chosen_output(among=list(cols)) for name, cols in heads.items()}

    # ------------------------------------------------------------------ #
    # Feature groups (SHAP is additive, so a group's SHAP is the sum of   #
    # its members' SHAP per sample)                                       #
    # ------------------------------------------------------------------ #
    def _group_columns(self) -> dict[str, list[int]]:
        """Resolve ``feature_groups`` to ``{group: [column indices]}``.

        Features not in any group form a group of their own, so group tables
        always account for the full attribution.
        """
        names = list(self.feature_names)
        pos = {n: j for j, n in enumerate(names)}
        spec = self.feature_groups
        groups: dict[str, list[int]] = {}
        if callable(spec):
            for j, n in enumerate(names):
                g = spec(n)
                groups.setdefault(str(g) if g is not None else n, []).append(j)
            return groups
        assigned = set()
        for g, members in (spec or {}).items():
            unknown = [m for m in members if m not in pos]
            if unknown:
                raise ValueError(f"feature_groups[{g!r}] has unknown features {unknown[:5]}.")
            cols = [pos[m] for m in members]
            dup = assigned.intersection(cols)
            if dup:
                raise ValueError(
                    f"Features {[names[j] for j in sorted(dup)][:5]} are in more than one group."
                )
            assigned.update(cols)
            groups[str(g)] = cols
        for j, n in enumerate(names):
            if j not in assigned:
                if n in groups:
                    raise ValueError(
                        f"Group {n!r} has the same name as a feature that is in no group; "
                        "rename the group or add the feature to it."
                    )
                groups[n] = [j]
        return groups

    @property
    def group_shap_values(self) -> dict[Any, pd.DataFrame]:
        """``{output: DataFrame (n_samples, n_groups)}`` — per-sample sum of member SHAP."""
        return self._tables("group", list(self.shap_values))

    def _tables(self, level, keys) -> dict[Any, pd.DataFrame]:
        """Per-sample SHAP per group (or feature) for the outputs ``keys`` only."""
        if level not in ("group", "feature"):
            raise ValueError("level must be 'group' or 'feature'.")
        cols = self._group_columns() if level == "group" else None
        tables = {}
        for k in keys:
            v = np.asarray(self.shap_values[k])
            if v.ndim == 3:  # (n, look_back, features): sum over time too
                v = v.sum(axis=1)
            if cols is None:
                tables[k] = pd.DataFrame(v, columns=self.feature_names or None)
            else:
                tables[k] = pd.DataFrame({g: v[:, idx].sum(axis=1) for g, idx in cols.items()})
        return tables

    def _output_keys(self, outputs=None) -> list:
        keys = list(self.shap_values) if outputs is None else list(outputs)
        missing = [k for k in keys if k not in self.shap_values]
        if missing:
            raise ValueError(f"Outputs {missing} are not in this result.")
        return keys

    def _importance(self, level="group", outputs=None, mask=None, tables=None) -> pd.Series:
        """Mean |SHAP| per group (or feature), summed over ``outputs``, on rows ``mask``.

        ``tables`` (from :meth:`_tables`) avoids recomputing the per-sample
        tables when several masks or outputs are evaluated in a row.
        """
        keys = self._output_keys(outputs)
        if tables is None:
            tables = self._tables(level, keys)
        total = None
        for k in keys:
            t = tables[k] if mask is None else tables[k][np.asarray(mask, dtype=bool)]
            imp = t.abs().mean(axis=0)
            total = imp if total is None else total + imp
        return total

    def group_importance(self, outputs=None, normalize=None) -> pd.DataFrame:
        """Mean |group SHAP| — rows are groups, columns are outputs, sorted by total.

        ``normalize='share'`` divides each column by its sum, which makes
        outputs (or models) with different logit scales comparable.
        """
        keys = self._output_keys(outputs)
        tables = self._tables("group", keys)
        table = pd.DataFrame({k: self._importance("group", [k], tables=tables) for k in keys})
        table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
        return _normalize(table, normalize)

    # ------------------------------------------------------------------ #
    # Per-sample labels (rows that mean different elements per sample)    #
    # ------------------------------------------------------------------ #
    def _label_matrix(self) -> np.ndarray:
        """``(n_samples, n_features)`` object array of ``row_labels(sample, feature)``."""
        if self.row_labels is None:
            raise ValueError("No row_labels were given; pass explain(..., row_labels=fn).")
        # Cached per row_labels function, so reassigning it recomputes.
        cached = self.extras.get("_label_matrix")
        if cached is not None and cached[0] is self.row_labels:
            return cached[1]
        n = len(next(iter(self.shap_values.values())))
        mat = np.array(
            [[self.row_labels(i, f) for f in self.feature_names] for i in range(n)],
            dtype=object,
        )
        self.extras["_label_matrix"] = (self.row_labels, mat)
        return mat

    def label_shap_values(self, output) -> pd.DataFrame:
        """Per-sample SHAP summed per real element label — ``(n_samples, n_labels)``.

        Features whose label is ``None`` are left out. An element that doesn't
        appear in a sample gets 0 there.
        """
        labels = self._label_matrix()
        sv = np.asarray(self.shap_values[output]).reshape(labels.shape)
        rows = []
        for i in range(labels.shape[0]):
            acc: dict[str, float] = {}
            for lab, v in zip(labels[i], sv[i]):
                if lab is not None:
                    acc[lab] = acc.get(lab, 0.0) + float(v)
            rows.append(acc)
        return pd.DataFrame(rows).fillna(0.0)

    def label_importance(self, output) -> pd.DataFrame:
        """Per element label: mean |SHAP| over the samples where it appears, and how often."""
        labels = self._label_matrix()
        table = self.label_shap_values(output)
        present = pd.DataFrame(
            [{lab: True for lab in labels[i] if lab is not None} for i in range(labels.shape[0])]
        ).reindex(columns=table.columns).fillna(False).astype(bool)
        out = pd.DataFrame({
            "mean_abs_shap": table.abs().where(present).mean(axis=0),
            "n_samples": present.sum(axis=0),
        })
        return out.sort_values("mean_abs_shap", ascending=False)

    # ------------------------------------------------------------------ #
    # Cohorts (subsets of states)                                         #
    # ------------------------------------------------------------------ #
    def by_cohort(self, labels) -> "ExplanationResult":
        """Return a copy of this result with cohorts set.

        ``labels`` is one label per sample (``['day', 'night', ...]``) or a
        mapping ``{cohort: boolean mask}``; masks may overlap.
        """
        import dataclasses

        return dataclasses.replace(self, cohorts=_resolve_cohorts(labels, self._n_samples()))

    def _n_samples(self) -> int:
        if self.raw_data_values is not None:
            return len(self.raw_data_values)
        return len(next(iter(self.shap_values.values())))

    def cohort_importance(self, level="group", outputs=None, normalize=None) -> pd.DataFrame:
        """Mean |SHAP| per group (or feature) and cohort; an ``all`` column covers every sample."""
        keys = self._output_keys(outputs)
        tables = self._tables(level, keys)
        cols = {"all": self._importance(level, keys, tables=tables)}
        for name, mask in (self.cohorts or {}).items():
            if np.asarray(mask).any():
                cols[name] = self._importance(level, keys, mask, tables)
        table = pd.DataFrame(cols)
        table = table.loc[table["all"].sort_values(ascending=False).index]
        return _normalize(table, normalize)

    def plot_cohort_heatmap(self, level="group", normalize="share", top=20):
        """Heatmap of :meth:`cohort_importance`, one panel per output (head family)."""
        from .comparison import plot_heatmap

        tables = {
            str(k): self.cohort_importance(level, [k], normalize).head(top)
            for k in self.shap_values
        }
        return plot_heatmap(tables)

    # ------------------------------------------------------------------ #
    # Optional disk persistence                                          #
    # ------------------------------------------------------------------ #
    def save_excel(self, path: str | os.PathLike) -> str:
        """Write a multi-sheet ``.xlsx`` mirroring the legacy audit format.

        One sheet per target, columns ``<features> | Model_Prediction | SHAP_*``.
        Returns the absolute path written.
        """
        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        with pd.ExcelWriter(path) as writer:
            for idx, shap_arr in self.shap_values.items():
                sheet_name = f"target_{idx}"[:31]
                sa = np.asarray(shap_arr)

                if sa.ndim == 3:  # time-series case — flatten per sample
                    look_back = sa.shape[1]
                    features = self.feature_names or [f"f{i}" for i in range(sa.shape[2])]
                    flat_val = self.raw_data_values.reshape(sa.shape[0], -1)
                    flat_shap = sa.reshape(sa.shape[0], -1)
                    val_cols = [f"Val_{f}_t-{look_back - 1 - k}" for k in range(look_back) for f in features]
                    shap_cols = [f"SHAP_{f}_t-{look_back - 1 - k}" for k in range(look_back) for f in features]
                    sheet_df = pd.concat(
                        [
                            pd.DataFrame(flat_val, columns=val_cols),
                            pd.DataFrame(flat_shap, columns=shap_cols),
                        ],
                        axis=1,
                    )
                else:  # tabular 2D
                    features = list(self.raw_data.columns) if self.raw_data is not None else list(self.feature_names)
                    # Trim to selected / top features for this output.
                    sel = self._selected_indices(sa, features) if sa.ndim == 2 else list(range(len(features)))
                    df_features = (
                        self.raw_data.reset_index(drop=True)
                        if self.raw_data is not None
                        else pd.DataFrame(self.raw_data_values, columns=features)
                    )
                    sheet_df = df_features.iloc[:, sel].copy()
                    preds_series = self.predictions.get(idx)
                    if preds_series is not None:
                        sheet_df["Model_Prediction"] = np.asarray(preds_series).ravel()[: len(sheet_df)]
                    for j in sel:
                        if sa.ndim == 2 and j < sa.shape[1]:
                            sheet_df[f"SHAP_{features[j]}"] = sa[:, j]

                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Summary sheets: group importance per output, and per cohort.
            if self.feature_groups is not None:
                self.group_importance().to_excel(writer, sheet_name="group_importance")
            if self.cohorts:
                level = "group" if self.feature_groups is not None else "feature"
                self.cohort_importance(level=level).to_excel(writer, sheet_name="cohort_importance")

        return os.path.abspath(path)

    def save_notebook(
        self,
        path: str | os.PathLike,
        *,
        config: Mapping[str, Any] | None = None,
    ) -> str:
        """Render the Jupyter notebook report using the legacy generator.

        The generator is imported lazily so importing this module doesn't force
        ``nbformat`` / ``nbconvert`` into memory.
        """
        # Local import — delay the heavy nbconvert/nbformat chain until needed.
        from output.utils.report_gen import generate_notebook

        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        class _Shim:
            pass

        shim = _Shim()
        shim.config = dict(config or {})
        shim.config.setdefault("analysis", self.analysis)
        shim.config.setdefault("model_type", self.model_type)
        shim.config.setdefault("feature_names", self.feature_names)
        # Carry feature trimming into the notebook generator.
        if self.top_features is not None:
            shim.config.setdefault("top_features", self.top_features)
        if self.select_features is not None:
            shim.config.setdefault("select_features", self.select_features)
        shim.all_shap_values = self.shap_values
        shim.shap_values = next(iter(self.shap_values.values()), None)
        shim.raw_data_values = self.raw_data_values

        generate_notebook(
            explainer_inst=shim,
            all_shap_values=self.shap_values,
            raw_data=self.raw_data_values,
            output_path=path,
        )
        return os.path.abspath(path)

    def save_report_html(
        self,
        path: str | os.PathLike,
        *,
        output_labels: Any = None,
        max_outputs: int | None = None,
        title: str = "SHAP Interpretation Report",
        heads: Mapping[str, Any] | None = None,
        figures: tuple[str, ...] = (),
        max_categories: int = 10,
    ) -> str:
        """Render a self-contained HTML report — **no Jupyter kernel required**.

        This produces the SAME sections as the ``.ipynb`` report (beeswarm,
        global importance bar, temporal relevance for LSTMs, and top-5 drivers,
        with identical narrative text), because both reports draw their plots and
        wording from ``output.utils.report_content``. The difference is only the
        delivery: plots are rendered in-process with matplotlib and embedded as
        base64 PNGs, so no Jupyter kernel is launched and the pyzmq/kernel
        failures that can affect the notebook cannot occur. Recommended for
        multi-output / RL policies and for Windows users.

        Parameters
        ----------
        output_labels:
            Optional ``{idx: name}`` dict or list to title each output section.
        max_outputs:
            Cap the number of output sections rendered. ``None`` renders all.
        heads:
            ``{head: [output indices]}`` for a result explained on every output:
            renders one section per action head (the chosen category's SHAP,
            via :meth:`chosen_output`) instead of one per logit. Results from
            ``explain(targets=...)`` get head sections automatically.
        figures:
            Optional extra figures on the summary page: ``"group_heatmap"``
            (group share per output) and ``"cohort_heatmap"`` (group share per
            cohort, one panel per output).
        max_categories:
            Per head, how many chosen categories get a collapsible detail table.

        The report opens with a summary page: warnings (unmeasurable features,
        additivity), group or feature importance overall and per cohort, and
        the run settings.
        """
        import base64
        import html as _html
        import io as _io

        import matplotlib
        matplotlib.use("Agg")  # headless — no display / no kernel
        import matplotlib.pyplot as plt

        from output.utils import report_content as rc

        path = str(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Integer outputs in index order; named heads keep their own order.
        out_keys = list(self.shap_values)
        if all(isinstance(k, (int, np.integer)) for k in out_keys):
            out_keys = sorted(out_keys)
        if max_outputs is not None:
            out_keys = out_keys[:max_outputs]

        def _label(idx):
            is_int = isinstance(idx, (int, np.integer))
            if isinstance(output_labels, dict):
                found = output_labels.get(str(idx)) or output_labels.get(idx)
                if found:
                    return str(found)
            elif isinstance(output_labels, (list, tuple)) and is_int and idx < len(output_labels):
                return str(output_labels[idx])
            # Named heads (explain(targets=...)) are labelled by their own name.
            return f"output_{idx}" if is_int else str(idx)

        def _flatten(idx):
            """Return (shap_2d, data_2d, names, raw_3d_or_None, look_back)."""
            sv = np.asarray(self.shap_values[idx])
            rd = self.raw_data_values
            if sv.ndim == 3:  # LSTM (n, look_back, feat)
                n, lb, nf = sv.shape
                feats = self.feature_names or [f"f{i}" for i in range(nf)]
                names = [f"{f}_t-{lb - 1 - k}" for k in range(lb) for f in feats]
                data2 = np.asarray(rd).reshape(n, -1) if rd is not None else None
                return sv.reshape(n, -1), data2, names, sv, lb
            feats = self.feature_names or [f"f{i}" for i in range(sv.shape[1])]
            data2 = np.asarray(rd) if rd is not None else None
            return sv, data2, list(feats), None, None

        def _b64(fig):
            buf = _io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode("ascii")

        def _md(text):
            """Tiny markdown -> HTML for the shared narrative strings."""
            out, bullets = [], []
            for line in text.split("\n"):
                line = line.strip()
                while "**" in line:
                    line = line.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
                line = line.replace("`", "<code>", 1).replace("`", "</code>", 1) if line.count("`") >= 2 else line
                if line.startswith("* "):
                    bullets.append(f"<li>{line[2:]}</li>")
                else:
                    if bullets:
                        out.append("<ul>" + "".join(bullets) + "</ul>"); bullets = []
                    if line:
                        out.append(f"<p>{line}</p>")
            if bullets:
                out.append("<ul>" + "".join(bullets) + "</ul>")
            return "".join(out)

        def _plot_block(title_txt, text, fig):
            return (f'<div class="plot"><h3>{_html.escape(title_txt)}</h3>'
                    f'{_md(text)}<img src="data:image/png;base64,{_b64(fig)}"/></div>')

        def _table(df, fmt="{:.3f}"):
            """Small DataFrame -> HTML table (index as first column)."""
            cols = [" / ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in df.columns]
            head = "".join(f"<th>{_html.escape(c)}</th>" for c in [""] + cols)
            rows = []
            for idx, r in df.iterrows():
                cells = "".join(
                    f"<td>{fmt.format(v) if isinstance(v, (float, np.floating)) else _html.escape(str(v))}</td>"
                    for v in r.to_numpy()
                )
                rows.append(f"<tr><th>{_html.escape(str(idx))}</th>{cells}</tr>")
            return f'<table class="tbl"><tr>{head}</tr>{"".join(rows)}</table>'

        def _img(fig):
            return f'<img src="data:image/png;base64,{_b64(fig)}"/>'

        # ---------------- Summary page ---------------- #
        has_groups = self.feature_groups is not None
        level = "group" if has_groups else "feature"
        summary = []
        warn_items = []
        if self.unmeasurable_features:
            shown = ", ".join(_html.escape(f) for f in self.unmeasurable_features[:20])
            more = len(self.unmeasurable_features) - 20
            warn_items.append(
                f"<li><b>{len(self.unmeasurable_features)} unmeasurable feature(s)</b>: constant "
                f"over the background and data, so their SHAP is zero by construction "
                f"(not necessarily unimportant): {shown}{f' and {more} more' if more > 0 else ''}.</li>")
        tol = self.settings.get("additivity_tolerance")
        bad = {k: e for k, e in self.additivity_error.items()
               if tol is not None and e["relative_max"] > tol}
        if bad:
            worst = max(bad, key=lambda k: bad[k]["relative_max"])
            warn_items.append(
                f"<li><b>Additivity</b>: SHAP deviates from f(x) − E[f] by more than "
                f"{tol:.0%} for {len(bad)} output(s); worst {_html.escape(_label(worst))} "
                f"(max error {bad[worst]['max']:.3g}). Check the model wrapper.</li>")
        if warn_items:
            summary.append(f'<div class="warn"><h3>Warnings</h3><ul>{"".join(warn_items)}</ul></div>')
        elif self.settings:
            additivity = ("SHAP values add up to the model output within tolerance."
                          if self.additivity_error
                          else "additivity was not checked for this explainer.")
            summary.append(f'<div class="ok">No constant features; {additivity}</div>')

        imp = self.cohort_importance(level=level, normalize="share").head(25)
        what = "group" if has_groups else "feature"
        summary.append(
            f"<h3>{what.capitalize()} importance (share of total, all outputs)</h3>"
            f"<p class='note'>Mean |SHAP| per {what}, summed over the explained outputs and "
            f"normalised so each column sums to 1. "
            + ("Columns after <i>all</i> are cohorts. " if self.cohorts else "")
            + ("Group SHAP is the per-sample sum of its members' SHAP." if has_groups else "")
            + "</p>" + _table(imp))
        if has_groups and 1 < len(self.shap_values) <= 12:
            per_out = self.group_importance(normalize="share").head(25)
            per_out.columns = [_label(k) for k in per_out.columns]
            summary.append("<h3>Group importance per output (share)</h3>" + _table(per_out))
        if "group_heatmap" in figures and has_groups:
            from .comparison import plot_heatmap

            gi = self.group_importance(normalize="share").head(25)
            gi = gi.iloc[:, :30]
            gi.columns = [_label(k) for k in gi.columns]
            summary.append("<h3>Group heatmap (share per output)</h3>"
                           + _img(plot_heatmap(gi, title="Group importance per output")))
        if "cohort_heatmap" in figures and self.cohorts:
            summary.append("<h3>Cohort heatmap (share, one panel per output)</h3>"
                           + _img(self.plot_cohort_heatmap(level=level)))
        if self.settings:
            rows = pd.DataFrame({"value": {k: str(v) for k, v in self.settings.items()}})
            summary.append("<details><summary>Run settings</summary>" + _table(rows) + "</details>")

        # ---------------- Output / head sections ---------------- #
        # A view is one section: an output key, the SHAP to plot, and (for
        # discrete heads) which category each head chose per sample.
        views = []
        if self.chosen_index is not None:
            for k in out_keys:
                members = (self.target_families or {}).get(k, [k])
                views.append((k, _label(k), None,
                              {h: self.chosen_index[h] for h in members if h in self.chosen_index}))
        elif heads:
            for name, cols in list(heads.items())[:max_outputs]:
                idx, shap_c = self.chosen_output(among=list(cols))
                views.append((name, str(name), shap_c, {str(name): idx}))
        else:
            views = [(k, _label(k), None, None) for k in out_keys]

        def _category_details(shap_2d, names, chosen):
            """Collapsible per-category tables: top features when that category was chosen."""
            out = []
            for h, idx in chosen.items():
                idx = np.asarray(idx)
                cats, counts = np.unique(idx, return_counts=True)
                order = np.argsort(-counts)[:max_categories]
                dist = pd.DataFrame(
                    {"chosen in": [f"{c} / {len(idx)}" for c in counts[order]],
                     "share": counts[order] / len(idx)},
                    index=[_label(int(c)) for c in cats[order]])
                parts = [f"<h3>Head {_html.escape(str(h))}: chosen categories</h3>" + _table(dist)]
                for c in cats[order]:
                    mask = idx == c
                    mean_abs = np.abs(shap_2d[mask]).mean(axis=0)
                    top = np.argsort(-mean_abs)[:10]
                    t = pd.DataFrame({"mean |SHAP|": mean_abs[top]}, index=[names[j] for j in top])
                    parts.append(
                        f"<details><summary>{_html.escape(_label(int(c)))} "
                        f"&mdash; chosen in {int(mask.sum())} sample(s)</summary>"
                        f"<p class='note'>Top features when this category was chosen.</p>"
                        + _table(t) + "</details>")
                out.append("".join(parts))
            return "".join(out)

        toc, sections = [], []
        for n_view, (key, label, shap_override, chosen) in enumerate(views):
            if shap_override is not None:
                s_flat = np.asarray(shap_override)
                d_flat = np.asarray(self.raw_data_values) if self.raw_data_values is not None else None
                names = list(self.feature_names or [f"f{i}" for i in range(s_flat.shape[1])])
                raw3d = look_back = None
            else:
                s_flat, d_flat, names, raw3d, look_back = _flatten(key)
            full_shap, full_names = s_flat, names
            # Trim to the selected / top features for this output.
            sel = self._selected_indices(s_flat, names)
            if len(sel) != len(names):
                s_flat = s_flat[:, sel]
                d_flat = d_flat[:, sel] if d_flat is not None else None
                names = [names[j] for j in sel]
                # Temporal plot needs the 3D array; skip it when trimming an LSTM
                # (time-step aggregation spans all features), to avoid confusion.
                raw3d = None
            blocks = [
                _plot_block(rc.BEESWARM_TITLE, rc.BEESWARM_TEXT,
                            rc.fig_beeswarm(s_flat, d_flat, names, f"Impact Distribution: {label}")),
                _plot_block(rc.BAR_TITLE, rc.BAR_TEXT,
                            rc.fig_bar(s_flat, d_flat, names, f"Mean Influence: {label}")),
            ]
            if raw3d is not None:
                blocks.append(_plot_block(rc.TEMPORAL_TITLE, rc.TEMPORAL_TEXT,
                                          rc.fig_temporal(raw3d, look_back, f"Temporal Relevance: {label}")))
            blocks.append(_plot_block(rc.TOP5_TITLE, rc.TOP5_TEXT,
                                      rc.fig_top5(s_flat, names, f"Top 5 Drivers: {label}")))
            if chosen:
                blocks.append(_category_details(full_shap, full_names, chosen))
            anchor = f"out{n_view}"
            heading = "Head" if chosen else "Analysis for"
            toc.append(f'<li><a href="#{anchor}">{_html.escape(label)}</a></li>')
            sections.append(f'<section id="{anchor}"><h2>{heading}: {_html.escape(label)}</h2>'
                            + "".join(blocks) + "</section>")

        n_samples = (np.asarray(self.raw_data_values).shape[0]
                     if self.raw_data_values is not None else "?")
        meta = (f"analysis: <b>{_html.escape(self.analysis)}</b> &nbsp;&middot;&nbsp; "
                f"model: <b>{_html.escape(self.model_type)}</b> &nbsp;&middot;&nbsp; "
                f"sections: <b>{len(views)}</b> &nbsp;&middot;&nbsp; "
                f"samples: <b>{n_samples}</b> &nbsp;&middot;&nbsp; "
                f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        intro = (
            "<p><strong>SHAP (SHapley Additive exPlanations)</strong> decomposes each "
            "prediction into the contribution of every input feature. A positive SHAP "
            "value pushes the output higher; a negative one pushes it lower. The "
            "summary comes first; each later section explains one "
            + ("action head (the category the agent chose)" if any(v[3] for v in views)
               else "model output") + ".</p>"
        )
        html_doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/>
<title>{_html.escape(title)}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;color:#1f2d3d;background:#fff}}
 header{{background:#2c3e50;color:#fff;padding:20px 28px}}
 header h1{{margin:0 0 6px;font-size:22px}} header .meta{{font-size:13px;color:#cfd8e3}}
 .wrap{{max-width:900px;margin:0 auto;padding:24px 28px}}
 .intro{{font-size:14px;color:#334155;margin-bottom:18px}}
 nav{{background:#f5f7fa;border:1px solid #e2e8f0;border-radius:8px;padding:12px 18px;margin-bottom:24px}}
 nav h3{{margin:0 0 8px;font-size:13px;text-transform:uppercase;color:#64748b}}
 nav ul{{columns:3;margin:0;padding-left:18px;font-size:13px}}
 section{{border-top:2px solid #e2e8f0;padding:18px 0}}
 section h2{{font-size:18px;margin:0 0 12px;color:#2c3e50}}
 .plot{{margin:14px 0}} .plot h3{{font-size:15px;margin:0 0 4px}}
 .plot p{{margin:4px 0;font-size:13px;color:#475569}} .plot ul{{margin:4px 0 8px 18px;font-size:13px;color:#475569}}
 .plot img{{max-width:100%;width:640px;border:1px solid #edf0f3;border-radius:6px;display:block;margin-top:8px}}
 footer{{color:#94a3b8;font-size:12px;padding:16px 28px;border-top:1px solid #e2e8f0}}
 .summary h3{{font-size:15px;margin:18px 0 6px}} .note{{font-size:12px;color:#64748b;margin:2px 0 6px}}
 .tbl{{border-collapse:collapse;font-size:12px;margin:6px 0 10px}}
 .tbl th,.tbl td{{border:1px solid #e2e8f0;padding:3px 8px;text-align:right}} .tbl th{{background:#f5f7fa;text-align:left}}
 .warn{{background:#fff7ed;border:1px solid #fdba74;border-radius:8px;padding:8px 16px;font-size:13px}}
 .warn h3{{margin:4px 0}} .ok{{background:#f0fdf4;border:1px solid #86efac;border-radius:8px;padding:8px 16px;font-size:13px}}
 details{{margin:4px 0;font-size:13px}} summary{{cursor:pointer;color:#2c3e50}}
 section img, .summary img{{max-width:100%;border:1px solid #edf0f3;border-radius:6px}}
</style></head>
<body>
<header><h1>{_html.escape(title)}</h1><div class="meta">{meta}</div></header>
<div class="wrap">
<div class="intro">{intro}</div>
<nav><h3>Sections</h3><ul><li><a href="#summary">Summary</a></li>{''.join(toc)}</ul></nav>
<section id="summary" class="summary"><h2>Summary</h2>{''.join(summary)}</section>
{''.join(sections)}
</div>
<footer>Generated by ai-explainability &middot; kernel-free HTML report (same sections as the .ipynb report).</footer>
</body></html>"""

        with open(path, "w", encoding="utf-8") as f:
            f.write(html_doc)
        return os.path.abspath(path)

    # ------------------------------------------------------------------ #
    # Convenience                                                        #
    # ------------------------------------------------------------------ #
    def timestamped_filename(self, stem: str, ext: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{stem}_{self.model_type}_{ts}.{ext.lstrip('.')}"


# ---------------------------------------------------------------------- #
# Module helpers                                                          #
# ---------------------------------------------------------------------- #
def _normalize(table: pd.DataFrame, normalize) -> pd.DataFrame:
    """``None`` keeps raw values; ``'share'`` divides each column by its sum."""
    if normalize is None:
        return table
    if normalize != "share":
        raise ValueError(f"normalize must be None or 'share'; got {normalize!r}.")
    sums = table.sum(axis=0).replace(0, np.nan)
    return table / sums


def _resolve_cohorts(labels, n: int) -> dict[str, np.ndarray]:
    """``{name: mask}`` or one label per sample -> ``{name: bool mask of length n}``."""
    if isinstance(labels, Mapping):
        out = {}
        for name, mask in labels.items():
            mask = np.asarray(mask, dtype=bool).ravel()
            if len(mask) < n:
                raise ValueError(f"Cohort {name!r} mask has {len(mask)} entries, expected {n}.")
            # Longer masks are allowed: explain() may have explained a prefix
            # of the data (dataset_scope='subset').
            out[str(name)] = mask[:n]
        return out
    arr = np.asarray(labels).ravel()
    if len(arr) < n:
        raise ValueError(f"Got {len(arr)} cohort labels for {n} samples.")
    arr = arr[:n]
    return {str(v): arr == v for v in pd.unique(arr)}
