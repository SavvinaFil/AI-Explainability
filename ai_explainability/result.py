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

        out_keys = sorted(self.shap_values)
        if max_outputs is not None:
            out_keys = out_keys[:max_outputs]

        def _label(idx):
            if isinstance(output_labels, dict):
                return str(output_labels.get(str(idx)) or output_labels.get(idx) or f"output_{idx}")
            if isinstance(output_labels, (list, tuple)) and isinstance(idx, (int, np.integer)) and idx < len(output_labels):
                return str(output_labels[idx])
            return f"output_{idx}"

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

        toc, sections = [], []
        for idx in out_keys:
            s_flat, d_flat, names, raw3d, look_back = _flatten(idx)
            # Trim to the selected / top features for this output.
            sel = self._selected_indices(s_flat, names)
            if len(sel) != len(names):
                s_flat = s_flat[:, sel]
                d_flat = d_flat[:, sel] if d_flat is not None else None
                names = [names[j] for j in sel]
                # Temporal plot needs the 3D array; skip it when trimming an LSTM
                # (time-step aggregation spans all features), to avoid confusion.
                raw3d = None if raw3d is not None else raw3d
            label = _label(idx)
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
            anchor = f"out{idx}"
            toc.append(f'<li><a href="#{anchor}">{_html.escape(label)}</a></li>')
            sections.append(f'<section id="{anchor}"><h2>Analysis for: {_html.escape(label)}</h2>'
                            + "".join(blocks) + "</section>")

        n_samples = (np.asarray(self.raw_data_values).shape[0]
                     if self.raw_data_values is not None else "?")
        meta = (f"analysis: <b>{_html.escape(self.analysis)}</b> &nbsp;&middot;&nbsp; "
                f"model: <b>{_html.escape(self.model_type)}</b> &nbsp;&middot;&nbsp; "
                f"outputs: <b>{len(out_keys)}</b> &nbsp;&middot;&nbsp; "
                f"samples: <b>{n_samples}</b> &nbsp;&middot;&nbsp; "
                f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        intro = (
            "<p><strong>SHAP (SHapley Additive exPlanations)</strong> decomposes each "
            "prediction into the contribution of every input feature. A positive SHAP "
            "value pushes the output higher; a negative one pushes it lower. Each "
            "section below explains one model output.</p>"
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
</style></head>
<body>
<header><h1>{_html.escape(title)}</h1><div class="meta">{meta}</div></header>
<div class="wrap">
<div class="intro">{intro}</div>
<nav><h3>Outputs</h3><ul>{''.join(toc)}</ul></nav>
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
