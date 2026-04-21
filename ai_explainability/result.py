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
    """

    shap_values: dict[int, np.ndarray] = field(default_factory=dict)
    predictions: dict[int, np.ndarray] = field(default_factory=dict)
    raw_data: pd.DataFrame | None = None
    raw_data_values: np.ndarray | None = None
    feature_names: list[str] = field(default_factory=list)
    analysis: str = ""
    model_type: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

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
        base = pd.DataFrame(arr, columns=features)

        for idx, preds in self.predictions.items():
            base[f"Model_Prediction_{idx}"] = np.asarray(preds).ravel()[: len(base)]

        for idx, shap_arr in self.shap_values.items():
            sa = np.asarray(shap_arr)
            if sa.ndim == 2 and sa.shape[1] == len(features):
                for j, name in enumerate(features):
                    base[f"SHAP_{name}_t{idx}"] = sa[:, j]
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
                    features = list(self.raw_data.columns) if self.raw_data is not None else self.feature_names
                    df_features = (
                        self.raw_data.reset_index(drop=True)
                        if self.raw_data is not None
                        else pd.DataFrame(self.raw_data_values, columns=features)
                    )
                    preds_series = self.predictions.get(idx)
                    sheet_df = df_features.copy()
                    if preds_series is not None:
                        sheet_df["Model_Prediction"] = np.asarray(preds_series).ravel()[: len(sheet_df)]
                    for j, name in enumerate(features):
                        if sa.ndim == 2 and j < sa.shape[1]:
                            sheet_df[f"SHAP_{name}"] = sa[:, j]

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

    # ------------------------------------------------------------------ #
    # Convenience                                                        #
    # ------------------------------------------------------------------ #
    def timestamped_filename(self, stem: str, ext: str) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{stem}_{self.model_type}_{ts}.{ext.lstrip('.')}"
