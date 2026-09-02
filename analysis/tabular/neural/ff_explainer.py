"""
Feedforward neural-network explainer.

Explains a standard fully-connected (MLP) model trained with **either**
PyTorch or TensorFlow/Keras, on tabular data. The caller chooses the SHAP
backend via ``config["explainer_type"]``:

- ``"kernel"``  — model-agnostic :class:`shap.KernelExplainer`. Works for any
  framework because it only needs a numpy predict function. Safe default.
- ``"deep"``    — framework-native :class:`shap.DeepExplainer`.
- ``"gradient"``— :class:`shap.GradientExplainer`.

Everything is in-memory friendly: pass a fitted ``model`` and a ``data``
DataFrame (plus an optional ``background_data``) and get an
:class:`ai_explainability.ExplanationResult` back with 2D SHAP arrays shaped
``(n_samples, n_features)`` — exactly like the tree-based explainer, so
``to_dataframe()`` / ``save_excel()`` behave identically.
"""

import numpy as np
import shap

from ..tree_based.base import ExplainerBase


class FeedForwardExplainer(ExplainerBase):
    """SHAP explainer for feedforward / MLP networks (PyTorch or TensorFlow)."""

    #: SHAP backends this explainer knows how to build.
    SUPPORTED_EXPLAINERS = ("kernel", "deep", "gradient")

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #
    def __init__(self, config, *, model=None, data=None, background_data=None):
        super().__init__(config, model=model, data=data)
        self.background_data = background_data
        self.framework = None

    def load_model(self):
        from ai_explainability.io import detect_framework, to_fitted_model

        if self.model is None:
            # Path-based load: torch state/full-module via .pt/.pth, keras via
            # .keras/.h5, else pickle. We defer to to_fitted_model for the
            # common cases and add a keras branch here.
            model_path = self.get_path("model_path")
            self.model = self._load_model_from_path(model_path, to_fitted_model)

        # Resolve the framework once — explicit config wins, else auto-detect.
        self.framework = (
            self.config.get("package")
            or self.config.get("framework")
            or detect_framework(self.model)
        ).lower()
        if self.framework in {"torch",}:
            self.framework = "pytorch"
        if self.framework in {"tf", "keras"}:
            self.framework = "tensorflow"

        if self.framework not in {"pytorch", "tensorflow"}:
            raise ValueError(
                "Could not determine the model framework. Set "
                "config['package'] to 'pytorch' or 'tensorflow'."
            )
        print(f"Feedforward model ready ({self.framework}).")

    @staticmethod
    def _load_model_from_path(model_path, to_fitted_model):
        if model_path is None:
            raise ValueError(
                "FeedForwardExplainer needs either a 'model' kwarg or "
                "config['model_path']."
            )
        lowered = str(model_path).lower()
        if lowered.endswith((".keras", ".h5", ".hdf5")):
            # Keras native format.
            from tensorflow import keras

            return keras.models.load_model(model_path)
        # torch (.pt/.pth) and pickle are handled by to_fitted_model.
        return to_fitted_model(model_path)

    # ------------------------------------------------------------------ #
    # Explanation                                                        #
    # ------------------------------------------------------------------ #
    def explain(self):
        from ai_explainability.io import make_predict_fn, to_pandas

        explainer_type = self.config.get("explainer_type", "kernel")
        if explainer_type not in self.SUPPORTED_EXPLAINERS:
            raise ValueError(
                f"Unsupported explainer_type {explainer_type!r} for "
                f"FeedForwardExplainer. Choose one of {self.SUPPORTED_EXPLAINERS}."
            )

        # 1. Coerce the data to explain.
        if self.raw_data is None:
            self.raw_data = to_pandas(
                self.get_path("dataset_path"), feature_names=self.feature_names
            )
        else:
            self.raw_data = to_pandas(self.raw_data, feature_names=self.feature_names)
        if self.feature_names:
            self.raw_data = self.raw_data[self.feature_names]
        if self.config.get("dataset_scope") == "subset":
            self.raw_data = self.raw_data.iloc[: self.config.get("subset_end", 100)]

        X = self.raw_data.to_numpy(dtype="float32")
        self.raw_data_values = X

        # 2. Build a background / baseline distribution appropriate to the backend.
        background = self._resolve_background(X, explainer_type)

        # 3. Dispatch on explainer type.
        if explainer_type == "kernel":
            predict_fn = make_predict_fn(self.model, self.framework)
            explainer = shap.KernelExplainer(predict_fn, background)
            nsamples = self.config.get("kernel_nsamples", "auto")
            shap_raw = explainer.shap_values(X, nsamples=nsamples)
        elif explainer_type == "gradient":
            explainer = shap.GradientExplainer(
                self.model, self._as_framework_tensor(background)
            )
            shap_raw = explainer.shap_values(self._as_framework_tensor(X))
        else:  # "deep"
            explainer = shap.DeepExplainer(
                self.model, self._as_framework_tensor(background)
            )
            shap_raw = explainer.shap_values(self._as_framework_tensor(X))

        # 4. Normalise to a 2D array (n_samples, n_features).
        self.all_shap_values = {0: self._normalise_shap(shap_raw)}

        # 5. Predictions for the result object.
        preds = make_predict_fn(self.model, self.framework)(X)
        self.all_predictions = {0: np.asarray(preds).reshape(len(X), -1)[:, 0]}

        # Compatibility for the notebook generator.
        self.shap_values = self.all_shap_values[0]

    # ------------------------------------------------------------------ #
    # Helpers                                                             #
    # ------------------------------------------------------------------ #
    def _resolve_background(self, X, explainer_type):
        """Return a background baseline appropriate to the chosen backend.

        Prefers an explicit ``background_data`` (kwarg or path). Otherwise it
        summarises the data itself:

        - For ``kernel`` we may return a :func:`shap.kmeans` summary (a
          ``DenseData`` object) — KernelExplainer understands it and it keeps
          the coalition sampling tractable.
        - For ``gradient`` / ``deep`` we must return a plain **ndarray** of
          real rows, because those backends convert the background into a
          framework tensor (a ``DenseData`` object would break that).
        """
        from ai_explainability.io import to_numpy_2d

        if self.background_data is not None:
            return to_numpy_2d(self.background_data, feature_names=self.feature_names)

        bg_path = self.get_path("background_data_path")
        if bg_path:
            return to_numpy_2d(bg_path, feature_names=self.feature_names)

        # No explicit background — subsample the data itself.
        k = int(self.config.get("background_size", min(100, len(X))))
        if len(X) <= k:
            return X

        if explainer_type == "kernel":
            try:
                return shap.kmeans(X, k)  # DenseData summary — kernel only
            except Exception:
                pass  # fall through to a plain row sample
        idx = np.random.RandomState(0).choice(len(X), k, replace=False)
        return X[idx]

    def _as_framework_tensor(self, arr):
        """Convert a numpy array to the tensor type the deep/gradient backend wants."""
        if self.framework == "pytorch":
            import torch

            if isinstance(arr, np.ndarray):
                return torch.as_tensor(arr, dtype=torch.float32)
            return arr  # already a tensor / kmeans summary handled by caller
        # TensorFlow DeepExplainer accepts numpy arrays directly.
        return np.asarray(arr, dtype="float32")

    @staticmethod
    def _normalise_shap(shap_raw):
        """Collapse SHAP output to a single 2D (n_samples, n_features) array.

        Handles: bare 2D ndarray (regression), list of arrays (classification —
        keep class 1 / positive), 3D ``(n, feat, n_outputs)`` (keep output 1 for
        binary, else output 0), and a trailing singleton axis.
        """
        if isinstance(shap_raw, list):
            arr = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
        else:
            arr = shap_raw
        arr = np.asarray(arr)
        if arr.ndim == 3:
            # (n, feat, n_outputs) — pick positive class for binary, else first.
            arr = arr[:, :, 1] if arr.shape[-1] > 1 else arr[:, :, 0]
        return arr

    # ------------------------------------------------------------------ #
    # In-memory result view                                              #
    # ------------------------------------------------------------------ #
    def to_result(self):
        from ai_explainability.result import ExplanationResult

        return ExplanationResult(
            shap_values=dict(self.all_shap_values),
            predictions=dict(getattr(self, "all_predictions", {})),
            raw_data=self.raw_data,
            raw_data_values=self.raw_data_values,
            feature_names=list(self.raw_data.columns),
            analysis="tabular",
            model_type=self.config.get("model_type", "feedforward"),
            extras={"framework": self.framework},
        )

    # ------------------------------------------------------------------ #
    # Disk outputs (mirror the tree-based explainer)                     #
    # ------------------------------------------------------------------ #
    def save_results_to_excel(self):
        import os
        from datetime import datetime

        self.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shap_audit_{self.config.get('model_type', 'feedforward')}_{timestamp}.xlsx"
        output_path = os.path.join(self.output_dir, filename)

        shap_arr = self.all_shap_values[0]
        preds = self.all_predictions.get(0)
        df_features = self.raw_data.reset_index(drop=True)
        sheet = df_features.copy()
        if preds is not None:
            sheet["Model_Prediction"] = np.asarray(preds).ravel()[: len(sheet)]
        for j, name in enumerate(self.raw_data.columns):
            if j < shap_arr.shape[1]:
                sheet[f"SHAP_{name}"] = shap_arr[:, j]
        sheet.to_excel(output_path, index=False)
        print(f"Feedforward SHAP audit saved: {output_path}")

    def plot_results(self):
        # Local import — nbformat / nbconvert only needed for the notebook report.
        import os
        from datetime import datetime

        from output.utils.report_gen import generate_notebook

        self.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nb_name = f"report_{self.config.get('model_type', 'feedforward')}_{timestamp}.ipynb"
        nb_path = os.path.join(self.output_dir, nb_name)
        # Tag the config so the notebook narrative names the right explainer.
        self.config.setdefault("explainer_type", self.config.get("explainer_type", "kernel"))
        generate_notebook(
            explainer_inst=self,
            all_shap_values=self.all_shap_values,
            raw_data=self.raw_data_values,
            output_path=nb_path,
        )
