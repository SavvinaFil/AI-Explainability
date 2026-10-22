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

        # A plain callable (e.g. a wrapper around an RL policy's heads) is a
        # valid, framework-agnostic model — but it only supports KernelSHAP.
        if self.framework not in {"pytorch", "tensorflow"}:
            if callable(self.model):
                self.framework = "callable"
            else:
                raise ValueError(
                    "Could not determine the model framework. Set "
                    "config['package'] to 'pytorch' or 'tensorflow', or pass a "
                    "plain callable obs(ndarray) -> values(ndarray)."
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
        if self.framework == "callable" and explainer_type != "kernel":
            raise ValueError(
                f"explainer_type={explainer_type!r} needs a torch/keras model; a "
                "plain callable only supports 'kernel'. Pass the underlying "
                "nn.Module/keras model for deep/gradient, or use explainer_type='kernel'."
            )
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

        # 3. Predict once to learn the output width (= number of action
        #    components for an RL policy) and capture per-output predictions.
        predict_fn = make_predict_fn(self.model, self.framework)
        preds_all = np.asarray(predict_fn(X))
        if preds_all.ndim == 1:
            preds_all = preds_all.reshape(-1, 1)
        n_outputs = preds_all.shape[1]
        self.n_outputs = n_outputs

        # Per-sample targets (RL): explain only the selected output(s) per
        # sample instead of every output column.
        if self.config.get("targets") is not None:
            self._explain_selected_targets(
                X, background, explainer_type, preds_all, predict_fn
            )
            return

        # Which outputs (actions / setpoints) to explain. Default: all of them.
        targets = self.config.get("target_index")
        if targets is None:
            targets = list(range(n_outputs))
        elif isinstance(targets, (int, np.integer)):
            targets = [int(targets)]
        else:
            targets = [int(t) for t in targets]

        # 4. Compute SHAP once, then split into one 2D array per output. This is
        #    framework- and action-type-agnostic: continuous setpoints and
        #    discrete logits are both just output columns.
        shap_raw = self._run_shap(self.model, predict_fn, background, X, explainer_type)
        per_output = self._split_outputs(shap_raw, n_outputs)

        # 5. Keep only the requested outputs.
        self.all_shap_values = {}
        self.all_predictions = {}
        for idx in targets:
            if idx not in per_output:
                raise ValueError(
                    f"target_index {idx} is out of range for a model with "
                    f"{n_outputs} output(s)."
                )
            self.all_shap_values[idx] = per_output[idx]
            self.all_predictions[idx] = preds_all[:, idx]

        # Compatibility for the notebook generator.
        self.shap_values = self.all_shap_values[targets[0]]

    def _run_shap(self, model, predict_fn, background, X, explainer_type):
        """Build the requested SHAP backend and return its raw output for ``X``."""
        if explainer_type == "gradient" and self.framework == "pytorch":
            n_outputs = int(np.asarray(predict_fn(X[:1])).reshape(1, -1).shape[1])
            cols = np.broadcast_to(np.arange(n_outputs), (len(X), n_outputs))
            return list(self._expected_gradients_torch(model, background, X, cols).transpose(2, 0, 1))
        if explainer_type == "kernel":
            explainer = shap.KernelExplainer(predict_fn, background)
            nsamples = self.config.get("kernel_nsamples", "auto")
            return explainer.shap_values(X, nsamples=nsamples)
        if explainer_type == "gradient":
            explainer = shap.GradientExplainer(
                model, self._as_framework_tensor(background)
            )
        else:  # "deep"
            explainer = shap.DeepExplainer(
                model, self._as_framework_tensor(background)
            )
        return explainer.shap_values(self._as_framework_tensor(X))

    # ------------------------------------------------------------------ #
    # Per-sample targets (explain only the action taken)                  #
    # ------------------------------------------------------------------ #
    def _resolve_target_heads(self, n_outputs):
        """Normalise ``config['targets']`` to ``{head_name: [output indices]}``."""
        targets = self.config["targets"]
        if isinstance(targets, str):
            if targets != "argmax":
                raise ValueError(
                    f"targets must be 'argmax' or a {{head: [output indices]}} "
                    f"mapping; got {targets!r}."
                )
            return {"argmax": list(range(n_outputs))}
        if not isinstance(targets, dict) or not targets:
            raise ValueError(
                "targets must be 'argmax' or a non-empty {head: [output indices]} mapping."
            )
        heads = {}
        for name, cols in targets.items():
            cols = [int(c) for c in (cols if hasattr(cols, "__iter__") else [cols])]
            if not cols:
                raise ValueError(f"Head {name!r} has no output indices.")
            bad = [c for c in cols if not 0 <= c < n_outputs]
            if bad:
                raise ValueError(
                    f"Head {name!r} references outputs {bad}, but the model has "
                    f"{n_outputs} output(s)."
                )
            heads[str(name)] = cols
        return heads

    def _resolve_families(self, head_names):
        """Return ``{output_name: [head names]}`` — one output per head unless summed.

        With ``targets_reduce='sum'`` the heads listed in each
        ``target_families`` entry are collapsed into one output; heads not
        listed in any family stay as their own output.
        """
        reduce = self.config.get("targets_reduce")
        if reduce is None:
            return {h: [h] for h in head_names}
        if reduce != "sum":
            raise ValueError(f"targets_reduce must be None or 'sum'; got {reduce!r}.")
        families = self.config.get("target_families")
        if not families:
            raise ValueError(
                "targets_reduce='sum' needs target_families={family: [head names]}."
            )
        out, seen = {}, set()
        for fam, members in families.items():
            members = [str(m) for m in members]
            unknown = [m for m in members if m not in head_names]
            if unknown:
                raise ValueError(f"Family {fam!r} references unknown heads {unknown}.")
            dup = [m for m in members if m in seen]
            if dup:
                raise ValueError(f"Heads {dup} appear in more than one family.")
            seen.update(members)
            out[str(fam)] = members
        for h in head_names:
            if h not in seen:
                if h in out:
                    raise ValueError(
                        f"Family name {h!r} collides with a head that is not in any family."
                    )
                out[h] = [h]
        return out

    def _selection_model(self, S):
        """Wrap the model so it returns ``model(x) @ S`` (the selected outputs).

        ``S`` is a fixed ``(n_outputs, n_selected)`` 0/1 matrix, so the wrapper
        is linear in the logits and every SHAP backend handles it exactly.
        """
        if self.framework == "pytorch":
            import torch

            class _SelectOutputs(torch.nn.Module):
                def __init__(self, base, sel):
                    super().__init__()
                    self.base = base
                    self.register_buffer("sel", torch.as_tensor(sel, dtype=torch.float32))

                def forward(self, x):
                    return self.base(x) @ self.sel

            return _SelectOutputs(self.model, S).eval()
        if self.framework == "tensorflow":
            import tensorflow as tf

            # A frozen bias-free Dense layer instead of tf.linalg.matmul: raw TF
            # ops cannot be applied to KerasTensors under Keras 3 (TF >= 2.16).
            proj = tf.keras.layers.Dense(S.shape[1], use_bias=False, trainable=False)
            out = proj(self.model.output)
            proj.set_weights([np.asarray(S, dtype="float32")])
            return tf.keras.Model(inputs=self.model.inputs, outputs=out)
        return None  # callable: only the kernel path, via the predict fn.

    def _explain_selected_targets(self, X, background, explainer_type, preds_all, base_predict):
        """Explain, per sample, only the output each head actually selected.

        The argmax of every head is computed once on ``X`` (no gradients).
        Samples are then grouped by their chosen-index tuple and SHAP runs once
        per group on a wrapped model that returns only those logits (or their
        per-family sums). The selection is therefore fixed per explained
        sample, never re-derived on SHAP's perturbed or interpolated inputs.
        """
        n_outputs = preds_all.shape[1]
        heads = self._resolve_target_heads(n_outputs)
        head_names = list(heads)
        families = self._resolve_families(head_names)
        fam_names = list(families)
        head_pos = {h: i for i, h in enumerate(head_names)}

        # (n, n_heads) original output index chosen by each head.
        chosen = np.column_stack(
            [np.asarray(heads[h])[preds_all[:, heads[h]].argmax(axis=1)] for h in head_names]
        )

        n, n_feat = X.shape
        shap_out = {f: np.zeros((n, n_feat), dtype=float) for f in fam_names}
        pred_out = {f: np.zeros(n, dtype=float) for f in fam_names}

        self.chosen_index = {h: chosen[:, head_pos[h]] for h in head_names}
        self.target_heads = heads
        self.target_families = families

        if explainer_type == "gradient" and self.framework == "pytorch":
            # The choice travels with each sample, so every sample is explained in one batched run instead of one
            # SHAP run per distinct action combination.
            fam_matrix = np.zeros((len(head_names), len(fam_names)), dtype="float32")
            for j, fam in enumerate(fam_names):
                for h in families[fam]:
                    fam_matrix[head_pos[h], j] = 1.0
            phi = self._expected_gradients_torch(self.model, background, X, chosen, fam_matrix)
            sel_preds = np.take_along_axis(preds_all, chosen, axis=1) @ fam_matrix
            for j, fam in enumerate(fam_names):
                shap_out[fam] = phi[:, :, j]
                pred_out[fam] = sel_preds[:, j]
            self.all_shap_values = shap_out
            self.all_predictions = pred_out
            self.n_groups = 1
            print(f"Explained {len(fam_names)} selected output(s) instead of {n_outputs} "
                  f"(one batched expected-gradients run over {n} sample(s)).")
            self.shap_values = self.all_shap_values[fam_names[0]]
            return

        groups, inverse = np.unique(chosen, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        for g, combo in enumerate(groups):
            rows = np.flatnonzero(inverse == g)
            S = np.zeros((n_outputs, len(fam_names)), dtype="float32")
            for j, fam in enumerate(fam_names):
                for h in families[fam]:
                    # += so two heads in one family picking the same column count twice.
                    S[combo[head_pos[h]], j] += 1.0

            def predict_fn(x, _S=S):
                p = np.asarray(base_predict(x))
                return (p.reshape(len(p), -1) @ _S).astype("float32")

            model = None if explainer_type == "kernel" else self._selection_model(S)
            shap_raw = self._run_shap(model, predict_fn, background, X[rows], explainer_type)
            per_output = self._split_outputs(shap_raw, len(fam_names))
            sel_preds = preds_all[rows] @ S
            for j, fam in enumerate(fam_names):
                shap_out[fam][rows] = np.asarray(per_output[j]).reshape(len(rows), n_feat)
                pred_out[fam][rows] = sel_preds[:, j]

        self.all_shap_values = shap_out
        self.all_predictions = pred_out
        self.n_groups = len(groups)
        print(
            f"Explained {len(fam_names)} selected output(s) instead of {n_outputs} "
            f"({len(groups)} distinct action combination(s))."
        )
        self.shap_values = self.all_shap_values[fam_names[0]]

    def _expected_gradients_torch(self, model, background, X, cols, fam_matrix=None):
        """Batched expected gradients (the estimator behind ``shap.GradientExplainer``) for PyTorch.

        For every sample ``x`` it draws ``gradient_nsamples`` background rows ``b`` and points ``t ~ U(0, 1)`` and
        averages ``grad f(b + t (x - b)) * (x - b)``. Unlike ``shap.GradientExplainer`` it

        - runs one forward pass per batch for **all** explained outputs, and gets all their gradients from one
          batched backward pass (``is_grads_batched``), instead of a forward and backward per output;
        - batches the interpolation points of **all** samples together in large chunks, instead of one sample at a
          time in batches of 50.

        :param model: Module mapping ``(B, n_features)`` to ``(B, n_outputs)``.
        :param background: ``(n_bg, n_features)`` baseline rows.
        :param X: ``(n, n_features)`` samples to explain.
        :param cols: ``(n, k)`` output column explained per sample and slot; fixed per sample, so every interpolation
            point of a sample explains the same outputs.
        :param fam_matrix: Optional ``(k, n_out)`` 0/1 matrix summing the k slots into ``n_out`` outputs.
        :return: ``(n, n_features, n_out)`` SHAP values (``n_out = k`` without ``fam_matrix``).
        """
        import torch

        nsamples = int(self.config.get("gradient_nsamples", 200))
        rng = np.random.default_rng(int(self.config.get("seed", 0)))
        bg = torch.as_tensor(np.asarray(background, dtype="float32"))
        X_t = torch.as_tensor(np.asarray(X, dtype="float32"))
        cols_t = torch.as_tensor(np.ascontiguousarray(cols), dtype=torch.long)
        fam = None if fam_matrix is None else torch.as_tensor(fam_matrix, dtype=torch.float32)
        n, n_feat = X_t.shape
        n_out = cols_t.shape[1] if fam is None else fam.shape[1]

        # Rows per chunk: keep the (n_out, rows, n_features) gradient tensor around 64M floats
        default_rows = max(nsamples, int(64e6 // max(n_out * n_feat, 1)) // nsamples * nsamples)
        rows_per_chunk = int(self.config.get("gradient_batch_size", default_rows))
        per_chunk = max(1, rows_per_chunk // nsamples)  # whole samples per chunk

        phi = np.zeros((n, n_feat, n_out), dtype="float32")
        batched = True
        model.eval()
        for start in range(0, n, per_chunk):
            idx = np.arange(start, min(start + per_chunk, n))
            m = len(idx)
            b = bg[rng.integers(0, len(bg), size=m * nsamples)]
            t = torch.as_tensor(rng.random(m * nsamples, dtype="float32")).unsqueeze(1)
            x = X_t[idx].repeat_interleave(nsamples, dim=0)
            z = (b + t * (x - b)).requires_grad_(True)
            out = model(z).gather(1, cols_t[idx].repeat_interleave(nsamples, dim=0))
            if fam is not None:
                out = out @ fam
            grads = None
            if batched:
                try:
                    eye = torch.eye(n_out).unsqueeze(1).expand(n_out, len(z), n_out)
                    (grads,) = torch.autograd.grad(out, z, grad_outputs=eye, is_grads_batched=True)
                except (RuntimeError, NotImplementedError):
                    batched = False  # model not vmap-compatible: one backward per output, still one forward
            if grads is None:
                grads = torch.stack([
                    torch.autograd.grad(out[:, j].sum(), z, retain_graph=j + 1 < n_out)[0] for j in range(n_out)
                ])
            contrib = grads * (x - b).unsqueeze(0)  # (n_out, m * nsamples, n_features)
            phi[idx] = contrib.reshape(n_out, m, nsamples, n_feat).mean(dim=2).permute(1, 2, 0).detach().numpy()
        return phi

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
                import warnings

                with warnings.catch_warnings():
                    # sklearn's KMeans emits a benign MKL memory-leak UserWarning
                    # on Windows when there are fewer chunks than threads. It
                    # doesn't affect the summary, so silence just that message
                    # (setting OMP_NUM_THREADS at runtime wouldn't help — MKL
                    # reads it at init).
                    warnings.filterwarnings(
                        "ignore",
                        message=".*KMeans is known to have a memory leak.*",
                    )
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
    def _split_outputs(shap_raw, n_outputs):
        """Split a SHAP result into ``{output_idx: (n_samples, n_features)}``.

        Handles the shapes SHAP returns for multi-output models:

        - list of arrays (one per output) — kernel/deep/gradient on multi-output;
        - 3D ndarray ``(n, feat, n_outputs)`` — some SHAP versions;
        - 2D ndarray — a single-output model.

        A trailing singleton output axis is squeezed.
        """
        if isinstance(shap_raw, list):
            out = {i: np.asarray(a) for i, a in enumerate(shap_raw)}
            # Squeeze any trailing singleton feature axis per entry.
            return {i: (a[..., 0] if a.ndim == 3 and a.shape[-1] == 1 else a)
                    for i, a in out.items()}
        arr = np.asarray(shap_raw)
        if arr.ndim == 3:
            if arr.shape[-1] == 1:
                return {0: arr[:, :, 0]}
            return {i: arr[:, :, i] for i in range(arr.shape[2])}
        # 2D — single output.
        return {0: arr}

    @staticmethod
    def _normalise_shap(shap_raw):
        """Collapse SHAP output to a single 2D array (legacy convenience).

        Retained for callers/tests that want a single-output view; the explainer
        itself now uses :meth:`_split_outputs` to keep every output.
        """
        if isinstance(shap_raw, list):
            arr = shap_raw[1] if len(shap_raw) > 1 else shap_raw[0]
        else:
            arr = shap_raw
        arr = np.asarray(arr)
        if arr.ndim == 3:
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
            chosen_index=getattr(self, "chosen_index", None),
            target_heads=getattr(self, "target_heads", None),
            target_families=getattr(self, "target_families", None),
        )

    # ------------------------------------------------------------------ #
    # Disk outputs (mirror the tree-based explainer)                     #
    # ------------------------------------------------------------------ #
    def save_results_to_excel(self):
        import os
        from datetime import datetime

        import pandas as pd

        self.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shap_audit_{self.config.get('model_type', 'feedforward')}_{timestamp}.xlsx"
        output_path = os.path.join(self.output_dir, filename)

        from output.utils.report_content import select_feature_indices

        df_features = self.raw_data.reset_index(drop=True)
        cols = list(self.raw_data.columns)
        output_labels = self.config.get("output_labels", {})

        with pd.ExcelWriter(output_path) as writer:
            for idx, shap_arr in self.all_shap_values.items():
                # Name the sheet after the action/output if labels were provided.
                label = None
                if isinstance(output_labels, dict):
                    label = output_labels.get(str(idx)) or output_labels.get(idx)
                elif isinstance(output_labels, (list, tuple)) and isinstance(idx, (int, np.integer)) and idx < len(output_labels):
                    label = output_labels[idx]
                sheet_name = str(label) if label else f"output_{idx}"

                # Trim to selected / top features (per output) when requested.
                sel = select_feature_indices(
                    shap_arr, cols,
                    top_features=self.config.get("top_features"),
                    select_features=self.config.get("select_features"),
                )
                sheet = df_features.iloc[:, sel].copy()
                preds = self.all_predictions.get(idx)
                if preds is not None:
                    sheet["Model_Prediction"] = np.asarray(preds).ravel()[: len(sheet)]
                for j in sel:
                    if j < shap_arr.shape[1]:
                        sheet[f"SHAP_{cols[j]}"] = shap_arr[:, j]
                sheet.to_excel(writer, sheet_name=sheet_name[:31], index=False)
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
