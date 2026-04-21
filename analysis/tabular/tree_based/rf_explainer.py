import os
import numpy as np
import pandas as pd
import shap
from datetime import datetime

from .base import ExplainerBase


class RFExplainer(ExplainerBase):
    """Tree-based explainer (RandomForest, XGBoost, DecisionTree, …).

    Supports two usage patterns:

    1. **Legacy / CLI** — config-driven, with ``model_path`` and
       ``dataset_path`` pointing to files on disk. Behaviour unchanged.
    2. **In-memory** — pass a fitted ``model`` and a ``data`` DataFrame (or
       ndarray / pyspark DataFrame) via the base-class kwargs. Neither
       ``pickle.load`` nor ``pd.read_csv`` is called. No ``os.makedirs``
       happens unless the caller asks for disk output.
    """

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #
    def load_model(self):
        self.config["explainer_type"] = "tree"

        if self.model is None:
            # Legacy path-based load. Imported lazily so the in-memory path
            # never touches the filesystem.
            from ai_explainability.io import to_fitted_model

            self.model = to_fitted_model(self.get_path("model_path"))

        self.is_multi_output = (
            hasattr(self.model, "estimators_") and len(self.model.estimators_) > 1
        )
        self.is_classification = hasattr(self.model, "predict_proba")
        print(f"Model loaded: {type(self.model).__name__}")

    # ------------------------------------------------------------------ #
    # Core explanation loop                                              #
    # ------------------------------------------------------------------ #
    def explain(self):
        # 1. Load / coerce data
        if self.raw_data is None:
            df = pd.read_csv(self.get_path("dataset_path"))
        else:
            # In-memory input — could be a DataFrame, ndarray, or pyspark DF.
            from ai_explainability.io import to_pandas

            df = to_pandas(self.raw_data, feature_names=self.feature_names)

        # If the caller gave us feature_names, restrict to those columns; this
        # mirrors the legacy behaviour but now works on in-memory data too.
        self.raw_data = df[self.feature_names] if self.feature_names else df
        if self.config.get("dataset_scope") == "subset":
            self.raw_data = self.raw_data.iloc[: self.config.get("subset_end", 100)]

        self.raw_data_values = self.raw_data.values

        # 2. Prepare Target List
        targets = self.config.get("target_index", 0)
        if isinstance(targets, int):
            targets = [targets]

        self.all_shap_values = {}
        self.all_predictions = {}

        # 3. Enumerate and Explain
        for idx in targets:
            print(f"Explaining target index: {idx}...")

            # Extract sub-model for MultiOutput wrappers
            # Works for both MultiOutputClassifier and MultiOutputRegressor
            if hasattr(self.model, "estimators_"):
                model_to_explain = self.model.estimators_[idx]
            else:
                model_to_explain = self.model

            explainer = shap.TreeExplainer(model_to_explain)
            shap_output = explainer.shap_values(self.raw_data)

            # Handle dimensionality
            # For binary classification, shap_output is a list of 2 arrays
            # [class_0, class_1]; we always want class_1 (positive class).
            if isinstance(shap_output, list):
                self.all_shap_values[idx] = (
                    shap_output[1] if len(shap_output) > 1 else shap_output[0]
                )
            elif shap_output.ndim == 3:
                self.all_shap_values[idx] = shap_output[:, :, 1]
            else:
                self.all_shap_values[idx] = shap_output

            # Capture predictions for the result object — same shape semantics
            # as the existing Excel writer.
            preds = self.model.predict(self.raw_data)
            if preds.ndim > 1:
                preds = preds[:, idx]
            self.all_predictions[idx] = preds

        # Compatibility for the notebook
        self.shap_values = self.all_shap_values[targets[0]]

    # ------------------------------------------------------------------ #
    # In-memory result view                                              #
    # ------------------------------------------------------------------ #
    def to_result(self):
        """Return the run as an :class:`ai_explainability.ExplanationResult`.

        Imported lazily so this module doesn't depend on the shim package
        (avoids a circular import in development setups).
        """
        from ai_explainability.result import ExplanationResult

        return ExplanationResult(
            shap_values=dict(self.all_shap_values),
            predictions=dict(getattr(self, "all_predictions", {})),
            raw_data=self.raw_data,
            raw_data_values=self.raw_data_values,
            feature_names=list(self.raw_data.columns),
            analysis="tabular",
            model_type=self.config.get("model_type", ""),
        )

    # ------------------------------------------------------------------ #
    # Optional disk serialisation (unchanged semantics)                  #
    # ------------------------------------------------------------------ #
    def save_results_to_excel(self):
        """Saves a multi-sheet Excel file, one sheet per target."""
        self.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"shap_audit_{self.config['model_type']}_{timestamp}.xlsx"
        output_path = os.path.join(self.output_dir, filename)

        with pd.ExcelWriter(output_path) as writer:
            for idx, shap_arr in self.all_shap_values.items():

                # --- Resolve target name correctly for all config shapes ---
                output_labels = self.config.get("output_labels", {})
                raw = output_labels.get(str(idx), f"target_{idx}")

                if isinstance(raw, dict):
                    # Shape B — multioutput classification: {"0": "OFF", "1": "ON"}
                    # The actual name is stored in e.g. "0_name": "Generator 1"
                    target_name = output_labels.get(f"{idx}_name", f"target_{idx}")
                else:
                    # Shape A — regression or binary: value is the name directly
                    target_name = str(raw) if raw else f"target_{idx}"

                # Safety: never allow empty or blank sheet name (Excel would crash)
                if not target_name or not target_name.strip():
                    target_name = f"target_{idx}"

                # Get Preds (reuse cached value if available)
                preds = self.all_predictions.get(idx) if hasattr(self, "all_predictions") else None
                if preds is None:
                    preds = self.model.predict(self.raw_data)
                    if preds.ndim > 1:
                        preds = preds[:, idx]

                df_features = self.raw_data.reset_index(drop=True)
                df_preds = pd.DataFrame({"Model_Prediction": preds})
                df_shap = pd.DataFrame(
                    shap_arr, columns=[f"SHAP_{c}" for c in self.raw_data.columns]
                )

                sheet_df = pd.concat([df_features, df_preds, df_shap], axis=1)
                sheet_df.to_excel(writer, sheet_name=target_name[:31], index=False)

        print(f"Multi-target Excel audit saved: {output_path}")

    def plot_results(self):
        # Local import — nbformat / nbconvert chain is only needed when the
        # caller actually asks for a notebook report.
        from output.utils.report_gen import generate_notebook

        self.ensure_output_dir()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nb_name = f"report_{self.config['model_type']}_{timestamp}.ipynb"
        nb_path = os.path.join(self.output_dir, nb_name)

        generate_notebook(
            explainer_inst=self,
            all_shap_values=self.all_shap_values,
            raw_data=self.raw_data_values,
            output_path=nb_path,
        )
