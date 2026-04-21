import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

from .base import TimeseriesExplainerBase


class LSTMForecaster(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2
        )
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class LSTMExplainer(TimeseriesExplainerBase):
    """LSTM explainer.

    Accepts in-memory inputs via the base class's kwargs:

    - ``model``: a fitted ``torch.nn.Module`` (either :class:`LSTMForecaster`
      or a compatible custom class). When omitted, the architecture is
      reconstructed from ``config["input_dim"]`` / ``config["hidden_size"]``
      and its weights loaded from ``config["model_path"]``.
    - ``background_data``: a ``torch.Tensor`` / ndarray / path used as the
      SHAP background distribution.
    - ``test_data``: a ``torch.Tensor`` / ndarray / path containing the
      samples to explain.
    """

    def __init__(self, config, *, model=None, background_data=None, test_data=None):
        super().__init__(
            config,
            model=model,
            background_data=background_data,
            test_data=test_data,
        )
        # Legacy-compat convenience: expose generate_notebook as a method so
        # third-party code calling ``explainer.generate_notebook(...)`` still
        # works. Bind lazily so the nbformat/nbconvert chain is only imported
        # when the method is actually accessed.
        self._bound_generate_notebook = None

    # ------------------------------------------------------------------ #
    # Setup                                                              #
    # ------------------------------------------------------------------ #
    def load_model(self):
        """Resolve ``self.model`` into a ready-to-evaluate torch Module.

        If the caller already provided a ``model`` kwarg, we trust it and
        just put it in eval mode. Otherwise we reconstruct the architecture
        from the config and load the state_dict from disk — the original
        behaviour.
        """
        self.input_dim = self.config.get("input_dim", 12)
        self.hidden_dim = self.config.get("hidden_size", 16)
        self.model_type = self.config.get("model_type", "lstm")
        self.output_dim = 1
        self.output_labels = self.config.get("output_labels", 0)

        if self.model is None:
            # Reconstruct architecture and load weights from ``model_path``.
            self.model = LSTMForecaster(
                input_dim=self.input_dim, hidden_dim=self.hidden_dim
            )

            model_full_path = self.get_path("model_path")
            if model_full_path is None:
                raise ValueError(
                    "LSTMExplainer needs either a 'model' kwarg or "
                    "config['model_path']; neither was provided."
                )

            state_dict = torch.load(
                model_full_path, map_location=torch.device("cpu")
            )
            self.model.load_state_dict(state_dict)

        self.model.eval()
        torch.set_grad_enabled(True)
        print("Model ready for explanation.")

    # ------------------------------------------------------------------ #
    # Explanation                                                        #
    # ------------------------------------------------------------------ #
    def explain(self):
        """Produce SHAP explanations using either in-memory tensors or disk paths."""
        from ai_explainability.io import to_torch_tensor

        # Resolve background & test data — prefer the in-memory kwargs passed
        # through the base class.
        if self.background_data is not None:
            background = to_torch_tensor(self.background_data)
        else:
            bg_path = self.get_path("background_data_path")
            if bg_path is None:
                raise ValueError(
                    "LSTMExplainer needs either a 'background_data' kwarg or "
                    "config['background_data_path']."
                )
            background = torch.load(bg_path)

        if self.test_data is not None:
            test_data = to_torch_tensor(self.test_data)
        else:
            test_path = self.get_path("test_data_path")
            if test_path is None:
                raise ValueError(
                    "LSTMExplainer needs either a 'test_data' kwarg or "
                    "config['test_data_path']."
                )
            test_data = torch.load(test_path)

        # Determine the subset to explain to save time
        explain_len = min(len(test_data), 50)
        test_subset = test_data[:explain_len]

        # Initialize Explainer
        if self.config["explainer_type"] == "gradient":
            explainer = shap.GradientExplainer(self.model, background)
            self.shap_values = explainer(test_subset)
        elif self.config["explainer_type"] == "deep":
            explainer = shap.DeepExplainer(self.model, background)
            self.shap_values = explainer.shap_values(
                test_subset, check_additivity=False
            )

        # Define the numpy versions consistently (exact 50 samples explained)
        self.raw_data_values = (
            test_subset.detach().cpu().numpy()
            if isinstance(test_subset, torch.Tensor)
            else np.asarray(test_subset)
        )

        # Extract values for the dictionary
        if hasattr(self.shap_values, "values"):
            # GradientExplainer returns an Explanation object
            val_to_plot = self.shap_values.values
        else:
            # DeepExplainer returns a list of arrays (one per output)
            val_to_plot = (
                self.shap_values[0]
                if isinstance(self.shap_values, list)
                else self.shap_values
            )

        # Align with the Multi-Target format for generate_notebook
        self.all_shap_values = {0: val_to_plot}

        # Predictions for the same subset
        with torch.no_grad():
            preds = self.model(test_subset).detach().cpu().numpy().flatten()
        self.all_predictions = {0: preds}

        print(f"SHAP explanation complete. Data shape: {self.raw_data_values.shape}")

    # ------------------------------------------------------------------ #
    # In-memory result view                                              #
    # ------------------------------------------------------------------ #
    def to_result(self):
        from ai_explainability.result import ExplanationResult

        return ExplanationResult(
            shap_values=dict(self.all_shap_values),
            predictions=dict(getattr(self, "all_predictions", {})),
            raw_data=None,  # 3D — not naturally a DataFrame
            raw_data_values=self.raw_data_values,
            feature_names=list(self.config.get("feature_names", [])),
            analysis="timeseries",
            model_type=self.config.get("model_type", "lstm"),
        )

    # ------------------------------------------------------------------ #
    # Disk outputs                                                       #
    # ------------------------------------------------------------------ #
    def plot_results(self):
        # Local import — nbformat / nbconvert chain is only needed when the
        # caller actually asks for a notebook report.
        from output.utils.report_gen import generate_notebook

        output_dir = self.config.get("output_dir", "output/")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nb_name = f"report_lstm_{timestamp}.ipynb"
        nb_path = os.path.join(output_dir, nb_name)

        generate_notebook(
            explainer_inst=self,
            all_shap_values=self.all_shap_values,
            raw_data=self.raw_data_values,
            output_path=nb_path,
        )

    def save_results_to_excel(self):
        """Flattens 3D LSTM SHAP values and saves to Excel with timestamps."""
        output_dir = self.get_path("output_dir") or self.config.get("output_dir")
        if output_dir is None:
            raise ValueError(
                "save_results_to_excel requires 'output_dir' to be set in the config."
            )
        os.makedirs(output_dir, exist_ok=True)

        if hasattr(self.shap_values, "values"):
            shap_array = self.shap_values.values
        else:
            shap_array = (
                self.shap_values[0]
                if isinstance(self.shap_values, list)
                else self.shap_values
            )

        look_back = self.config["look_back"]
        features = self.config["feature_names"]

        flat_cols_shap = [
            f"SHAP_{feat}_t-{look_back - 1 - i}"
            for i in range(look_back)
            for feat in features
        ]
        flat_cols_data = [
            f"Val_{feat}_t-{look_back - 1 - i}"
            for i in range(look_back)
            for feat in features
        ]

        shap_flat = shap_array.reshape(shap_array.shape[0], -1)
        data_flat = self.raw_data_values.reshape(self.raw_data_values.shape[0], -1)

        shap_df = pd.DataFrame(shap_flat, columns=flat_cols_shap)
        data_df = pd.DataFrame(data_flat, columns=flat_cols_data)

        preds = (
            self.all_predictions.get(0)
            if getattr(self, "all_predictions", None)
            else None
        )
        if preds is None:
            self.model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(self.raw_data_values).float()
                preds = self.model(test_tensor).numpy().flatten()

        pred_df = pd.DataFrame({"Model_Prediction": preds})
        output_df = pd.concat([data_df, pred_df, shap_df], axis=1)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"shap_audit_{timestamp}.xlsx")

        try:
            output_df.to_excel(output_path, index=False)
            print(f"Excel audit saved: {output_path}")
        except Exception:
            csv_path = output_path.replace(".xlsx", ".csv")
            output_df.to_csv(csv_path, index=False)
            print(f"Excel failed, saved CSV instead: {csv_path}")
