import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import TimeseriesExplainerBase


class ARIMAExplainer(TimeseriesExplainerBase):
    """ARIMA / SARIMAX coefficient-based explainer.

    Accepts either a fitted pmdarima/statsmodels object via the ``model``
    kwarg, or loads it from ``config["model_path"]`` when the kwarg is
    omitted.
    """

    def load_model(self):
        """Resolve ``self.model`` into a fitted ARIMA-style object."""
        if self.model is None:
            from ai_explainability.io import to_fitted_model

            # ``to_fitted_model`` routes ``.joblib`` → joblib, ``.pkl`` → pickle.
            self.model = to_fitted_model(self.config.get("model_path"))
        print("ARIMA Model ready.")

    def explain(self):
        """Extract coefficients and statistical significance.

        For ARIMA, 'explanation' is often found in the summary table.
        """
        summary = self.model.summary()

        # Extract the coefficient table as a DataFrame
        results_as_html = summary.tables[1].as_html()
        self.stats_df = pd.read_html(results_as_html, header=0, index_col=0)[0]

        # Separate AR/MA terms from Exogenous features (the user-named ones)
        self.exog_importance = self.stats_df.loc[
            self.stats_df.index.isin(self.config.get("feature_names", []))
        ]

        # Prime the "all_*" attributes for the common result shape
        self.all_shap_values = {}  # ARIMA has coefficients, not SHAP values
        self.all_predictions = {}

    # ------------------------------------------------------------------ #
    # In-memory result view                                              #
    # ------------------------------------------------------------------ #
    def to_result(self):
        from ai_explainability.result import ExplanationResult

        return ExplanationResult(
            shap_values={},  # N/A for ARIMA
            predictions={},  # N/A
            raw_data=None,
            raw_data_values=None,
            feature_names=list(self.config.get("feature_names", [])),
            analysis="timeseries",
            model_type=self.config.get("model_type", "arima"),
            extras={
                "stats_df": self.stats_df,
                "exog_importance": self.exog_importance,
                "summary": str(self.model.summary()),
            },
        )

    def plot_results(self):
        """Visualizes feature importance based on coefficient weight."""
        output_dir = self.config.get("output_dir")
        if not output_dir:
            raise ValueError("plot_results requires 'output_dir' in config.")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Plot Exogenous Coefficients
        plt.figure(figsize=(10, 6))
        colors = ["red" if x < 0 else "green" for x in self.exog_importance["coef"]]
        self.exog_importance["coef"].plot(kind="barh", color=colors)

        plt.axvline(0, color="black", linewidth=0.8)
        plt.title("ARIMA Exogenous Feature Impact (Coefficients)")
        plt.xlabel("Coefficient Value (Direction of Impact)")

        save_path = os.path.join(output_dir, "arima_coefficients.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

        # 2. Diagnostic Plots (Standard for ARIMA)
        self.model.plot_diagnostics(figsize=(12, 8))
        diag_path = os.path.join(output_dir, "arima_diagnostics.png")
        plt.savefig(diag_path)
        plt.close()

        print(f"ARIMA Explanations saved to {output_dir}")
