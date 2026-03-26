# Photovoltaic (PV) Power Forecasting with LSTM

This project implements a Time-Series forecasting pipeline designed to predict solar power generation (PV) using Long Short-Term Memory (LSTM) networks. The model leverages historical power data, weather observations, and cyclical temporal features to capture the non-linear dynamics of solar energy production.

## 1. Model Overview: Many-to-One Forecasting
The core of the architecture is a **2-layer LSTM Forecaster**. It is configured as a "many-to-one" model, meaning it looks back at a window of historical data to predict a single value in the future.

* **Look-back Window:** 6 hours ($t-6$ to $t-1$).
* **Input Dimensions:** 12 features per time step, including lagged PV values, rolling statistics, and Global Horizontal Irradiance (GHI).
* **Temporal Encoding:** To help the model understand periodicity, we transform time variables (hour, day of week, month) into cyclical sine and cosine components.



## 2. SHAP Interpretability Strategy
Because LSTMs process data in 3D sequences (Samples, Time Steps, Features), standard feature importance methods often fail to capture the temporal nuances. To solve this, we utilize **SHAP (SHapley Additive exPlanations)** to attribute the model's predictions back to specific features at specific points in time.

To ensure the analysis is consistent and reproducible, the pipeline automatically exports "SHAP-Ready" artifacts:
* **Background Tensors:** A representative distribution of 100 training samples used as the reference baseline for the SHAP explainer.
* **Agnostic Explanations:** We save the raw 3D test tensors and the model state separately, allowing us to run post-hoc interpretability scripts without re-running the heavy training loop.

## 3. Toolbox Integration & Directory Structure
To explain your model, you must store your training outputs in the following directory structure within the projects root:

* **`source/models/`**: Store your trained PyTorch state dictionary here (e.g., `lstm_model.pth`). This allows the explainer to load the weights without needing the original training script.
* **`source/data/`**: Store your SHAP-specific datasets here as `.pt` tensors. This must include your `background_data.pt` (the reference distribution) and the `data_to_explain.pt` (the target samples).

By centralizing these artifacts, the toolbox can decouple the heavy model training from the explanation phase, enabling quick iterations on SHAP visualizations and summary reports.

## 4. Configuration and Runner Execution
The final step in the pipeline is to define a `config.json` file. This file acts as the "control center" for the toolbox, mapping your saved artifacts to the internal logic of the SHAP explainer. It ensures that the model architecture (input dimensions, hidden layers) and the time-series parameters (look-back window) match the training environment exactly.

To run the analysis, you must populate the configuration file with the paths to your exported `.pth` and `.pt` files. Below is the required JSON structure for a time-series LSTM analysis:

```json
{
  "analysis": "timeseries",
  "package": "pytorch",
  "model_type": "lstm",
  "model_path": "source/models/lstm_model.pth",
  "background_data_path": "source/data/lstm_background_data.pt",
  "test_data_path": "source/data/lstm_data_to_explain.pt",
  "dataset_path": "energy_forecasting_dataset.csv",
  "output_dir": "output/",
  "save_excel": true,
  "dataset_scope": "whole",
  "generate_notebook": true,                        
  "feature_names": [
    "PV", "ghi", "PV_lag_24", "PV_lag_168", 
    "PV_roll_mean_3", "PV_roll_std_3",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"
  ],
  "output_labels": ["PV"],
  "input_dim": 12,
  "hidden_size": 16,
  "look_back": 6,
  "look_ahead": 1,
  "explainer_type": "gradient"
}
```

## 5. Running the Analysis
Once your artifacts (model and tensors) are stored in the `source/` directory and your `config.json` is configured, you can trigger the automated interpretability pipeline. 

Run the following command from the project root. This command passes the configuration path to the main entry point, which routes the logic to the appropriate time-series runner:

```bash
python main.py --config examples/timeseries/lstm/config.json