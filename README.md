# 🛡️ AI Explainability Analysis Toolbox

Neural Networks excel at navigating the complexities of the energy transition, but their decisions often remain "black boxes." This toolbox strips away the mystery by providing a standardized, model-agnostic framework for AI explainability using SHAP. It transforms complex model behavior into auditable, physically grounded insights—ensuring that when an AI makes a high-stakes decision, you can trace it back to the fundamental drivers of the energy system.

---

## 🧠 Explainability Analysis in a Nutshell

### What are Shapley Values?
Derived from cooperative game theory, **Shapley values** provide a mathematically rigorous way to distribute the "payout" (the model's prediction) among the "players" (the input features). In the context of Machine Learning, a Shapley value represents the **average marginal contribution** of a feature toward a specific prediction, accounting for all possible combinations of other features.

---

### Why is this useful for Energy AI?
In the energy sector, knowing *that* a model predicted a price spike or a solar drop is only half the battle. Explainability is the key to:

* **Feature Validation:** Verifying that the model prioritizes **physical drivers** (e.g., solar irradiance, ambient temperature).
* **Trust & Adoption:** Providing grid operators with the transparency needed to act on AI-driven insights by isolating the **real-world variables** that trigger specific alerts.
* **Model Debugging:** Diagnosing **systemic biases** or data leakage by identifying overpowering features.
* **Regulatory & Market Compliance:** Establishing a clear **audit trail** for automated decisions.

> **Key Takeaway:** While traditional "Feature Importance" tells you what the model values across the entire dataset, **SHAP tells you why the model made a specific decision right now.**

---


## ⚡ Quickstart

### 1. Prepare your Assets
Place your trained model and the dataset you want to explain in the `source/` directory:
* **models:** place your trained AI agent in the `models/` folder.
* **data:** place your data in the `data/` folder.

### 2. Configure your Analysis
Create a JSON file to define the analysis scope. This file tells the toolbox where your files are and how to interpret the outputs.

<details>
<summary><b>Example: Tabular Multi-Target Regression (config.json)</b></summary>

```json
{
  "analysis": "tabular",
  "model_type": "random_forest",
  "model_path": "source/models/energy_model.pkl",
  "dataset_path": "source/data/energy_data.csv",
  "output_dir": "output/",
  "feature_names": ["Wind_Speed", "Temp", "Humidity"],
  "target_index": [0, 1, 2],
  "output_labels": {
    "0": "Power Generation",
    "1": "Grid Load",
    "2": "Frequency"
  },
  "save_excel": true,
  "generate_notebook": true
}
```
</details>

### 3. Run the Toolbox
Execute the analysis via the command line using the `--config` flag:

* Run Time-Series Analysis
```bash
python main.py --config examples/timeseries/lstm/config.json
```

* Run Tabular Classification
```bash
python main.py --config examples/tabular/binary_classify/config.json
```

* Run Tabular Regression
```bash
python main.py --config examples/tabular/multioutput_regress/config.json
```

---

## 📊 Outputs

The toolbox generates two primary artifacts in the output/ folder:

1. SHAP Audit (.xlsx): A multi-sheet spreadsheet containing original feature values, model predictions, and SHAP values for every row. Each target index gets its own sheet.
2. Interpretation Report (.ipynb): A fully executed Jupyter Notebook containing Summary Plots, Feature Importance Bar Charts, and Temporal Analysis (for LSTMs).

You can find examples of the jupyter notebooks here:

| Feature | **Single Output Regression** | **Binary Classification** | **Multioutput Regression** |
| :--- | :--- | :--- | :--- |
| **Example** | [LSTM Report](./output/report_lstm_20260223_154441.ipynb) | [RF Classify Report](./output/report_random_forest_20260223_160452.ipynb) | [RF Regress Report](./output/report_random_forest_20260223_160626.ipynb) |

---

## 🛠️ Supported Models
* Tree-Based: RandomForest, XGBoost, DecisionTrees.
* Deep Learning: LSTM (PyTorch, more models under construction).


## 📖 Documentation
For detailed guides and tutorials, refer to our documentation suite:

* **[Tutorial: LSTM Time-Series Analysis](./docs/tutorials/timeseries/lstm.md):** A step-by-step guide to training and explaining Long Short-Term Memory networks for temporal data.
* **[Tutorial: Random Forest Binary Classification](./docs/tutorials/tabular/dc_binary_classify.md):** A comprehensive walkthrough for training and interpreting a binary classification model.
* **[Tutorial: Multi-Output Random Forest Classification](./docs/tutorials/tabular/dc_multioutput_classify.md):** A specialized guide for handling multi-target classification tasks and analyzing joint feature importance.
* **[Tutorial: Random Forest Regression](./docs/tutorials/tabular/dc_multioutput_regress.md):** A deep dive into training regression models and decoding the drivers behind continuous predictions.
* **[Configuration Guide](./docs/configuration.md):** A complete technical breakdown of all `config.yaml` parameters and environment settings.
* **[Explainability Theory](./docs/theory.md):** A detailed exploration of the mathematical foundations of Shapley Additive Explanations (SHAP).


## 📁 Project Structure

```text
/
├── analysis/
│   ├── tabular/                # Logic for CSV-based data (RF, XGB, etc.)
│   │   ├── treebased/          # Tree-specific explainers
│   │   └── __init__.py         # Tabular manager and registry
│   └── timeseries/             # Logic for 3D temporal data (LSTM, GRU)
│       └── lstm_explainer.py   # PyTorch-specific SHAP implementation
│
├── output/                     # Generated Reports and Audit logs
│   ├── utils/
│   │   └── report_gen.py       # The core Notebook generation engine
│   └── (files)                 # .xlsx and .ipynb outputs appear here
│
├── source/                     # Input Assets
│   ├── models/                 # Your .pkl or .pt model files
│   └── data/                   # Your .csv or .pt data files
│
├── examples/                   # Pre-configured JSON templates
├── main.py                     # Central entry point
└── README.md
```