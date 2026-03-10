# Multi-Output Regression Tutorial (Energy System Forecasting)

This tutorial explains how to use the toolbox to run SHAP analysis on the 
**multi-output regression example** which is an energy system forecasting model built 
with a Multi-Output Random Forest Regressor.

---

## 1. Model Overview: Multi-Output Regression

The core of this example is a **MultiOutputRegressor** wrapping a **RandomForestRegressor**, 
trained to simultaneously predict five continuous energy system variables from five input features.

* **Task:** Multi-Output Regression (5 simultaneous continuous outputs)
* **Algorithm:** `MultiOutputRegressor(RandomForestRegressor)` (scikit-learn) — 150 estimators, max depth 10
* **Input Features:** 5 tabular features per sample
* **Output:** 5 continuous targets predicted simultaneously

### Feature Description

| Feature | Description |
|---|---|
| `Wind Speed` | Wind speed sampled, clipped to 0–25 m/s |
| `Temperature` | Ambient temperature, clipped to -15–40 °C |
| `Previous Power` | Previous power output, clipped to 200–800 MW |
| `Previous Load` | Previous system load, clipped to 250–900 MW |
| `Hour` | Hour of day (0–23), used as a raw cyclical signal |

### Output Description

| Index | Target | Description |
|---|---|---|
| `0` | `Power Forecast` | Predicted power generation (MW) |
| `1` | `Load Forecast` | Predicted system load (MW) |
| `2` | `Frequency Deviation` | Grid frequency deviation (Hz) |
| `3` | `Voltage Level` | Per-unit voltage level |
| `4` | `Reserve Requirement` | Operating reserve needed (MW) |

### Output Generation Logic

The targets are derived from the input features with added noise to simulate 
real-world conditions:

```python
solar_component     = 80 * np.maximum(0, hour_sin)
power_forecast      = 45 * wind_speed + solar_component + 2.5 * temperature + 150 + noise
load_forecast       = 120 * |hour_sin| + 3.5 * |temperature - 18| + 250 + noise
frequency_deviation = 0.008 * (load_forecast - power_forecast) + noise
voltage_level       = 1.0 - 0.15 * frequency_deviation - 0.00035 * (load_forecast - 550) + noise
reserve_requirement = 0.18 * |load_forecast - power_forecast| + 30 + noise
```

---

## 2. Toolbox Integration & Directory Structure

To plug this example into the toolbox, your trained model and dataset must be 
stored in the `source/` directory at the project root. The explainability pipeline decouples 
training from explanation, so you only need to run the training script once and 
then reuse the saved artifacts.

```
Explainable_AI-SHAP_Analysis/
│
├── source/
│   ├── models/
│   │   └── rf_regress.pkl             ← Trained MultiOutputRegressor
│   └── data/
│       └── rf_regress_dataset.csv     ← Feature dataset (1000 samples)
│
├── examples/
│   └── tabular/
│       └── multioutput_regress/
│           ├── config.json            ← Ready to use configuration
│           └── rf_regress.py          ← Training script
│
├── output/                            ← Results saved here automatically
├── config.json                        ← Active config (copy from examples/)
└── main.py
```

- `source/models/`: Store your trained model here as a `.pkl` file
(e.g., `rf_regress.pkl`). This allows the explainer to load the model directly
without needing to rerun the training script.

- `source/data/`: Store your dataset here as a `.csv` file (e.g.,
`rf_regress_dataset.csv`). This is the data the SHAP Explainer will use to
compute feature attributions across all output targets.

By centralizing these artifacts, the toolbox can decouple model training from the 
explanation phase, enabling quick iterations on SHAP visualizations and Excel audit reports.

### Generating the Artifacts

If the `source/` files are not yet present, generate them by running the training script:

```bash
cd examples/tabular/multioutput_regress
python rf_regress.py
```

This script will:
1. Generate a synthetic energy dataset of 1000 samples and save it to `source/data/rf_regress_dataset.csv`
2. Train the Multi-Output model and save it to `source/models/rf_regress.pkl`

---

## 3. Configuration and Runner Execution

The toolbox is controlled entirely by a `config.json` file. 
This file maps your saved artifacts to the internal SHAP explainability logic 
and defines which outputs to generate.

The "ready to use" configuration for this example is located at:

```
examples/tabular/multioutput_regress/config.json
```

### Configuration File

```json
{
  "analysis": "tabular",
  "package": "sklearn",
  "model_type": "random_forest",
  "model_path": "source/models/rf_regress.pkl",
  "dataset_path": "source/data/rf_regress_dataset.csv",
  "output_dir": "output/",

  "dataset_scope": "subset",
  "subset_end": 200,

  "save_excel": true,
  "generate_notebook": true,

  "feature_names": [
    "Wind Speed",
    "Temperature",
    "Previous Power",
    "Previous Load",
    "Hour"
  ],

  "target_index": [0, 1],
  "output_labels": {
    "0": "Power Forecast",
    "1": "Load Forecast",
    "2": "Frequency Deviation",
    "3": "Voltage Level",
    "4": "Reserve Requirement"
  }
}
```

### Config Parameters Explained

**`analysis`**: Selects the analysis pipeline. Must be `"tabular"` for this 
example.

**`model_path`**: Path to the saved `.pkl` model file.

**`dataset_path`**: Path to the `.csv` dataset used for explanation.

**`dataset_scope`**: Controls how much data is used. 
`"subset"` with `subset_end: 200` explains only the first 200 rows
for faster iteration. We recommend to switch to `"full"` to explain all 1000 
samples.

**`feature_names`**: The list of input columns, in the **exact same order** as during training. 
This is critical: wrong order produces incorrect SHAP values.

```python
# Training order
X = df[["Wind Speed", "Temperature", "Previous Power", "Previous Load", "Hour"]]

# config.json must match exactly
"feature_names": ["Wind Speed", "Temperature", "Previous Power", "Previous Load", "Hour"]  ✓
```

**`target_index`**: Which output targets to explain. Accepts a **list of indices** for multi-output models. 
In this example `[0, 1]` runs SHAP analysis for `Power Forecast` and `Load Forecast`. 
Add more indices (e.g. `[0, 1, 2, 3, 4]`) to explain all five outputs.

**`output_labels`**: Names for each output index. Used in plot titles, Excel 
sheet names and notebook headers.

**`save_excel`**: Set to `true` to generate an Excel audit file with one sheet per explained target, 
containing raw SHAP values per sample.

**`generate_notebook`**: Set to `true` to generate a Jupyter Notebook summarising 
all results with inline plots.

---

## 4. Running the Analysis

Once your model and dataset are stored in `source/` and your `config.json` is in 
place, trigger the full explainability pipeline with a single command.

Copy the configuration to the project root:

```bash
cp examples/tabular/multioutput_regress/config.json config.json
```

Then run the analysis from the project root:

```bash
python main.py --config examples/tabular/multioutput_regress/config.json
```

The `main.py` entry point reads the config, routes to the tabular analysis module, 
runs the SHAP TreeExplainer for each target in `target_index` and writes all outputs to the `output/` directory.

### Expected Output Structure

```
output/
│
├── shap_audit_random_forest_TIMESTAMP.xlsx
│   ├── Sheet: "Power Forecast"  → Features + Prediction + SHAP values
│   └── Sheet: "Load Forecast"   → Features + Prediction + SHAP values
│
└── multi_report_random_forest_TIMESTAMP.ipynb
    └── Interactive notebook with all SHAP plots and summaries
```