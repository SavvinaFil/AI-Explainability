# Multi-Output Classification Tutorial (Energy System Operations)

This tutorial explains how to use the toolbox to run SHAP analysis on the 
**multi-output classification example** which is an energy system operations model built 
with a Multi-Output Random Forest Classifier.

---

## 1. Model Overview: Multi-Output Classification

The core of this example is a **MultiOutputClassifier** wrapping a **RandomForestClassifier**, 
trained to simultaneously predict ten binary operational decisions from ten input features.

* **Task:** Multi-Output Classification (10 simultaneous binary outputs)
* **Algorithm:** `MultiOutputClassifier(RandomForestClassifier)` 
(200 estimators, max depth 12)
* **Input Features:** 10 tabular features per sample
* **Output:** 10 binary targets predicted simultaneously (`0` or `1` per output)

### Feature Description

| Feature | Description                                                 |
|---|-------------------------------------------------------------|
| `Wind Speed` | Wind speed sampled, clipped to 0–25 m/s                     |
| `Solar Irradiance` | Solar irradiance, clipped to 0–1000 W/m²                    |
| `Temperature` | Ambient temperature, clipped to -10–40 °C                   |
| `Previous Generation` | Previous power generation, clipped to 250–1000 MW           |
| `Previous Load` | Previous system load, clipped to 300–1200 MW                |
| `Day Ahead Price` | Day-ahead electricity market price, clipped to 20–250 €/MWh |
| `Battery SOC` | Battery state of charge, 15–95%                             |
| `Grid Frequency` | Grid frequency sampled around 50 Hz                         |
| `Congestion Index` | Network congestion level, 0–1                               |
| `Hour` | Hour of day (0–23), encoded as cyclical sine component      |

### Output Description

| Index | Target | Description |
|---|---|---|
| `0` | `Export Mode` | System is exporting power (generation > load) |
| `1` | `Import Mode` | System is importing power (load > generation) |
| `2` | `Battery Charging` | Battery is being charged (low price, sufficient generation) |
| `3` | `Battery Discharging` | Battery is discharging (high price, sufficient SOC) |
| `4` | `Reserve Activation` | Operating reserve is activated (large power imbalance) |
| `5` | `Frequency Support Active` | Frequency support is active (frequency deviation > threshold) |
| `6` | `Congestion Management` | Congestion management is active (high congestion index) |
| `7` | `Curtailment Active` | Generation curtailment is active (excess generation + congestion) |
| `8` | `Peak Load Response` | Peak load response is active (high load during evening hours) |
| `9` | `High Price Operation` | System operating under high market price conditions |

### Output Generation Logic

The binary labels are derived from the input features to simulate 
realistic energy system decision-making:

```python
power_balance = total_generation - load_estimation

export_mode          = (power_balance > 50)
import_mode          = (power_balance < -50)
battery_charging     = (day_ahead_price < 75) & (battery_soc < 85) & (power_balance > 0)
battery_discharging  = (day_ahead_price > 115) & (battery_soc > 30) & (power_balance < 0)
reserve_activation   = (|power_balance| > 200)
frequency_support    = (|grid_frequency - 50| > 0.12)
congestion_mgmt      = (congestion_index > 0.7)
curtailment_active   = (total_generation > 900) & (congestion_index > 0.45)
peak_load_response   = (load_estimation > 800) & (hour >= 16) & (hour <= 22)
high_price_operation = (day_ahead_price > 130)
```

This creates ten interdependent binary decisions driven by market 
conditions, making SHAP analysis valuable for understanding which features 
activate each operational mode.

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
│   │   └── multioutput_classify_model.pkl    ← Trained MultiOutputClassifier
│   └── data/
│       └── multioutput_classify_dataset.csv  ← Feature dataset (1500 samples)
│
├── examples/
│   └── tabular/
│       └── multioutput_classify/
│           ├── config.json                   ← Ready to use configuration
│           └── multioutput_classify.py       ← Training script
│
├── output/                                   ← Results saved here automatically
├── config.json                               ← Active config (copy from examples/)
└── main.py
```

- `source/models/`: Store your trained model here as a `.pkl` file
(e.g., `multioutput_classify_model.pkl`). This allows the explainer to load the model directly
without needing to rerun the training script.

- `source/data/`: Store your dataset here as a `.csv` file (e.g.,
`multioutput_classify_dataset.csv`). This is the data the SHAP Explainer will use to
compute feature attributions across all output targets.

By centralizing these artifacts, the toolbox can decouple model training from the 
explanation phase, enabling quick iterations on SHAP visualizations and 
Excel audit reports.

### Generating the Artifacts

If the `source/` files are not yet present, generate them by running the training script:

```bash
cd examples/tabular/multioutput_classify
python multioutput_classify.py
```

This script will:
1. Generate a synthetic energy operations dataset of 1500 samples and save it to `source/data/multioutput_classify_dataset.csv`
2. Train the Multi-Output Classifier and save it to `source/models/multioutput_classify_model.pkl`

---

## 3. Configuration and Runner Execution

The toolbox is controlled entirely by a `config.json` file. 
This file maps your saved artifacts to the internal SHAP explainability logic 
and defines which outputs to generate.

The "ready to use" configuration for this example is located at:

```
examples/tabular/multioutput_classify/config.json
```

### Configuration File

```json
{
  "analysis": "tabular",
  "package": "sklearn",
  "model_type": "random_forest",
  "model_path": "source/models/multioutput_classify_model.pkl",
  "dataset_path": "source/data/multioutput_classify_dataset.csv",
  "output_dir": "output/",

  "dataset_scope": "whole",

  "save_excel": true,
  "generate_notebook": true,

  "feature_names": [
    "Wind Speed",
    "Solar Irradiance",
    "Temperature",
    "Previous Generation",
    "Previous Load",
    "Day Ahead Price",
    "Battery SOC",
    "Grid Frequency",
    "Congestion Index",
    "Hour"
  ],

  "target_index": [0, 1],
  "output_labels": {
    "0": "Export Mode",
    "1": "Import Mode",
    "2": "Battery Charging",
    "3": "Battery Discharging",
    "4": "Reserve Activation",
    "5": "Frequency Support",
    "6": "Congestion Management",
    "7": "Curtailment",
    "8": "Peak Load Response",
    "9": "High Price Operation"
  }
}
```

### Config Parameters Explained

**`analysis`**: Selects the analysis pipeline. Must be `"tabular"` for this example.

**`model_path`**: Path to the saved `.pkl` model file.

**`dataset_path`**: Path to the `.csv` dataset used for explanation.

**`dataset_scope`**: Controls how much data is used. 
`"subset"` with `subset_end: 200` explains only the first 200 rows
for faster iteration. We recommned you to switch to `"full"` to explain all samples.

**`feature_names`**: The list of input columns, in the **exact same order** as during training. 
This is critical: wrong order produces incorrect SHAP values.

```python
# Training order
X = df[["Wind Speed", "Solar Irradiance", "Temperature", "Previous Generation",
        "Previous Load", "Day Ahead Price", "Battery SOC", "Grid Frequency",
        "Congestion Index", "Hour"]]

# config.json must match exactly
"feature_names": ["Wind Speed", "Solar Irradiance", "Temperature", ...]  ✓

And not that this way:
"feature_names": ["Solar Irradiance", "Wind Speed", "Temperature", ...]

So NOT different order for the given features!!!
```

**`target_index`**: Which output targets to explain. Accepts a **list of indices** for multi-output models.
In this example `[0, 1]` runs SHAP analysis for `Export Mode` and `Import Mode`.
Add more indices (e.g. `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`) to explain all ten outputs.

**`output_labels`**: Names for each output index. Used in plot titles, Excel sheet names, and notebook headers.

**`save_excel`**: Set to `true` to generate an Excel audit file with one sheet per explained target,
containing raw SHAP values per sample.

**`generate_notebook`**: Set to `true` to generate a Jupyter Notebook summarising
all results with inline plots.

---

## 4. Running the Analysis

Once your model and dataset are stored in `source/` and your `config.json` is in 
place, trigger the full explainability pipeline with a single command.

```bash
python main.py --config examples/tabular/multioutput_classify/config.json
```

The `main.py` entry point reads the config, routes to the tabular analysis module, 
runs the SHAP Explainer for each target in `target_index` and writes all outputs to the `output/` directory.

### Expected Output Structure

```
output/
│
├── shap_audit_random_forest_TIMESTAMP.xlsx
│   ├── Sheet: "Export Mode"   → Features + Prediction + SHAP values
│   └── Sheet: "Import Mode"   → Features + Prediction + SHAP values
│
└── multi_report_random_forest_TIMESTAMP.ipynb
    └── Interactive notebook with all SHAP plots and summaries
```