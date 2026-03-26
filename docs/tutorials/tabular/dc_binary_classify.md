# Binary Classification Tutorial (Loan Approval Prediction)

This tutorial explains how to use the toolbox to run SHAP analysis on the 
**binary classification example** which is a loan approval prediction model built 
with a Random Forest Classifier.

---

## 1. Model Overview: Binary Classification

The core of this example is a **Random Forest Classifier** trained to predict whether a loan application should be approved or denied based on three input features.

* **Task:** Binary Classification (Loan Approved / Loan Denied)
* **Algorithm:** `RandomForestClassifier` (scikit-learn) — 50 estimators, max depth 4
* **Input Features:** 3 tabular features per sample
* **Output:** `0` (Loan Rejected) or `1` (Loan Approved)

### Feature Description

| Feature  | Description |
|---|---|
| `Age`  | Applicant's age |
| `Income per Year`  | Annual income (base + experience bonus) |
| `Years of Employment`  | Work experience relative to age |

### Loan Approval Logic

The label is deterministic simulating a realistic credit policy:

```python
df["Loan Approved"] = (
    (df["Income per Year"] > 40000) &
    (df["Age"] >= 30) & (df["Age"] <= 56) &
    (df["Years of Employment"] >= 2) &
    (
        ((df["Age"] > 35) & (df["Age"] <= 50) & (df["Years of Employment"] > 5)) |
        ((df["Income per Year"] > 60000) & (df["Years of Employment"] > 3))
    )
).astype(int)
```

This creates a nontrivial, multicondition decision boundary that challenges 
the model to learn compound rules.

---

## 2. Toolbox Integration & Directory Structure

To plug this example into the toolbox, your trained model and dataset must be 
stored in the `source/` directory at the project root. 
The explainability pipeline decouples training from explanation, so you only 
need to run the training script once and then reuse the saved artifacts.

```
Explainable_AI-SHAP_Analysis/
│
├── source/
│   ├── models/
│   │   └── rf_classify.pkl           ← Trained RandomForestClassifier
│   └── data/
│       └── rf_classify_data.csv      ← Feature dataset (1000 samples)
│
├── examples/
│   └── tabular/
│       └── binary_classify/
│           ├── config.json           ← Ready to use configuration
│           └── rf_classify.py        ← Training script
│
├── output/                           ← Results saved here automatically
├── config.json                       ← Active config (copy from examples/)
└── main.py
```

-`source/models/`: Store your trained scikit-learn model here as a `.pkl` file 
(e.g., `rf_classify.pkl`). This allows the explainer to load the model directly 
without needing to re-run the training script.

-`source/data/`: Store your dataset here as a `.csv` file (e.g., 
`rf_classify_data.csv`). This is the data the SHAP Explainer will use to 
compute feature attributions.

By centralizing these artifacts, the toolbox can decouple model training from the explanation phase, enabling quick iterations on SHAP visualizations and Excel audit reports.

### Generating the Artifacts

If the `source/` files are not yet present, generate them by running the training script:

```bash
cd examples/tabular/binary_classify
python rf_classify.py
```

This script will:
1. Generate a synthetic dataset of 1000 applicants and save it to `source/data/rf_classify_data.csv`
2. Train the Random Forest model and save it to `source/models/rf_classify.pkl`

---


## 3. Configuration and Runner Execution

The toolbox is controlled entirely by a `config.json` file. 
This file maps your saved artifacts to the internal SHAP explainability logic 
and defines which outputs to generate.

The "ready to use" configuration for this example is located at:

```
examples/tabular/binary_classify/config.json
```

### Configuration File

```json
{
  "analysis": "tabular",
  "package": "sklearn",
  "model_type": "random_forest",
  "model_path": "source/models/rf_classify.pkl",
  "dataset_path": "source/data/rf_classify_data.csv",
  "output_dir": "output/",

  "dataset_scope": "subset",
  "subset_end": 200,

  "save_excel": true,
  "generate_notebook": true,

  "feature_names": [
    "Age",
    "Income per Year",
    "Years of Employment"
  ],

  "target_index": 1,
  "output_labels": {
    "0": "Loan Denied",
    "1": "Loan Approved"
  }
}
```

### Config Parameters Explained

**`analysis`**: Selects the analysis pipeline. Must be `"tabular"` for this example.

**`model_path`**: Path to the saved `.pkl` model file.

**`dataset_path`**: Path to the `.csv` dataset used for explanation.

**`dataset_scope`**: Controls how much data is used. 
`"subset"` with `subset_end: 200` explains only the first 200 rows
for faster iteration. We recommend to switch to `"full"` to explain all 1000 samples.

**`feature_names`**: The list of input columns, in the **exact same order** as during training. This is critical: wrong order produces incorrect SHAP values.

```python
# Training order
X = df[["Age", "Income per Year", "Years of Employment"]]

# config.json must match exactly
"feature_names": ["Age", "Income per Year", "Years of Employment"]  ✓
```

**`target_index`**: Which class index to explain.`1` means we explain the "Loan Approved" class, which is typically the most actionable target in credit scoring.

**`output_labels`**: Human-readable names for each class index. Used in plot titles, Excel sheets, and notebook headers.

**`save_excel`**: Set to `true` to generate an Excel audit file with raw SHAP values per sample.

**`generate_notebook`**: Set to `true` to generate a Jupyter Notebook summarising all results with inline plots.

---

## 4. Running the Analysis

Once your model and dataset are stored in `source/` and your `config.json` is in place, trigger the full explainability pipeline with a single command.

Copy the configuration to the project root:

```bash
cp examples/tabular/binary_classify/config.json config.json
```

Then run the analysis from the project root:

```bash
python main.py --config examples/tabular/binary_classify/config.json
```

The `main.py` entry point reads the config, routes to the tabular analysis module, 
runs the SHAP TreeExplainer and writes all outputs to the `output/` directory.


### Expected Output Structure

```
output/
│
├── shap_audit_random_forest_TIMESTAMP.xlsx
│   └── Sheet: "Loan Approved" → Features + Prediction + SHAP values
│
└── multi_report_random_forest_TIMESTAMP.ipynb
    └── Interactive notebook with all SHAP plots and summaries
```