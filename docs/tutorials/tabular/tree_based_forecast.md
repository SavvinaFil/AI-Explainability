# Tutorial: Explaining a tree-based forecasting model

This guide shows how to explain a **tree-based forecasting model** (RandomForest,
XGBoost, …) with the toolbox — the common setup where a model predicts a future
value (next-hour load, day-ahead price, PV output) from **many** engineered
features (lags, rolling stats, weather, calendar). It mirrors the in-memory
Databricks workflow in [`docs/databricks.md`](../../databricks.md), with extra
focus on two things forecasting teams ask about: **keeping the output readable
when you have lots of features**, and **what the configuration terms mean**.

> **What you need:** a fitted tree model and a table of rows to explain, with
> **named columns** (the header gives the feature names). You do *not* need the
> target values — SHAP queries the model itself.

---

## 1. What gets explained

Tree models use `shap.TreeExplainer` — fast and exact for trees. For each row you
explain, SHAP returns one value per feature: how much that feature pushed the
prediction up or down for that row.

- **Single-target forecast** (one value, e.g. next-hour load) → one set of SHAP
  values.
- **Multi-target forecast** (several horizons/targets at once, e.g. t+1, t+2,
  t+3, or a multi-output regressor) → **one set of SHAP values per target**, and
  — importantly — the drivers can differ per target (see §4).

---

## 2. Minimal example (in memory)

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import ai_explainability as aie

# A fitted forecasting model (here trained inline; use your trained model).
FEATURES = ["load_lag_1", "load_lag_24", "load_roll_mean_24", "temp", "temp_lag_24",
            "ghi", "wind", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_holiday"]
X = pd.DataFrame(np.random.randn(500, len(FEATURES)), columns=FEATURES)
y = np.random.randn(500)                       # next-hour load
model = RandomForestRegressor(n_estimators=200).fit(X, y)

# Explain a batch of rows.
result = aie.explain(
    model=model,
    data=X.iloc[:200],          # rows to explain (DataFrame with a header)
    analysis="tabular",
    model_type="random_forest", # tree family (covers RandomForest / XGBoost / …)
    feature_names=FEATURES,
)

result.shap_values[0]          # (200, n_features) SHAP values
result.to_dataframe()          # flat table of inputs + predictions + SHAP
```

On Databricks you'd load the model from MLflow and the rows from a Delta table —
see [`docs/databricks.md`](../../databricks.md). You can also pass a Spark
DataFrame directly.

---

## 3. Too many features? Trim the output

Forecasting models often have **dozens of features**, which makes the plots and
the result tables unwieldy. Two options on `aie.explain(...)` trim **both the
plots and the results** down to what you care about:

```python
# (a) keep only the N most important features (by mean |SHAP|)
result = aie.explain(..., top_features=10)

# (b) keep an explicit list of features
result = aie.explain(..., select_features=["temp", "load_lag_24", "ghi"])
```

| Option | Meaning |
| :--- | :--- |
| `top_features=N` | Keep the **N most important** features (largest mean absolute SHAP). Computed **per output**. |
| `select_features=[...]` | Keep exactly these features (in this order). Takes precedence over `top_features`. |

This trimming flows through everything the toolbox shows you — `to_dataframe()`,
the Excel audit, and the HTML/notebook report all shrink to the chosen features.
Your **raw** `result.shap_values` arrays are left untouched (full width), so you
never lose data — only the *views* are trimmed.

```python
result = aie.explain(model=model, data=X, model_type="random_forest",
                     feature_names=FEATURES, top_features=8)
result.save_report_html("output/forecast_report.html")   # report shows 8 features
result.to_dataframe()                                      # table shows 8 features
```

---

## 4. Multiple outputs → per-output features

If your model forecasts **several targets at once** (e.g. a horizon t+1…t+3, or a
`MultiOutputRegressor`), name the targets you want with `target_index`, and the
toolbox explains **each target separately**. With `top_features`, each target
keeps **its own** top drivers — target *t+1* might be driven by recent lags while
*t+24* leans on weather:

```python
result = aie.explain(
    model=multi_horizon_model,
    data=X,
    analysis="tabular",
    model_type="random_forest",
    feature_names=FEATURES,
    target_index=[0, 1, 2],                         # explain 3 horizons
    output_labels={"0": "t+1", "1": "t+2", "2": "t+3"},
    top_features=6,                                 # 6 drivers *per* horizon
)

result.shap_values[0]   # SHAP for horizon t+1
result.shap_values[2]   # SHAP for horizon t+3 (possibly different top features)
result.to_dataframe()   # per-target SHAP columns, each trimmed to that target's top-6
```

The report and Excel audit get **one section/sheet per target**, each showing
that target's own top features.

---

## 5. Saving a report

By default an in-memory run writes nothing. The recommended output is the
**kernel-free HTML report** (renders in-process, so it can't hit Jupyter-kernel
problems on Windows), which has the same layout as the notebook report:

```python
result.save_report_html("output/forecast_report.html")   # one section per target
result.save_excel("output/forecast_audit.xlsx")          # one sheet per target
result.save_notebook("output/forecast_report.ipynb")     # ready-to-run notebook
```

Or let `explain()` write them: `aie.explain(..., save_excel=True,
generate_notebook=True, output_dir="output/")`.

---

## 6. Glossary — what every term means

These are the configuration terms you'll see in `aie.explain(...)` and in
`config.json`.

| Term | What it means |
| :--- | :--- |
| **`analysis`** | The problem family: `"tabular"` (trees, feedforward nets) or `"timeseries"` (LSTM, ARIMA). Tree forecasting is `"tabular"`. |
| **`model_type`** | Which explainer to use: `"random_forest"` (any tree model — RandomForest, XGBoost, DecisionTree), `"feedforward"`, `"lstm"`, `"arima"`. |
| **`model`** | Your fitted model object (or `model_path` to a `.pkl` on disk). |
| **`data`** | The rows to explain, as a DataFrame / ndarray / Spark DataFrame (or `dataset_path` to a `.csv`). |
| **`feature_names`** | The names of the input features, in column order. Usually taken from the DataFrame header; name them explicitly if your data has no header. |
| **`target_index`** | **Which output(s) to explain**, for a multi-output model — an integer or a list, e.g. `0` or `[0, 1, 2]`. For a single-output forecast leave it at the default (`0`). (This is the `target_idx` you saw: it just selects which forecast target/horizon to explain.) |
| **`output_labels`** | Optional human names for each output, e.g. `{"0": "t+1", "1": "t+2"}` — used to title report sections and Excel sheets. |
| **`top_features`** | Keep only the N most important features in the plots/results (per output). |
| **`select_features`** | Keep exactly this list of features in the plots/results. |
| **`explainer_type`** | The SHAP algorithm. For trees this is `"tree"` (chosen automatically); it matters mainly for neural models (`"kernel"` / `"deep"` / `"gradient"`). |
| **`save_excel`** | If `True`, write the multi-sheet Excel audit to `output_dir`. Default `False`. |
| **`generate_notebook`** | If `True`, write the Jupyter report to `output_dir`. Default `False`. |
| **`execute_notebook`** | If `True`, also *run* the notebook to pre-render plots (in an isolated subprocess). Default `False` — the notebook is saved ready-to-run. |
| **`output_dir`** | Where to write files when `save_excel` / `generate_notebook` are on. |
| **`background_data`** | A reference sample used by kernel/deep/gradient SHAP (neural models). Not needed for tree models. |
| **`package`** | Framework for neural models: `"pytorch"` / `"tensorflow"`. Not used for trees. |

And the key objects you get back:

| Object / attribute | What it is |
| :--- | :--- |
| **`ExplanationResult`** | What `aie.explain(...)` returns. |
| **`result.shap_values`** | `{target_index: array}` — the SHAP values per output, shape `(n_rows, n_features)`. Always full width (trimming only affects views). |
| **`result.predictions`** | `{target_index: array}` — the model's prediction per output. |
| **`result.to_dataframe()`** | One flat table: inputs + predictions + SHAP columns (trimmed if you set `top_features`/`select_features`). |
| **`result.save_report_html(path)`** | Kernel-free HTML report (recommended). |
| **`result.save_excel(path)` / `result.save_notebook(path)`** | Excel audit / Jupyter report. |

---

## 7. A note on "trimming inputs" vs "trimming the report"

`top_features` / `select_features` trim **what is shown**, not what the model
uses. A tree model needs *all* the features it was trained on to make a
prediction, so you can't drop input features. What you *can* do — and what these
options do — is focus the explanation on the features that matter, which is
exactly what you want when a model has dozens of them.
