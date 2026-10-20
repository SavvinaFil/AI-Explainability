# Running the toolbox in-memory (Databricks, Colab, plain Python)

The library accepts **either** file paths (legacy) **or** already-loaded
Python objects. The in-memory path is what you want on Databricks clusters
or anywhere else you'd rather avoid disk round-trips.

The API is a single function: `ai_explainability.explain(...)`.

```python
import ai_explainability as aie

result = aie.explain(
    model=my_fitted_model,        # sklearn, xgboost, torch, statsmodels, …
    data=my_dataframe,            # pd.DataFrame, pyspark.sql.DataFrame, ndarray, or path
    analysis="tabular",
    model_type="random_forest",
    feature_names=["Wind_Speed", "Temp", "Humidity"],
    target_index=[0, 1, 2],
)

result.shap_values           # dict[int, np.ndarray]
result.predictions           # dict[int, np.ndarray]
result.to_dataframe()        # flat DataFrame — great for display()
```

Nothing is written to disk. `output_dir` is not created. `save_excel` and
`generate_notebook` default to `False`.

---

## Do I have to fill in every parameter? No.

You only pass what the toolbox can't work out for itself:

- **`feature_names`** are read from your DataFrame header automatically — you can
  drop them if your data has named columns.
- **`model_type`** and **`package`** can be **auto-detected** from the model.

For the detection, use `inspect_model`, which reads the model's metadata, fills
in what it can, and hands back a list of anything still needed. A UI (or you)
answers the rest, and `resolve_config` assembles the final config:

```python
import ai_explainability as aie

profile = aie.inspect_model(model, df)     # reads model + DataFrame header
print(profile.describe())                  # shows inferred values + what's missing

config  = aie.resolve_config(profile, answers={"target_index": [0, 1, 2]})
result  = aie.explain(model=model, data=df, **config)
```

So the big `config.json` of the file-based workflow is **optional** — the
in-memory API needs only a handful of arguments, and `inspect_model` can infer
most of them.

---

## Trimming large outputs (many features)

Models with lots of features produce wide tables and busy plots. Keep only what
matters with `top_features` (the N strongest, computed per output) or
`select_features` (an explicit list):

```python
result = aie.explain(model=model, data=df, model_type="random_forest",
                     target_index=[0, 1, 2], top_features=10)   # or select_features=[...]
display(result.to_dataframe())             # trimmed to 10 features per target
```

Trimming flows through `to_dataframe()`, the Excel audit, and the reports; the
raw `result.shap_values` arrays stay full-size.

---

## Databricks — end-to-end

Install the library in a notebook cell (the `%pip` magic restarts Python
automatically once the install finishes):

```python
%pip install git+https://github.com/SavvinaFil/AI-Explainability
```

In a subsequent cell, load a model from Unity Catalog via MLflow, pull
features from a Delta table, and hand both to `aie.explain`:

```python
import ai_explainability as aie
import mlflow.sklearn

# 1. Model from Unity Catalog — no disk I/O, no pickle file
model = mlflow.sklearn.load_model("models:/my_catalog.my_schema.my_model/production")

# 2. Features from a Delta table. You can pass the Spark DataFrame directly —
#    aie.explain will call .toPandas() for you. If you'd rather do that step
#    yourself (e.g. to filter or sample first) it also accepts the pandas view.
features_sdf = spark.table("my_catalog.my_schema.features")

# 3. Explain
result = aie.explain(
    model=model,
    data=features_sdf,
    analysis="tabular",
    model_type="random_forest",
    feature_names=["Wind_Speed", "Temp", "Humidity"],
    target_index=[0, 1, 2],
)

# 4. Show the results inline in the notebook
display(result.to_dataframe())
```

### LSTM (time-series)

Pass the trained `torch.nn.Module` and your background / test tensors
directly:

```python
import torch
import ai_explainability as aie

# model is already trained and in-memory (e.g. from an MLflow run)
background = torch.load("...")  # or already a tensor / ndarray
samples    = torch.load("...")

result = aie.explain(
    model=model,
    analysis="timeseries",
    model_type="lstm",
    explainer_type="gradient",       # or "deep"
    background_data=background,
    test_data=samples,
    feature_names=["PV", "ghi", "PV_lag_24", ...],
    input_dim=12,
    hidden_size=16,
    look_back=6,
)

result.shap_values[0].shape  # (n_samples, look_back, n_features)
result.to_dataframe()        # columns flattened as SHAP_<feat>_t-<k>
```

---

## Mixing in-memory and file-path inputs

You can hybridise — any argument that isn't passed in memory will be read
from the config/path. For instance, a model on disk plus an in-memory
DataFrame:

```python
result = aie.explain(
    data=df,                                   # in-memory
    model_path="source/models/rf_classify.pkl",  # still on disk
    analysis="tabular",
    model_type="random_forest",
    feature_names=[...],
    target_index=[0, 1],
)
```

---

## When you do want files on disk

`save_excel` and `generate_notebook` default to `False`. Flip them on and
provide an `output_dir` to get the original Excel audit + Jupyter notebook
report:

```python
result = aie.explain(
    model=model,
    data=df,
    analysis="tabular",
    model_type="random_forest",
    feature_names=[...],
    target_index=[0, 1, 2],
    save_excel=True,
    generate_notebook=True,
    output_dir="/dbfs/FileStore/explanations/",
)
```

Or keep the analysis purely in-memory and serialise the result later:

```python
result = aie.explain(...)                 # no disk I/O
result.save_excel("/tmp/audit.xlsx")      # explicit, when you decide you want it
result.save_report_html("/tmp/report.html")  # recommended: kernel-free HTML report
```

### Reports

There are two report formats:

- **HTML (recommended)** — `result.save_report_html(path)` renders the plots
  in-process with matplotlib and embeds them in one self-contained file. No
  Jupyter kernel is launched, so it avoids the kernel/zmq problems that can
  affect notebook execution (notably on Windows), and it handles multi-output
  models cleanly (one section per output).
- **Jupyter notebook** — `generate_notebook=True` (or `result.save_notebook`)
  writes a `.ipynb`. By default it's saved **ready-to-run** (open it and "Run
  All"); pass `execute_notebook=True` to pre-render the plots in an isolated
  subprocess. Both reports share the same layout.

---

## PySpark support

`pyspark.sql.DataFrame` is accepted via duck-typing, so the core install
does **not** pull in `pyspark`. On Databricks this doesn't matter because
Spark is already on the cluster. Off-cluster (local, Colab, etc.) you can
pull pyspark explicitly:

```bash
pip install "ai-explainability[spark] @ git+https://github.com/SavvinaFil/AI-Explainability"
```

Under the hood the library calls `.toPandas()` on the Spark DataFrame, so
keep the slice you hand in small enough to fit on the driver node.
