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
result = aie.explain(...)             # no disk I/O
result.save_excel("/tmp/audit.xlsx")  # explicit, when you decide you want it
```

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
