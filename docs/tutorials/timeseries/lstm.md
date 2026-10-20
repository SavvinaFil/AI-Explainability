# Tutorial: Explaining an LSTM time-series forecaster

This guide shows how to explain an **LSTM forecasting model** (e.g. predicting PV
power or load from a window of recent history) with the toolbox. LSTMs process
data in **3D sequences** — `(samples, look_back, features)` — so SHAP attributes
each prediction back to *which feature at which time-step* mattered.

> **What you need:** a trained `torch.nn.Module`, a small **background** tensor
> (a reference sample of input windows) and a **test** tensor (the windows you
> want to explain), both shaped `(n, look_back, n_features)`. You don't need the
> targets — SHAP queries the model itself.

---

## 1. What gets explained

For a many-to-one LSTM (a `look_back` window → one forecast), SHAP returns a 3D
array of shape `(n_samples, look_back, n_features)`: the contribution of every
feature at every step in the window. The toolbox turns this into:

- a **beeswarm / importance** view over the flattened `feature_t-k` columns, and
- a **temporal relevance** plot: which time-steps (`t-0`, `t-1`, …) drive the
  forecast most.

---

## 2. Minimal example (in memory)

```python
import numpy as np
import torch
import ai_explainability as aie

# A trained LSTM (obs window -> one forecast). Use your trained model/weights.
model = ...                      # torch.nn.Module, already trained & in eval mode

LOOK_BACK, N_FEATURES = 6, 12
FEATURES = ["PV", "ghi", "PV_lag_24", "PV_lag_168", "PV_roll_mean_3",
            "PV_roll_std_3", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "month_sin", "month_cos"]

# Background = reference windows; test = windows to explain. Shape (n, look_back, feat).
background = torch.randn(100, LOOK_BACK, N_FEATURES)
test       = torch.randn(50,  LOOK_BACK, N_FEATURES)

result = aie.explain(
    model=model,
    analysis="timeseries",
    model_type="lstm",
    package="pytorch",
    explainer_type="gradient",      # "gradient" (portable) or "deep"
    background_data=background,
    test_data=test,
    feature_names=FEATURES,
    look_back=LOOK_BACK,
    input_dim=N_FEATURES,
    hidden_size=16,
)

result.shap_values[0].shape        # (50, 6, 12) = (samples, look_back, features)
result.to_dataframe()              # flat table: SHAP_<feature>_t-<k> columns
```

On Databricks you'd load the model from MLflow and the tensors from a Delta
table / volume — see [`docs/databricks.md`](../../databricks.md).

---

## 3. Let the toolbox fill in the config for you

You don't have to remember every parameter. `inspect_model` reads what it can
from the model and tells you what's left:

```python
profile = aie.inspect_model(model)
print(profile.describe())
# Inferred: model_type='lstm', analysis='timeseries', package='pytorch',
#           input_dim=12 (from the network)
# Still needed: look_back (required), feature_names, explainer_type (default 'gradient')

config = aie.resolve_config(profile, answers={
    "look_back": 6,
    "feature_names": FEATURES,
    "explainer_type": "gradient",
})
result = aie.explain(model=model, background_data=background, test_data=test, **config)
```

`inspect_model` can read the input width and recognise the LSTM, but it can't
know your **look-back window** or **feature names**, so those remain prompts.

---

## 4. Reading the results

```python
sv = result.shap_values[0]         # (n, look_back, n_features)
result.to_dataframe()              # SHAP_<feat>_t-<k> + Val_<feat>_t-<k> columns
```

The report (see §6) adds a **Temporal Relevance** plot showing how much each
time-step in the window contributes — useful for answering "does the model rely
on the most recent hour, or older history?".

---

## 5. Too many features? Trim the output

With a wide feature set the flattened output (`features × look_back`) gets large.
Trim what the plots and tables show with `top_features` / `select_features`:

```python
# keep the strongest feature-timestep columns
result = aie.explain(..., top_features=12)

# or keep specific base features (all their time-steps)
result = aie.explain(..., select_features=["PV", "ghi", "PV_lag_24"])
```

For LSTMs, `select_features` matches the **base feature name** and keeps all of
its time-steps; `top_features` keeps the N strongest flattened columns. Trimming
affects the report, the Excel audit, and `to_dataframe()` — the raw
`result.shap_values` array stays full-size.

---

## 6. Saving a report

The recommended output is the **kernel-free HTML report** (renders in-process,
so it can't hit Jupyter-kernel/zmq problems on Windows):

```python
result.save_report_html("output/lstm_report.html")   # beeswarm, importance, temporal
result.save_excel("output/lstm_audit.xlsx")
result.save_notebook("output/lstm_report.ipynb")     # ready-to-run notebook
```

Or let `explain()` write them: `aie.explain(..., save_excel=True,
generate_notebook=True, output_dir="output/")`.

---

## 7. File-based / CLI usage (legacy)

If you'd rather drive everything from files, save the model (`.pth`) and the
background/test tensors (`.pt`) to disk, write a `config.json`, and run the CLI.
`inspect_model` + `resolve_config` can generate most of this config for you.

```json
{
  "analysis": "timeseries",
  "package": "pytorch",
  "model_type": "lstm",
  "model_path": "source/models/lstm_model.pth",
  "background_data_path": "source/data/lstm_background_data.pt",
  "test_data_path": "source/data/lstm_data_to_explain.pt",
  "feature_names": ["PV", "ghi", "PV_lag_24", "PV_lag_168", "PV_roll_mean_3",
                    "PV_roll_std_3", "hour_sin", "hour_cos", "dow_sin",
                    "dow_cos", "month_sin", "month_cos"],
  "output_labels": ["PV"],
  "input_dim": 12,
  "hidden_size": 16,
  "look_back": 6,
  "explainer_type": "gradient",
  "output_dir": "output/",
  "save_excel": true,
  "generate_notebook": true
}
```

```bash
ai-explainability --config config.json
# or, from Python:
python -c "import ai_explainability as aie; aie.run(['--config', 'config.json'])"
```

> **Tip:** when the model reconstructs its architecture from the config (the
> file path case), `input_dim` and `hidden_size` must match the trained network.
> In the in-memory case you pass the already-built `model`, so those are only
> used as metadata.

---

## 8. Choosing the SHAP backend

| `explainer_type` | Notes |
| :--- | :--- |
| `gradient` | `shap.GradientExplainer` — the most **portable** choice for recurrent nets; recommended default. |
| `deep` | `shap.DeepExplainer` — can be faster but support for LSTM/RNN layers varies by SHAP/torch version. |

Both need a **background** tensor as the reference distribution (100-ish windows
is a good size).

---

## 9. Glossary — what every term means

| Term | What it means |
| :--- | :--- |
| **`analysis`** | `"timeseries"` for LSTM/ARIMA (vs `"tabular"` for trees / feedforward nets). |
| **`model_type`** | `"lstm"` here. |
| **`package`** | Framework of the model — `"pytorch"`. Auto-detected by `inspect_model`. |
| **`background_data`** | A reference sample of input windows used as the SHAP baseline (tensor/ndarray/path), shape `(n, look_back, features)`. |
| **`test_data`** | The windows you actually want to explain, same shape. |
| **`look_back`** | The window length — how many past time-steps the model sees (e.g. 6). Cannot be inferred from the weights, so you provide it. |
| **`input_dim`** | Number of features per time-step. |
| **`hidden_size`** | LSTM hidden units — only needed when the architecture is rebuilt from a config path. |
| **`feature_names`** | Names of the input features, in channel order. |
| **`explainer_type`** | SHAP backend: `"gradient"` (default) or `"deep"`. |
| **`explain_subset`** | How many test windows to explain (default 50). |
| **`top_features` / `select_features`** | Trim which features appear in the plots/results (see §5). |
| **`output_labels`** | Optional name(s) for the forecast target, e.g. `["PV"]`. |
| **`save_excel` / `generate_notebook` / `execute_notebook`** | Write the Excel audit / notebook; `execute_notebook=True` pre-renders the notebook (off by default — it's saved ready-to-run). |
| **`result.shap_values[0]`** | The 3D SHAP array `(samples, look_back, features)`. |
| **`result.to_dataframe()`** | Flattened `SHAP_<feat>_t-<k>` table. |
| **`result.save_report_html(path)`** | Kernel-free HTML report (recommended). |

---

## Related

- [Databricks / in-memory guide](../../databricks.md)
- [Tree-based forecasting tutorial](../tabular/tree_based_forecast.md)
- [RL agent tutorial](../rl/grid_rl_agent.md)
