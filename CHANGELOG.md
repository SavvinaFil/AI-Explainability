# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- **Per-sample targets** — explain only the action taken. `aie.explain(...,
  targets='argmax')` explains, per sample, the highest output;
  `targets={'P': [0, 1, 2], 'tap': [6, 7, 8, 9]}` does the same per head.
  SHAP runs on a wrapped model that returns only the chosen logits, so the
  number of explained outputs drops from the logit count to the head count.
  Samples are grouped by chosen action, so the selection is fixed per sample
  and results equal the full run + `chosen_actions`. Feedforward / policy
  models only (`kernel`, `gradient`, `deep`).
- `targets_reduce='sum'` + `target_families={family: [heads]}` collapses heads
  into families for a cheaper overview run.
- `ExplanationResult.chosen_index`, `target_heads`, `target_families`.
  `chosen_output` / `chosen_actions` return the stored choice on such results.
- **Batched expected gradients for PyTorch** (`explainer_type='gradient'`).
  It replaces `shap.GradientExplainer` with the same estimator, run as:
  - one forward pass per chunk for all explained outputs;
  - all their gradients from one batched backward pass (`is_grads_batched`),
    falling back to one backward per output if the model can't be vmapped;
  - the interpolation points of all samples batched together.
  The chosen outputs (`targets=`) and frozen feature values travel with each
  sample's rows, so all samples run in one batched pass instead of one SHAP
  run per group. The baseline `E[f]` is computed per sample, with the frozen
  columns of the background set to that sample's values. A progress bar runs
  over the chunks.
  With `targets=`, each sample's choice travels with its rows, so all samples
  run in one batched pass instead of one SHAP run per action combination.
  New config: `gradient_nsamples` (default 200, as before), `gradient_batch_size`
  (rows per chunk, default sized to ~64M gradient floats) and `seed`.
  Measured on a 45-head entity policy (887 features): 6 → 1.45 s per state on
  one CPU thread.
- **Feature groups**: `explain(..., feature_groups={group: [features]})` or a
  callable `feature -> group`. `result.group_shap_values` (per-sample sum of
  member SHAP) and `result.group_importance(normalize='share')`. Ungrouped
  features are their own group. The Excel export gains a `group_importance`
  sheet and the HTML report a group table.
- **Dict observations**: `data` may be a mapping of `(n, *shape)` arrays. It is
  flattened in key order (C order within each array), the model receives the
  rebuilt dict, feature names follow the shape (`bus[3][1]`, or `bus[3].va`
  with `field_names={'bus': ['vm', 'va']}`), and each key is a default group.
  PyTorch modules and callables.
- **Frozen features**: `frozen_features=[...]` (names or dict keys) holds masks
  and validity flags at each sample's own value; their SHAP is exactly zero.
- **Per-sample row labels**: `row_labels=fn(sample_idx, feature) -> label` with
  `result.label_shap_values(output)` / `result.label_importance(output)` for
  tables whose rows mean different elements in different samples.
- **Cohorts**: `explain(..., cohorts={'day': mask, ...})` or
  `result.by_cohort(labels)`, `result.cohort_importance()`,
  `result.plot_cohort_heatmap()`; `cohort_importance` sheet in the Excel export.
- **Comparing agents**: `aie.compare({'A': res_a, 'B': res_b}, groups=...)`
  returns a group × (agent, cohort) table, normalised to shares by default
  because logit scales differ between networks; `aie.plot_heatmap(table)`.
- **Correctness checks on every result** (`ai_explainability.diagnostics`):
  `result.unmeasurable_features` with a warning for features constant over
  background ∪ data; `result.additivity_error` (per output mean / max / relative
  max of `|Σ SHAP + E[f] − f(x)|`) with a warning above `additivity_tolerance`
  (default 5%, 50% for `gradient`); `result.expected_value`; `result.settings`
  (explainer, background source and size, seed, `nsamples`, frozen features,
  versions). Tree classifiers skip the additivity check (SHAP checks it).
- **Explicit baseline for tree models**: `background_data` now also applies to
  tree explainers (interventional TreeSHAP); `seed=` controls the random
  background sample of feedforward models. The `explain()` docstring documents
  how the default background is chosen.
- **Report that scales to many outputs** (`save_report_html`): a summary page
  (warnings, group / feature importance overall and per cohort, run settings),
  one section per action head with the chosen-category distribution and a
  collapsible top-feature table per category, `heads=` to get head sections
  from a full (all-logit) result, and optional `figures=("group_heatmap",
  "cohort_heatmap")`.

### Changed
- The `kernel` and `deep` backends of the feedforward explainer (and
  `gradient` with TensorFlow) run SHAP once per group of samples that share
  their chosen actions (`targets=`) and frozen feature values; with neither,
  that is a single run as before. PyTorch `gradient` runs batched, see above.

### Fixed
- A plain-array `background_data` (or `data`) without `feature_names` no longer
  fails with "Shape of passed values is (n, d), indices imply (n, 0)". An empty
  `feature_names` list now means "no names", like `None`.

## [0.8.0] - 2026-10-20

### Added
- **Feature selection** to trim large outputs: `aie.explain(..., top_features=N)`
  keeps the N most important features (by mean |SHAP|, **per output**), and
  `select_features=[...]` keeps an explicit subset. Trimming flows through
  `to_dataframe()`, `save_excel()`, `save_report_html()`, the notebook report,
  **and the config-driven Excel writers of all explainers** (tree, feedforward,
  LSTM) — plots and result tables both shrink. The raw `result.shap_values`
  arrays are left full-width (trimming only affects the views). For multi-output
  models each output keeps its own top features. Settable on the result too
  (`result.top_features` / `result.select_features`).
- Tutorial `docs/tutorials/tabular/tree_based_forecast.md` — explaining a
  tree-based forecasting model, including feature selection, per-output drivers,
  and a glossary of every config term (`target_index`, `feature_names`,
  `output_labels`, `model_type`, …).

### Fixed
- Silenced a benign `FutureWarning` ("The NumPy global RNG was seeded …") emitted
  from inside `shap.summary_plot` during report generation. Filtered only around
  the shap call, scoped so it doesn't affect other warnings.

### Docs
- Rewrote `docs/tutorials/timeseries/lstm.md` in the tutorial style (in-memory
  first, `inspect_model` auto-config, feature selection, glossary), and updated
  `docs/databricks.md` + `README.md` to note that the full `config.json` is
  optional (the in-memory API + `inspect_model` infer most parameters) and to
  document `top_features` / `select_features` and the kernel-free HTML report.

## [0.7.0] - 2026-10-20

### Added
- `ExplanationResult.save_report_html(path, ...)` — a **kernel-free** HTML report
  with the **same layout** as the `.ipynb` report (beeswarm, global importance
  bar, temporal relevance for LSTMs, top-5 drivers, and identical narrative
  text). Both reports now draw their plots and wording from a shared module,
  `output/utils/report_content.py`, so they stay in sync. The HTML report renders
  in-process with matplotlib and embeds base64 PNGs, so no Jupyter kernel is
  launched and the pyzmq/kernel failures that affect the notebook cannot occur —
  making it the recommended report for multi-output / RL policies and Windows.
  Supports `output_labels` (name each action logit) and `max_outputs`.
- Sample report `output/report_rl_agent_example.html` showing the EnliteAI
  all-discrete multi-head structure (23 action logits).

### Changed
- The notebook report generator (`report_gen.py`) now uses the shared
  `report_content` plot/text helpers instead of inline code, so the `.ipynb` and
  `.html` reports are guaranteed to have an identical layout.

## [0.6.2] - 2026-10-19

### Fixed
- Silenced the benign sklearn "KMeans is known to have a memory leak on Windows
  with MKL" UserWarning emitted when the feedforward/RL kernel-SHAP path
  summarizes the background set with `shap.kmeans`. The message is filtered only
  around that call; correctness is unchanged.

## [0.6.1] - 2026-10-19

### Fixed
- Notebook report generation could crash the **entire host process** with a
  native fault (`EXCEPTION_ACCESS_VIOLATION` in pyzmq/libzmq) on some Windows
  setups, because the notebook was executed in-process via a Jupyter kernel and
  a native crash cannot be caught by Python. Now:
  - the `.ipynb` is written to disk **before** any execution, so it always
    exists;
  - execution is **opt-in** via `execute_notebook=True` (default `False`) — by
    default a ready-to-run notebook is saved with code cells, which you open and
    "Run All";
  - when enabled, execution runs in an **isolated subprocess** (`jupyter
    nbconvert --execute`), so a kernel/zmq crash is contained and the unexecuted
    notebook is still delivered with a clear message.

### Changed
- The Jupyter report is no longer pre-executed by default (set
  `execute_notebook=True` to pre-render plots). The Excel audit is unaffected.

## [0.6.0] - 2026-10-19

### Added
- `ExplanationResult.chosen_actions(heads)` — apply the chosen-action (argmax)
  view to several discrete heads at once, e.g.
  `{"P": [0,1,2], "Q": [3,4,5], "tap": [6,7,8,9]}`, returning
  `{head: (chosen_idx, chosen_shap)}`. Supports agents whose action space is any
  mix of continuous and discrete heads (including fully discrete P/Q/tap).
- RL tutorial: section on fully-discrete multi-head agents.

## [0.5.1] - 2026-10-19

### Fixed
- Notebook report generation no longer hardcodes `kernel_name='python3'`, which
  caused `Warning: Execution failed. Error: No such kernel named python3` and a
  report saved **without rendered plots** in environments where no `python3`
  kernel spec is registered. `report_gen` now picks an available kernel (prefers
  `python3`, else any registered kernel), best-effort registers the current
  interpreter if none exists, and prints an actionable message if execution is
  still not possible. The `.ipynb` is always written either way.

## [0.5.0] - 2026-10-17

### Added
- **Reinforcement-learning / multi-output policy support** in the feedforward
  explainer.
  - `FeedForwardExplainer` now explains **every output** of a network (honoring
    `target_index` for a subset), returning one SHAP array per output. This
    covers RL policy nets with multiple action components — continuous setpoints
    (P/Q) and discrete tap logits are handled identically.
  - `model` may be a **plain callable** `obs(ndarray) -> values(ndarray)` (wrap
    any framework's policy heads into one 2D array); KernelSHAP only.
  - `ExplanationResult.chosen_output(among=[...])` attributes the **argmax
    (selected) action** per state — "why did the agent pick this action here".
  - `save_results_to_excel` writes one sheet per action, named via `output_labels`.
  - New model_type aliases `rl_policy` / `policy` / `rl_agent` (→ feedforward).
- Tutorial: `docs/tutorials/rl/grid_rl_agent.md` — explaining a power-grid RL
  agent, including the callable-wrapper integration pattern and `chosen_output`.

### Changed
- The feedforward explainer no longer collapses a multi-output model to a single
  output. A 2-logit classifier that previously returned only "class 1" now
  returns both outputs by default (pass `target_index` to select).

## [0.4.1] - 2026-10-17

### Changed
- `explain()` now prints a one-line reminder when a run produced no files
  (neither `save_excel` nor `generate_notebook` was set), pointing to the flags
  and `result.save_excel()` / `result.save_notebook()`. Silence it with the
  `AI_EXPLAINABILITY_QUIET=1` environment variable. This addresses confusion
  where an in-memory run appears to "do nothing" (no files, no error).

### Notes
- Report outputs are a SHAP audit (`.xlsx`) and an executed Jupyter report
  (`.ipynb`). There is no PDF output. `test_in_memory_local.py` now includes an
  end-to-end example that writes and verifies both.

## [0.4.0] - 2026-09-21

### Added
- **Auto-configuration / model introspection** (`ai_explainability/introspect.py`).
  - `inspect_model(model, data=None)` reads a fitted model's metadata and returns
    a `ModelProfile` with `inferred` config keys, a `missing` list of form-field
    specs the caller still needs, and `warnings` (e.g. data/model dimension
    mismatch). Detection is attribute-based, so no ML framework is imported.
  - `resolve_config(profile, answers)` merges inferred values with user answers
    into a config ready for `explain()` or `config.json`; raises if a required
    field is unanswered.
  - Supports scikit-learn / XGBoost, PyTorch, and TensorFlow/Keras.
  - Feature names are sourced from `feature_names_in_` or the uploaded data's
    header; `type="columns"` fields drive "select which columns to explain".
  - Exposed as `aie.inspect_model`, `aie.resolve_config`, `aie.ModelProfile`,
    `aie.FieldSpec` (lazily — importing the package pulls in no heavy deps).
- **Example notebooks** (`examples/notebooks/`): `package_based_walkthrough.ipynb`
  (in-memory) and `file_based_walkthrough.ipynb` (files → `config.json` → CLI),
  each annotated as a UI specification.
- `.gitattributes` to normalize line endings (LF) and prevent whole-tree CRLF
  churn on Windows/OneDrive checkouts.

## [0.3.0]

### Added
- **Feedforward / MLP explainer** (`analysis/tabular/neural/ff_explainer.py`) for
  PyTorch and TensorFlow/Keras models, with a user-selectable SHAP backend via
  `explainer_type`: `kernel` (model-agnostic), `deep`, or `gradient`.
- Framework detection and a numpy→numpy prediction-function builder
  (`detect_framework`, `make_predict_fn`) in `io.py`.
- `tensorflow` optional extra in `pyproject.toml`.

### Changed
- Tabular router (`analysis/tabular/__init__.py`) now lazily loads the explainer
  by `model_type`, so tree-based runs never import torch/tensorflow.

### Fixed
- **LSTM explainer hardening**: validate `explainer_type` with a clear error and
  default; unify gradient/deep on `.shap_values()`; stop leaking global torch
  grad state; configurable `explain_subset`; 3D input-shape validation; removed
  dead code.

## [0.2.2]

### Fixed
- `ANALYSIS_ROUTER` regressed to returning `None` from `.get()` (breaking the
  `main.py` / `ai_explainability.run(...)` CLI path); replaced the fragile dict
  subclass with a plain dict of lazy call-time wrappers.
- `generate_notebook` crashed on `raw_data=None` (ARIMA `save_notebook`).
- `explain_subset` / `kernel_nsamples` / `background_size` were silently dropped
  before reaching the config.

## [0.2.1]

### Fixed
- Native multi-output `RandomForestRegressor` raised `IndexError` and produced
  identical SHAP values across targets. `RFExplainer` now dispatches on
  `MultiOutputRegressor`/`MultiOutputClassifier` via `isinstance` and slices the
  native 3D SHAP output per target.

## [0.2.0]

### Added
- **In-memory API**: `ai_explainability.explain(...)` accepts fitted model
  objects and pandas / pyspark / ndarray data, returning an `ExplanationResult`
  with no disk I/O unless explicitly requested.
- I/O coercion helpers (`io.py`), `ExplanationResult` container (`result.py`),
  Databricks / in-memory documentation, and a `spark` extra.

### Changed
- Heavy imports (torch, shap, nbformat, pyspark) are now deferred so
  `import ai_explainability` stays cheap.

## [0.1.0]

### Added
- Initial pip-installable packaging (`pyproject.toml`, `ai_explainability` shim
  package, console script) installable from GitHub via pip / uv / Databricks.
