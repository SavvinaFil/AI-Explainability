# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

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
