# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

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
