"""
Model & data introspection — auto-fill as much of the config as possible.

The goal is a friendlier onboarding flow: a user uploads a fitted model (and,
ideally, a labelled DataFrame) and the toolbox reads whatever metadata the
objects expose, fills in the config keys it can infer, and hands back a
structured list of the parameters it still needs. A UI (or the example
notebooks) renders that list as form fields; nothing here is interactive, so
the same code path works for a web service, a notebook, or a CLI.

Public surface
--------------
- :func:`inspect_model` — returns a :class:`ModelProfile`.
- :func:`resolve_config` — merges the profile's inferred values with the
  answers the caller collected into a config dict ready for
  :func:`ai_explainability.explain` (or to serialise as ``config.json``).

Design notes
------------
Detection is **attribute-based / duck-typed**, never ``isinstance`` against a
real framework class. That keeps this module import-cheap (no torch / sklearn
/ tensorflow pulled in) and lets it inspect a model whether or not the backend
that produced it is installed in the current environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence


# --------------------------------------------------------------------------- #
# Data containers                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class FieldSpec:
    """One parameter the caller still needs to supply.

    A UI turns this straight into a form control:

    - ``key``      — the config key to set.
    - ``prompt``   — human-readable question.
    - ``type``     — one of ``bool`` / ``int`` / ``str`` / ``enum`` /
      ``list`` / ``columns``. ``columns`` means "pick a subset of the data's
      column names" (feature selection); ``enum`` uses ``options``.
    - ``options``  — allowed values for ``enum`` / suggested values otherwise.
    - ``default``  — a sensible default the UI can pre-select.
    - ``required`` — if ``False`` the caller may skip it and take the default.
    - ``reason``   — why we're asking (shown as helper text).
    - ``group``    — coarse grouping for UI layout (``model`` / ``features`` /
      ``outputs`` / ``output_io``).
    """

    key: str
    prompt: str
    type: str = "str"
    options: list | None = None
    default: Any = None
    required: bool = False
    reason: str = ""
    group: str = "model"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "prompt": self.prompt,
            "type": self.type,
            "options": self.options,
            "default": self.default,
            "required": self.required,
            "reason": self.reason,
            "group": self.group,
        }


@dataclass
class ModelProfile:
    """Result of :func:`inspect_model`.

    Attributes
    ----------
    inferred:
        Config keys we're confident about (``analysis``, ``model_type``,
        ``package``, ``input_dim``, ``feature_names``, ``target_index`` …).
    missing:
        Ordered list of :class:`FieldSpec` the caller still needs to answer.
    warnings:
        Non-fatal issues (e.g. a data/model dimension mismatch).
    notes:
        Free-form facts discovered during inspection (feature count, output
        count, where feature names came from, etc.) — handy for display.
    """

    inferred: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: dict = field(default_factory=dict)

    # -- convenience views ------------------------------------------------- #
    def required_keys(self) -> list:
        return [f.key for f in self.missing if f.required]

    def to_dict(self) -> dict:
        return {
            "inferred": dict(self.inferred),
            "missing": [f.to_dict() for f in self.missing],
            "warnings": list(self.warnings),
            "notes": dict(self.notes),
        }

    def describe(self) -> str:
        """A compact human-readable summary (used by the notebooks)."""
        lines = ["Inferred configuration:"]
        for k, v in self.inferred.items():
            lines.append(f"  - {k}: {v!r}")
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  ! {w}" for w in self.warnings)
        if self.missing:
            lines.append("Still needed (ask the user):")
            for spec in self.missing:
                req = "required" if spec.required else "optional"
                lines.append(f"  - {spec.key} ({spec.type}, {req}): {spec.prompt}")
        else:
            lines.append("Nothing else needed — config is complete.")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Low-level, attribute-based detectors                                        #
# --------------------------------------------------------------------------- #
def _class_chain(obj: Any) -> list:
    """Return '<module>.<ClassName>' for every class in the MRO."""
    out = []
    for klass in type(obj).__mro__:
        module = getattr(klass, "__module__", "") or ""
        out.append(f"{module}.{klass.__name__}")
    return out


def _looks_tree_based(model: Any) -> bool:
    names = " ".join(_class_chain(model)).lower()
    tree_tokens = (
        "randomforest", "extratrees", "gradientboosting", "decisiontree",
        "xgb", "xgboost", "lgbm", "lightgbm", "catboost", "histgradientboosting",
    )
    if any(tok in names for tok in tree_tokens):
        return True
    # Duck-typed: sklearn tree ensembles expose estimators_ that are trees.
    return hasattr(model, "estimators_") and hasattr(model, "feature_importances_")


def _detect_torch_lstm(model: Any) -> bool:
    """True if a torch module contains an LSTM/GRU/RNN layer."""
    names = " ".join(_class_chain(model)).lower()
    if any(tok in names for tok in ("lstm", "gru", "rnn")):
        return True
    # Walk child modules if the framework exposes them.
    modules_fn = getattr(model, "modules", None)
    if callable(modules_fn):
        try:
            for sub in modules_fn():
                sub_name = type(sub).__name__.lower()
                if sub_name in {"lstm", "gru", "rnn"}:
                    return True
        except Exception:
            pass
    return False


def _first_linear_in_features(model: Any):
    """Best-effort input dimension from a torch module's first Linear layer."""
    modules_fn = getattr(model, "modules", None)
    if not callable(modules_fn):
        return None
    try:
        for sub in modules_fn():
            in_f = getattr(sub, "in_features", None)
            if in_f is not None:
                return int(in_f)
            # LSTM/GRU expose input_size instead.
            in_s = getattr(sub, "input_size", None)
            if in_s is not None:
                return int(in_s)
    except Exception:
        return None
    return None


def _keras_io_shapes(model: Any):
    """Return (n_inputs, n_outputs) from a keras model, or (None, None)."""
    n_in = n_out = None
    ishape = getattr(model, "input_shape", None)
    oshape = getattr(model, "output_shape", None)
    if isinstance(ishape, tuple) and ishape:
        n_in = ishape[-1]
    if isinstance(oshape, tuple) and oshape:
        n_out = oshape[-1]
    return n_in, n_out


# --------------------------------------------------------------------------- #
# Main entry point                                                            #
# --------------------------------------------------------------------------- #
def inspect_model(model: Any, data: Any = None, *, analysis: str | None = None) -> ModelProfile:
    """Inspect a fitted ``model`` (and optional ``data``) and return a profile.

    Parameters
    ----------
    model:
        A fitted estimator: scikit-learn / XGBoost, a ``torch.nn.Module``, or a
        Keras model. Only its attributes are read; the backend need not be
        importable here.
    data:
        Optional dataset used to (a) source feature names from a DataFrame
        header and (b) cross-check the feature count against the model.
    analysis:
        Override the inferred analysis type (``tabular`` / ``timeseries``).
    """
    from .io import detect_framework

    profile = ModelProfile()
    inferred = profile.inferred

    # ---- 1. Framework + model_type + analysis --------------------------- #
    framework = detect_framework(model)  # pytorch / tensorflow / unknown
    is_tree = _looks_tree_based(model)

    if is_tree:
        inferred["model_type"] = "random_forest"
        inferred["analysis"] = analysis or "tabular"
        profile.notes["family"] = "tree"
    elif framework == "pytorch":
        inferred["package"] = "pytorch"
        if _detect_torch_lstm(model):
            inferred["model_type"] = "lstm"
            inferred["analysis"] = analysis or "timeseries"
            profile.notes["family"] = "lstm"
        else:
            inferred["model_type"] = "feedforward"
            inferred["analysis"] = analysis or "tabular"
            profile.notes["family"] = "feedforward"
    elif framework == "tensorflow":
        inferred["package"] = "tensorflow"
        inferred["model_type"] = "feedforward"
        inferred["analysis"] = analysis or "tabular"
        profile.notes["family"] = "feedforward"
    else:
        # Unknown — leave analysis/model_type for the caller to choose.
        profile.notes["family"] = "unknown"

    # ---- 2. Feature count / input dimension ----------------------------- #
    n_features = _infer_n_features(model, framework, is_tree)
    if n_features is not None:
        profile.notes["n_features"] = n_features
        if inferred.get("model_type") == "lstm":
            inferred["input_dim"] = n_features
        # For tabular, input_dim isn't a config key; feature_names carries width.

    # ---- 3. Feature names ---------------------------------------------- #
    feature_names, source = _infer_feature_names(model, data, n_features)
    if feature_names is not None:
        inferred["feature_names"] = list(feature_names)
        profile.notes["feature_names_source"] = source

    # ---- 4. Outputs / targets ------------------------------------------ #
    n_outputs, is_classifier = _infer_outputs(model, framework, is_tree)
    if n_outputs is not None:
        profile.notes["n_outputs"] = n_outputs
        profile.notes["is_classifier"] = bool(is_classifier)

    # ---- 5. Data cross-checks ------------------------------------------ #
    data_columns = _data_columns(data)
    if data_columns is not None:
        profile.notes["data_columns"] = data_columns
        if n_features is not None and inferred.get("analysis") == "tabular":
            # For tabular, a plain DataFrame's width should match the model
            # input width (feature columns only). We can only warn softly
            # because the frame may legitimately carry extra target columns.
            if len(data_columns) < n_features:
                profile.warnings.append(
                    f"The data has {len(data_columns)} columns but the model "
                    f"expects {n_features} features — some inputs may be missing."
                )

    # ---- 6. Assemble the 'still needed' field specs -------------------- #
    profile.missing = _build_missing_fields(
        inferred, profile.notes, n_outputs, feature_names, data_columns
    )
    return profile


# --------------------------------------------------------------------------- #
# Inference helpers                                                            #
# --------------------------------------------------------------------------- #
def _infer_n_features(model, framework, is_tree):
    # sklearn / xgboost
    n = getattr(model, "n_features_in_", None)
    if n is not None:
        return int(n)
    # xgboost Booster
    get_booster = getattr(model, "get_booster", None)
    if callable(get_booster):
        try:
            booster = get_booster()
            fn = getattr(booster, "feature_names", None)
            if fn:
                return len(fn)
        except Exception:
            pass
    if framework == "pytorch":
        return _first_linear_in_features(model)
    if framework == "tensorflow":
        n_in, _ = _keras_io_shapes(model)
        return int(n_in) if isinstance(n_in, int) else None
    return None


def _infer_feature_names(model, data, n_features):
    # sklearn stores names when fit on a DataFrame. It's a numpy array of
    # numpy str_ objects, so coerce to plain Python strings.
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(n) for n in names], "model.feature_names_in_"
    # xgboost booster feature names
    get_booster = getattr(model, "get_booster", None)
    if callable(get_booster):
        try:
            fn = get_booster().feature_names
            if fn:
                return [str(n) for n in fn], "xgboost.get_booster().feature_names"
        except Exception:
            pass
    # From a labelled DataFrame header.
    cols = _data_columns(data)
    if cols is not None:
        return list(cols), "data header"
    return None, None


def _infer_outputs(model, framework, is_tree):
    """Return (n_outputs, is_classifier) as best we can."""
    is_classifier = hasattr(model, "predict_proba") or hasattr(model, "classes_")

    # MultiOutput wrappers expose one estimator per target.
    chain = " ".join(_class_chain(model)).lower()
    if "multioutput" in chain and hasattr(model, "estimators_"):
        try:
            return len(model.estimators_), is_classifier
        except Exception:
            pass

    # sklearn regressors carry n_outputs_.
    n_out = getattr(model, "n_outputs_", None)
    if n_out is not None:
        try:
            return int(n_out), is_classifier
        except Exception:
            pass

    if framework == "tensorflow":
        _, n = _keras_io_shapes(model)
        if isinstance(n, int):
            return n, is_classifier

    # Single-output fallback.
    return 1, is_classifier


def _data_columns(data):
    """Return a list of column names if ``data`` is a labelled DataFrame."""
    cols = getattr(data, "columns", None)
    if cols is None:
        return None
    try:
        return [str(c) for c in list(cols)]
    except Exception:
        return None


def _build_missing_fields(inferred, notes, n_outputs, feature_names, data_columns):
    """Compose the ordered list of parameters the caller still needs."""
    missing: list = []
    family = notes.get("family", "unknown")

    # --- model_type / analysis, if we couldn't infer them --------------- #
    if "analysis" not in inferred:
        missing.append(FieldSpec(
            key="analysis", prompt="What kind of analysis is this?",
            type="enum", options=["tabular", "timeseries"],
            default="tabular", required=True, group="model",
            reason="Could not infer from the model.",
        ))
    if "model_type" not in inferred:
        missing.append(FieldSpec(
            key="model_type", prompt="Which model type best describes your model?",
            type="enum",
            options=["random_forest", "feedforward", "lstm", "arima"],
            required=True, group="model",
            reason="Could not infer from the model.",
        ))

    # --- framework for NN models, if undetected ------------------------- #
    if family in {"feedforward", "unknown"} and "package" not in inferred \
            and inferred.get("model_type") in {"feedforward", "mlp", None}:
        if inferred.get("model_type") in {"feedforward", "mlp"}:
            missing.append(FieldSpec(
                key="package", prompt="Which framework was the model trained with?",
                type="enum", options=["pytorch", "tensorflow"],
                required=True, group="model",
                reason="Needed to build the prediction function.",
            ))

    # --- explainer_type for neural models ------------------------------- #
    if inferred.get("model_type") in {"feedforward", "mlp"}:
        missing.append(FieldSpec(
            key="explainer_type", prompt="Which SHAP explainer should be used?",
            type="enum", options=["kernel", "deep", "gradient"],
            default="kernel", required=False, group="model",
            reason="kernel is model-agnostic and safest; deep/gradient are faster but backend-dependent.",
        ))
    elif inferred.get("model_type") == "lstm":
        missing.append(FieldSpec(
            key="explainer_type", prompt="Which SHAP explainer should be used?",
            type="enum", options=["gradient", "deep"],
            default="gradient", required=False, group="model",
            reason="Gradient is the most portable for recurrent nets.",
        ))

    # --- LSTM-specific windowing ---------------------------------------- #
    if inferred.get("model_type") == "lstm":
        if "look_back" not in inferred:
            missing.append(FieldSpec(
                key="look_back", prompt="How many past time steps does the window cover?",
                type="int", required=True, group="model",
                reason="The LSTM window length cannot be read from the weights.",
            ))

    # --- feature names / selection -------------------------------------- #
    if feature_names is None:
        if data_columns:
            missing.append(FieldSpec(
                key="feature_names",
                prompt="Select the columns to use as input features.",
                type="columns", options=list(data_columns),
                default=list(data_columns), required=True, group="features",
                reason="No feature names were stored on the model; pick them from the data header.",
            ))
        else:
            missing.append(FieldSpec(
                key="feature_names",
                prompt="Name the input features (in order).",
                type="list", required=True, group="features",
                reason="No feature names on the model and no labelled data provided.",
            ))
    else:
        # We have names, but for tabular the user may want a subset.
        if data_columns and inferred.get("analysis") == "tabular":
            missing.append(FieldSpec(
                key="feature_names",
                prompt="Confirm or narrow the feature columns to explain.",
                type="columns", options=list(feature_names),
                default=list(feature_names), required=False, group="features",
                reason="Defaults to all detected features; deselect any you don't want explained.",
            ))

    # --- which outputs / targets to explain ----------------------------- #
    if n_outputs and n_outputs > 1:
        missing.append(FieldSpec(
            key="target_index",
            prompt=f"Which of the {n_outputs} outputs should be explained?",
            type="list", options=list(range(n_outputs)),
            default=list(range(n_outputs)), required=False, group="outputs",
            reason="Multi-output model — choose one or more target indices.",
        ))
        missing.append(FieldSpec(
            key="output_labels",
            prompt="Optionally name each output (for nicer reports).",
            type="list", required=False, group="outputs",
            reason="Human-readable labels shown in the Excel/notebook outputs.",
        ))

    # --- operational toggles (always asked) ----------------------------- #
    missing.append(FieldSpec(
        key="save_excel", prompt="Save a multi-sheet Excel audit to disk?",
        type="bool", default=False, required=False, group="output_io",
        reason="Off by default — keeps the run fully in-memory.",
    ))
    missing.append(FieldSpec(
        key="generate_notebook", prompt="Generate an executed Jupyter report?",
        type="bool", default=False, required=False, group="output_io",
        reason="Off by default — requires nbformat/nbconvert and writes to disk.",
    ))
    missing.append(FieldSpec(
        key="output_dir", prompt="Where should outputs be written? (only if saving)",
        type="str", default="output/", required=False, group="output_io",
        reason="Only used when save_excel or generate_notebook is on.",
    ))
    return missing


# --------------------------------------------------------------------------- #
# Config resolution                                                           #
# --------------------------------------------------------------------------- #
def resolve_config(profile: ModelProfile, answers: dict | None = None) -> dict:
    """Merge a profile's inferred values with user ``answers`` into a config.

    The returned dict is ready to splat into
    :func:`ai_explainability.explain` (``aie.explain(model=..., data=...,
    **config)``) or to serialise as a ``config.json`` for the file-based CLI.

    Raises
    ------
    ValueError
        If a required field is still missing after applying ``answers``.
    """
    answers = dict(answers or {})
    config = dict(profile.inferred)

    # Apply answers, dropping Nones (treated as "not answered").
    for key, value in answers.items():
        if value is not None:
            config[key] = value

    # Fill optional defaults for any unanswered non-required fields.
    for spec in profile.missing:
        if spec.key not in config and not spec.required and spec.default is not None:
            config[spec.key] = spec.default

    # Validate required fields.
    still_missing = [
        spec.key for spec in profile.missing
        if spec.required and config.get(spec.key) in (None, [], "")
    ]
    if still_missing:
        raise ValueError(
            "Cannot build config — these required fields are still unanswered: "
            f"{still_missing}. Provide them via the answers dict."
        )

    return config
