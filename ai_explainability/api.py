"""
High-level Python API.

The entry point of the library: a single
``explain()`` call accepts in-memory models and data, runs the full
explainability pipeline, and returns a structured result without ever
touching disk unless the caller asks for it.

Example (Databricks notebook style)::

    import ai_explainability as aie
    import mlflow.sklearn

    model = mlflow.sklearn.load_model("models:/my_model/production")
    df = spark.table("catalog.schema.features").toPandas()

    result = aie.explain(
        model=model,
        data=df,
        analysis="tabular",
        model_type="random_forest",
        feature_names=["Wind_Speed", "Temp", "Humidity"],
        target_index=[0, 1, 2],
    )
    display(result.to_dataframe())
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


# Keys that are passed through to the legacy ``config`` dict when present as
# kwargs. Anything else in ``**extras`` is copied verbatim so power-users can
# override advanced knobs without us having to enumerate them here.
_KNOWN_KWARGS = {
    "analysis",
    "model_type",
    "package",
    "feature_names",
    "target_index",
    "output_labels",
    "output_dir",
    "save_excel",
    "generate_notebook",
    "dataset_scope",
    "subset_end",
    "explainer_type",
    "input_dim",
    "hidden_size",
    "look_back",
    "look_ahead",
}


def _build_config(
    *,
    analysis: str,
    model_type: str,
    feature_names: Sequence[str] | None,
    target_index: int | Sequence[int] | None,
    output_labels: Mapping | Sequence | None,
    output_dir: str | None,
    save_excel: bool,
    generate_notebook: bool,
    dataset_scope: str | None,
    subset_end: int | None,
    explainer_type: str | None,
    input_dim: int | None,
    hidden_size: int | None,
    look_back: int | None,
    look_ahead: int | None,
    extras: Mapping[str, Any],
) -> dict[str, Any]:
    """Assemble the legacy-style config dict the explainers still consume."""
    cfg: dict[str, Any] = {
        "analysis": analysis,
        "model_type": model_type,
    }
    if feature_names is not None:
        cfg["feature_names"] = list(feature_names)
    if target_index is not None:
        cfg["target_index"] = target_index
    if output_labels is not None:
        cfg["output_labels"] = output_labels
    if output_dir is not None:
        cfg["output_dir"] = output_dir
    if dataset_scope is not None:
        cfg["dataset_scope"] = dataset_scope
    if subset_end is not None:
        cfg["subset_end"] = subset_end
    if explainer_type is not None:
        cfg["explainer_type"] = explainer_type
    if input_dim is not None:
        cfg["input_dim"] = input_dim
    if hidden_size is not None:
        cfg["hidden_size"] = hidden_size
    if look_back is not None:
        cfg["look_back"] = look_back
    if look_ahead is not None:
        cfg["look_ahead"] = look_ahead

    # Disk I/O is off by default in the programmatic API — callers opt in
    # explicitly, matching the Databricks-friendly defaults.
    cfg["save_excel"] = bool(save_excel)
    cfg["generate_notebook"] = bool(generate_notebook)

    # Pass through advanced extras.
    for k, v in extras.items():
        cfg.setdefault(k, v)

    return cfg


def explain(
    model: Any = None,
    data: Any = None,
    *,
    analysis: str = "tabular",
    model_type: str = "random_forest",
    feature_names: Sequence[str] | None = None,
    target_index: int | Sequence[int] | None = None,
    output_labels: Mapping | Sequence | None = None,
    output_dir: str | None = None,
    save_excel: bool = False,
    generate_notebook: bool = False,
    dataset_scope: str | None = None,
    subset_end: int | None = None,
    # Time-series / LSTM-specific ------------------------------------- #
    background_data: Any = None,
    test_data: Any = None,
    explainer_type: str | None = None,
    input_dim: int | None = None,
    hidden_size: int | None = None,
    look_back: int | None = None,
    look_ahead: int | None = None,
    **extras: Any,
):
    """Run an explainability analysis on an in-memory model and dataset.

    Parameters
    ----------
    model:
        A fitted model object (sklearn / xgboost / torch.nn.Module /
        statsmodels). Falls back to ``extras["model_path"]`` when ``None``.
    data:
        Tabular input as :class:`pandas.DataFrame`, :class:`numpy.ndarray`,
        ``pyspark.sql.DataFrame``, or a path. For time-series this is treated
        as ``test_data`` unless ``test_data`` is also given.
    analysis, model_type:
        Routing knobs, identical to the ``config.json`` keys.
    feature_names, target_index, output_labels:
        Pass-through metadata. Semantics identical to the JSON config.
    output_dir, save_excel, generate_notebook:
        Disk I/O controls. Default to ``None`` / ``False`` / ``False`` so a
        pure in-memory run never writes to the filesystem.
    background_data, test_data:
        LSTM-only in-memory tensors.
    explainer_type, input_dim, hidden_size, look_back, look_ahead:
        LSTM-only hyper-parameters.
    **extras:
        Additional config keys (e.g. ``model_path``, ``dataset_path``,
        ``background_data_path``) passed straight through to the explainer.
        Useful for hybrid flows where some inputs are in-memory and some
        still live on disk.

    Returns
    -------
    ai_explainability.ExplanationResult
        Structured result. See :class:`ai_explainability.ExplanationResult`.
    """
    if analysis not in ("tabular", "timeseries"):
        raise ValueError(
            f"analysis must be 'tabular' or 'timeseries'; got {analysis!r}"
        )

    # Filter extras down to things the config actually accepts (still accept
    # unknown keys — they may be new knobs we haven't catalogued).
    filtered_extras = {k: v for k, v in extras.items() if k not in _KNOWN_KWARGS}

    config = _build_config(
        analysis=analysis,
        model_type=model_type,
        feature_names=feature_names,
        target_index=target_index,
        output_labels=output_labels,
        output_dir=output_dir,
        save_excel=save_excel,
        generate_notebook=generate_notebook,
        dataset_scope=dataset_scope,
        subset_end=subset_end,
        explainer_type=explainer_type,
        input_dim=input_dim,
        hidden_size=hidden_size,
        look_back=look_back,
        look_ahead=look_ahead,
        extras=filtered_extras,
    )

    # Dispatch to the appropriate orchestrator.
    if analysis == "tabular":
        from analysis.tabular import run_tabular_analysis

        return run_tabular_analysis(config, model=model, data=data)

    # timeseries
    from analysis.timeseries import run_timeseries_analysis

    return run_timeseries_analysis(
        config,
        model=model,
        background_data=background_data,
        test_data=test_data,
        data=data,
    )
