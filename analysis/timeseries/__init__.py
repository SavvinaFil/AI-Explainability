# analysis/timeseries/__init__.py
#
# Lazy model loader — importing this subpackage must not pull in torch or
# statsmodels. Each explainer class is imported on demand, inside
# ``run_timeseries_analysis``, based on ``config["model_type"]``.

from importlib import import_module


def _load_explainer_class(model_type: str):
    if model_type == "lstm":
        return import_module("analysis.timeseries.lstm_pytorch").LSTMExplainer
    if model_type == "arima":
        return import_module("analysis.timeseries.arima_stats").ARIMAExplainer
    return None


def run_timeseries_analysis(
    config,
    *,
    model=None,
    background_data=None,
    test_data=None,
    data=None,
):
    """Orchestrates the specific timeseries model based on config.

    Parameters
    ----------
    config:
        Legacy configuration dict. ``*_path`` entries become optional when
        the corresponding in-memory kwarg is supplied.
    model:
        Fitted model object (``torch.nn.Module`` for LSTM, pmdarima /
        statsmodels for ARIMA).
    background_data, test_data:
        LSTM-specific in-memory tensors / ndarrays / DataFrames.
    data:
        Convenience alias: for ARIMA or any explainer that only needs a
        single data stream, ``data=`` is treated as ``test_data``.

    Returns
    -------
    ExplanationResult | None
    """
    model_type = config.get("model_type")
    explainer_class = _load_explainer_class(model_type)

    if not explainer_class:
        raise ValueError(
            f"Model type '{model_type}' not found in timeseries analysis."
        )

    # Route `data` to `test_data` as a convenience alias.
    if test_data is None and data is not None:
        test_data = data

    # Build kwargs dynamically so each explainer only sees the ones it cares about.
    init_kwargs = {}
    if model is not None:
        init_kwargs["model"] = model
    if model_type == "lstm":
        if background_data is not None:
            init_kwargs["background_data"] = background_data
        if test_data is not None:
            init_kwargs["test_data"] = test_data

    explainer = explainer_class(config, **init_kwargs)

    # Standard workflow
    explainer.load_model()
    explainer.explain()

    if config.get("save_excel"):
        explainer.save_results_to_excel()

    if config.get("generate_notebook"):
        explainer.plot_results()

    if hasattr(explainer, "to_result"):
        return explainer.to_result()
    return None
