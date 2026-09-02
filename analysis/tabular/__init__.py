# analysis/tabular/__init__.py
#
# Lazy explainer loader. Each explainer class is imported on demand based on
# ``config["model_type"]`` so a tree-based run never imports torch/tensorflow
# and a neural run never pays for anything it doesn't need.

from importlib import import_module


# model_type -> (module path, class attribute). Resolved lazily.
_MODEL_ROUTES = {
    # Tree-based
    "random_forest": ("analysis.tabular.tree_based", "RFExplainer"),
    "xgboost": ("analysis.tabular.tree_based", "RFExplainer"),
    # Feedforward neural nets (PyTorch / TensorFlow)
    "feedforward": ("analysis.tabular.neural", "FeedForwardExplainer"),
    "mlp": ("analysis.tabular.neural", "FeedForwardExplainer"),
    "neural_net": ("analysis.tabular.neural", "FeedForwardExplainer"),
}


def _load_explainer_class(model_type):
    route = _MODEL_ROUTES.get(model_type)
    if route is None:
        return None
    module_path, attr = route
    return getattr(import_module(module_path), attr)


def run_tabular_analysis(config, *, model=None, data=None, background_data=None):
    """Orchestrates any tabular model (Tree, feedforward NN, etc.).

    Parameters
    ----------
    config:
        Legacy configuration dict — at minimum needs ``model_type``. All
        ``*_path`` entries become optional when the corresponding in-memory
        kwarg is supplied.
    model:
        Fitted model object. Takes precedence over ``config["model_path"]``.
    data:
        In-memory dataset (``pd.DataFrame`` / ``pyspark.sql.DataFrame`` /
        ``np.ndarray``). Takes precedence over ``config["dataset_path"]``.
    background_data:
        Optional SHAP baseline (used by the feedforward explainer's kernel /
        deep / gradient backends). Ignored by tree-based explainers.

    Returns
    -------
    ExplanationResult | None
    """
    model_type = config.get("model_type")
    explainer_class = _load_explainer_class(model_type)

    if not explainer_class:
        raise ValueError(
            f"Model {model_type} not supported in Tabular analysis. "
            f"Known types: {sorted(_MODEL_ROUTES)}."
        )

    # Only the neural explainer accepts a background_data kwarg. Build the
    # init kwargs dynamically so tree-based classes keep their simple signature.
    init_kwargs = {"model": model, "data": data}
    if background_data is not None and model_type in {"feedforward", "mlp", "neural_net"}:
        init_kwargs["background_data"] = background_data

    explainer = explainer_class(config, **init_kwargs)

    # Standard workflow
    explainer.load_model()
    explainer.explain()

    # Optional side-effects — only when the config explicitly asks, so a pure
    # in-memory caller never writes to disk.
    if config.get("save_excel"):
        explainer.save_results_to_excel()

    if config.get("generate_notebook"):
        explainer.plot_results()

    if hasattr(explainer, "to_result"):
        return explainer.to_result()
    return None


# Backwards-compatible alias — some callers imported this map directly. It maps
# model_type -> class, resolved lazily so importing this module stays cheap.
class _LazyModelMap:
    def __getitem__(self, key):
        cls = _load_explainer_class(key)
        if cls is None:
            raise KeyError(key)
        return cls

    def get(self, key, default=None):
        cls = _load_explainer_class(key)
        return cls if cls is not None else default

    def __contains__(self, key):
        return key in _MODEL_ROUTES

    def keys(self):
        return _MODEL_ROUTES.keys()


TABULAR_MAP = _LazyModelMap()
