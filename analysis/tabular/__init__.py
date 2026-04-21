# analysis/tabular/__init__.py
from .tree_based import TREE_MODEL_MAP

# Manager Map: Combines all tabular subtypes
TABULAR_MAP = {
    **TREE_MODEL_MAP,
    # "linear_regression": LinearExplainer, (Future addition)
}


def run_tabular_analysis(config, *, model=None, data=None):
    """Orchestrates any tabular model (Tree, Linear, etc.).

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

    Returns
    -------
    ExplanationResult | None
        An :class:`ai_explainability.ExplanationResult` when the explainer
        exposes ``to_result()`` (all current explainers do). Returns ``None``
        only for legacy explainer classes that haven't been refactored yet.
    """
    model_type = config.get("model_type")
    explainer_class = TABULAR_MAP.get(model_type)

    if not explainer_class:
        raise ValueError(f"Model {model_type} not supported in Tabular analysis.")

    # Instantiate with whichever inputs the caller has on hand.
    explainer = explainer_class(config, model=model, data=data)

    # Standard workflow
    explainer.load_model()
    explainer.explain()

    # Optional side-effects — only run when the config explicitly asks for
    # them, so a pure in-memory caller never writes to disk.
    if config.get("save_excel"):
        explainer.save_results_to_excel()

    if config.get("generate_notebook"):
        explainer.plot_results()

    # Return the structured result if the explainer provides one.
    if hasattr(explainer, "to_result"):
        return explainer.to_result()
    return None
