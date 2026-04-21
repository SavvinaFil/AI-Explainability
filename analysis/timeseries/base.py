from abc import ABC, abstractmethod
import os

import pandas as pd


class TimeseriesExplainerBase(ABC):
    """Base class for time-series explainers.

    Like the tabular base, accepts either a config with file paths *or*
    in-memory objects via kwargs (``model``, ``background_data``,
    ``test_data``). Subclasses fall back to loading from ``config`` when
    the relevant kwarg is ``None``.
    """

    def __init__(
        self,
        config,
        *,
        model=None,
        background_data=None,
        test_data=None,
    ):
        self.config = config
        self.model = model
        self.data = None
        # In-memory tensors / arrays supplied by the caller. When None the
        # subclass falls back to the config's ``*_path`` entries.
        self.background_data = background_data
        self.test_data = test_data
        self.base_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )

    def get_path(self, key):
        """Resolve a config path against the project root.

        Returns ``None`` if the key is missing or empty — the caller is
        expected to interpret that as "use the in-memory object passed in
        via kwargs instead".
        """
        relative_path = self.config.get(key)
        if not relative_path:
            return None
        return os.path.join(self.base_path, relative_path)

    def ensure_output_dir(self):
        """Create ``output_dir`` lazily — only when a caller actually writes."""
        output_dir = self.config.get("output_dir")
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def explain(self):
        pass

    def save_results(self, results: pd.DataFrame):
        """Shared logic: all explainers save results the same way."""
        output_path = self.config.get("output_dir", "outputs")
        os.makedirs(output_path, exist_ok=True)
        results.to_csv(f"{output_path}/explanation_results.csv")
        print(f"Results saved to {output_path}")

    @abstractmethod
    def plot_results(self):
        pass
