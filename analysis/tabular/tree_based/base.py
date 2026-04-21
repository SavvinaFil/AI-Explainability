from abc import ABC, abstractmethod
import os


class ExplainerBase(ABC):
    """Base class for tabular explainers.

    Accepts either a file-path driven config (legacy behaviour) *or* an
    in-memory ``model`` and ``data`` passed via kwargs. Disk I/O (creating
    ``output_dir``) is deferred until a method that actually writes to disk
    is called, so a purely in-memory Databricks workflow touches the
    filesystem zero times.
    """

    def __init__(self, config, *, model=None, data=None):
        self.config = config
        self.model = model
        self.shap_values = None
        # ``raw_data`` is the pandas view used for feature naming / display.
        # The caller can pass a DataFrame, ndarray, pyspark.sql.DataFrame, etc.
        # Coercion happens inside ``explain()`` so subclass hooks stay simple.
        self.raw_data = data
        self.feature_names = config.get("feature_names", [])
        self.output_dir = config.get("output_dir", "output/")

    # ------------------------------------------------------------------ #
    # Path / IO helpers                                                  #
    # ------------------------------------------------------------------ #
    def get_path(self, key):
        return self.config.get(key)

    def ensure_output_dir(self):
        """Create ``self.output_dir`` if and only if disk output is requested.

        Callers that want a fully in-memory run never invoke this, which means
        ``os.makedirs`` is never executed.
        """
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Abstract API                                                       #
    # ------------------------------------------------------------------ #
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def explain(self):
        pass

    @abstractmethod
    def plot_results(self):
        pass
