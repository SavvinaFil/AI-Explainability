"""
Input coercion helpers — the glue that lets the library accept either a
file path or an already-loaded Python object.

None of the heavy backends (``torch``, ``joblib``, ``pyspark``) are imported
at module load time; they are only touched inside the function that needs
them. That keeps ``import ai_explainability`` cheap and avoids hard
dependencies on optional backends.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Model coercion                                                              #
# --------------------------------------------------------------------------- #
def to_fitted_model(obj: Any, *, hint: str | None = None) -> Any:
    """Return a fitted model object regardless of whether ``obj`` is a path or an object.

    Parameters
    ----------
    obj:
        Either a fitted model instance (anything that is not a ``str`` /
        ``os.PathLike``) or a path to a serialised model on disk.
    hint:
        Optional ``"sklearn"`` / ``"xgboost"`` / ``"pytorch"`` / ``"arima"``
        hint that disambiguates the loader for file paths. When ``obj`` is
        already an in-memory object ``hint`` is ignored.

    Notes
    -----
    The file-path branch picks a loader from the extension:

    - ``.pkl``, ``.pickle``                → :func:`pickle.load`
    - ``.joblib``                           → :func:`joblib.load`
    - ``.pt``, ``.pth``, ``.bin``          → :func:`torch.load` (state_dict
      or full module — caller decides)
    - Anything else                        → :func:`pickle.load`

    Users with custom serialisation formats should load the model themselves
    and pass the in-memory object in, which is the whole point of this API.
    """
    if obj is None:
        raise ValueError("model must not be None")

    # Already in-memory — nothing to do.
    if not isinstance(obj, (str, os.PathLike)):
        return obj

    path = Path(obj)
    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix in (".pt", ".pth", ".bin"):
        # Torch serialisation — state_dict or full module. Callers that rely on
        # this branch (LSTMExplainer) reconstruct the architecture manually.
        import torch

        return torch.load(path, map_location=torch.device("cpu"))

    if suffix == ".joblib":
        import joblib

        return joblib.load(path)

    # Default: pickle. Covers .pkl, .pickle, and unknown suffixes.
    with open(path, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------- #
# DataFrame / array coercion                                                  #
# --------------------------------------------------------------------------- #
def _looks_like_spark_dataframe(obj: Any) -> bool:
    """Duck-type check for ``pyspark.sql.DataFrame`` without importing pyspark.

    We only need to recognise it; we don't need to hold a reference to the
    ``pyspark`` module itself. Checking the class-chain names keeps this
    zero-dependency.
    """
    if isinstance(obj, pd.DataFrame):
        return False
    cls = type(obj)
    for klass in cls.__mro__:
        qual = f"{klass.__module__}.{klass.__name__}"
        if qual in {"pyspark.sql.dataframe.DataFrame", "pyspark.sql.connect.dataframe.DataFrame"}:
            return True
    return hasattr(obj, "toPandas") and hasattr(obj, "schema") and hasattr(obj, "rdd")


def to_pandas(
    data: Any,
    *,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Coerce ``data`` to a :class:`pandas.DataFrame`.

    Accepts:

    - :class:`pandas.DataFrame`                              → pass-through
    - :class:`pyspark.sql.DataFrame` (duck-typed)            → ``.toPandas()``
    - :class:`numpy.ndarray`                                 → wrap with ``feature_names``
    - ``str`` / ``os.PathLike`` ending in ``.csv`` / ``.tsv`` / ``.parquet``
                                                             → read from disk
    - Anything with ``to_pandas()`` (e.g. polars, pyarrow)   → best-effort

    Raises
    ------
    TypeError
        If ``data`` cannot be coerced by any of the above strategies.
    """
    if data is None:
        raise ValueError("data must not be None")

    if isinstance(data, pd.DataFrame):
        return data

    if _looks_like_spark_dataframe(data):
        # pyspark.sql.DataFrame.toPandas() blocks until the Spark job finishes.
        # That's fine for explainability, where the slice is small.
        return data.toPandas()

    if isinstance(data, np.ndarray):
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        if data.ndim != 2:
            raise TypeError(
                f"ndarray data must be 1D or 2D for tabular analysis; got {data.ndim}D"
            )
        cols = list(feature_names) if feature_names is not None else [
            f"f{i}" for i in range(data.shape[1])
        ]
        return pd.DataFrame(data, columns=cols)

    if isinstance(data, (str, os.PathLike)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Data path does not exist: {path}")
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            return pd.read_csv(path, sep=sep)
        if suffix == ".parquet":
            return pd.read_parquet(path)
        # Fallback — assume CSV.
        return pd.read_csv(path)

    # Last resort: libraries that follow the Arrow / polars convention.
    if hasattr(data, "to_pandas"):
        return data.to_pandas()

    raise TypeError(
        f"Cannot coerce object of type {type(data).__name__} to pandas.DataFrame. "
        "Pass a DataFrame, a numpy array, a pyspark.sql.DataFrame, or a file path."
    )


# --------------------------------------------------------------------------- #
# Tensor coercion (used by the LSTM explainer)                                #
# --------------------------------------------------------------------------- #
def to_torch_tensor(obj: Any) -> Any:
    """Coerce ``obj`` to a ``torch.Tensor``.

    Accepts a tensor (pass-through), a numpy array, a pandas DataFrame (uses
    ``.values``), or a path to a ``torch.save`` artefact. ``torch`` is only
    imported on demand.
    """
    if obj is None:
        raise ValueError("tensor input must not be None")

    import torch  # local import — don't pay the torch cost at module load

    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()

    if isinstance(obj, pd.DataFrame):
        return torch.from_numpy(obj.values).float()

    if isinstance(obj, (str, os.PathLike)):
        path = Path(obj)
        if not path.exists():
            raise FileNotFoundError(f"Tensor path does not exist: {path}")
        return torch.load(path, map_location=torch.device("cpu"))

    raise TypeError(
        f"Cannot coerce object of type {type(obj).__name__} to torch.Tensor."
    )
