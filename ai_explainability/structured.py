"""
Structured (dict) observations.

RL observations are often a mapping of arrays, e.g.
``{"bus": (n, 32, 2), "line": (n, 40, 3), "mask": (n, 32)}``. SHAP needs one flat
2D matrix, so :class:`DictLayout` flattens the mapping in a documented order and
rebuilds it before the model is called:

- keys are taken in the mapping's own order;
- each array is flattened in C order (last axis fastest);
- feature names follow the array shape: ``key`` for a scalar per sample,
  ``key[i]`` for a vector, ``key[r][c]`` for a table. With
  ``field_names={"bus": ["vm", "va"]}`` the last axis is named instead:
  ``bus[3].vm``.

Each top-level key becomes a default feature group.
"""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import numpy as np


class DictLayout:
    """Flattening order and names for a mapping of ``(n, *shape)`` arrays."""

    def __init__(self, sample: Mapping[str, Any], field_names: Mapping[str, Sequence[str]] | None = None):
        if not sample:
            raise ValueError("Dict input is empty.")
        field_names = dict(field_names or {})
        self.keys: list[str] = []
        self.shapes: dict[str, tuple[int, ...]] = {}
        self.slices: dict[str, slice] = {}
        self.feature_names: list[str] = []
        n = None
        start = 0
        for key, arr in sample.items():
            arr = _to_numpy(arr)
            if arr.ndim == 0:
                raise ValueError(f"Dict input {key!r} needs a leading sample axis.")
            if n is None:
                n = arr.shape[0]
            elif arr.shape[0] != n:
                raise ValueError(
                    f"Dict input {key!r} has {arr.shape[0]} samples, expected {n}."
                )
            key = str(key)
            shape = tuple(arr.shape[1:])
            size = int(np.prod(shape)) if shape else 1
            self.keys.append(key)
            self.shapes[key] = shape
            self.slices[key] = slice(start, start + size)
            self.feature_names.extend(_names_for(key, shape, field_names.get(key)))
            start += size
        self.n_features = start

    # ------------------------------------------------------------------ #
    @property
    def groups(self) -> dict[str, list[str]]:
        """Default feature groups: one per top-level key."""
        return {k: self.feature_names[self.slices[k]] for k in self.keys}

    def flatten(self, data: Mapping[str, Any]) -> np.ndarray:
        """Mapping of ``(n, *shape)`` arrays -> ``(n, n_features)`` float32 matrix."""
        missing = [k for k in self.keys if k not in data]
        if missing:
            raise ValueError(f"Dict input is missing keys {missing}.")
        parts = []
        for k in self.keys:
            arr = _to_numpy(data[k]).astype("float32")
            if tuple(arr.shape[1:]) != self.shapes[k]:
                raise ValueError(
                    f"Dict input {k!r} has shape {arr.shape[1:]}, expected {self.shapes[k]}."
                )
            parts.append(arr.reshape(len(arr), -1))
        return np.concatenate(parts, axis=1)

    def unflatten(self, flat):
        """``(n, n_features)`` numpy array or torch tensor -> mapping of the original shapes."""
        n = flat.shape[0]
        return {k: flat[:, self.slices[k]].reshape((n,) + self.shapes[k]) for k in self.keys}

    def wrap_model(self, model, framework: str):
        """Return a model that takes the flat matrix and calls ``model`` with the dict."""
        layout = self
        if framework == "pytorch":
            import torch

            class _DictInput(torch.nn.Module):
                def __init__(self, base):
                    super().__init__()
                    self.base = base

                def forward(self, x):
                    return self.base(layout.unflatten(x))

            return _DictInput(model).eval()
        if framework == "callable":
            return lambda x: model(layout.unflatten(np.asarray(x, dtype="float32")))
        raise ValueError(
            "Dict inputs support PyTorch modules and plain callables. For a Keras "
            "model, pass a callable that builds the dict and calls it (kernel only)."
        )


def _to_numpy(arr) -> np.ndarray:
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    return np.asarray(arr)


def _names_for(key: str, shape: tuple[int, ...], fields: Sequence[str] | None) -> list[str]:
    if not shape:
        return [key]
    if fields is not None:
        if len(fields) != shape[-1]:
            raise ValueError(
                f"field_names[{key!r}] has {len(fields)} names but the last axis has "
                f"{shape[-1]} entries."
            )
        lead = shape[:-1]
        if not lead:
            return [f"{key}.{f}" for f in fields]
        return [
            f"{key}{''.join(f'[{i}]' for i in idx)}.{f}"
            for idx in itertools.product(*(range(s) for s in lead))
            for f in fields
        ]
    return [
        f"{key}{''.join(f'[{i}]' for i in idx)}"
        for idx in itertools.product(*(range(s) for s in shape))
    ]
