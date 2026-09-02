# analysis/tabular/neural/__init__.py
#
# Feedforward / MLP explainer registry. Imports are kept lazy at the parent
# level (analysis/tabular/__init__.py) so torch / tensorflow are only pulled in
# when a neural model_type is actually requested.

from .ff_explainer import FeedForwardExplainer

# model_type strings that route to the feedforward explainer.
NN_MODEL_MAP = {
    "feedforward": FeedForwardExplainer,
    "mlp": FeedForwardExplainer,
    "neural_net": FeedForwardExplainer,
}
