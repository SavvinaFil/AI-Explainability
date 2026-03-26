"""
gRPC Server - SHAP Explainability Service

The ExplainRequest mirrors exactly the config.json structure.
The server reconstructs the config dict from the request and
runs the existing SHAP pipeline unchanged.

How to run/use through terminal:
  python server.py
  python server.py --port 50052
"""

import os
import tempfile
import argparse
import logging
import traceback
from concurrent import futures

import grpc
import numpy as np

import AI_explainability_pb2      as pb2
import AI_explainability_pb2_grpc as pb2_grpc

from analysis import ANALYSIS_ROUTER

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)


# Reconstruct config.json from proto request

def _build_config(request: pb2.ExplainRequest, tmp_dir: str) -> dict:
    """
    Saves uploaded bytes to temp files and builds a config dict
    identical in shape to the JSON configs the pipeline already uses.

    dataset_scope, save_excel, generate_notebook are always fixed:
      - dataset_scope     = "whole"
      - save_excel        = True
      - generate_notebook = True
    """

    # Save model file
    model_ext  = ".pth" if request.package == "pytorch" else ".pkl"
    model_path = os.path.join(tmp_dir, f"model{model_ext}")
    with open(model_path, "wb") as f:
        f.write(request.model_file)

    # Save dataset file
    dataset_ext  = ".pt" if request.package == "pytorch" else ".csv"
    dataset_path = os.path.join(tmp_dir, f"dataset{dataset_ext}")
    with open(dataset_path, "wb") as f:
        f.write(request.dataset_file)

    # Save background data (LSTM only)
    background_path = ""
    if request.background_data_file:
        background_path = os.path.join(tmp_dir, "background_data.pt")
        with open(background_path, "wb") as f:
            f.write(request.background_data_file)

    # Save test data (LSTM only)
    test_path = ""
    if request.test_data_file:
        test_path = os.path.join(tmp_dir, "test_data.pt")
        with open(test_path, "wb") as f:
            f.write(request.test_data_file)

    # Reconstruct output_labels
    # Shape A - regression / binary:   {"0": "Power Forecast", ...}
    # Shape B - multioutput classify:  {"0": {"0":"OFF","1":"ON"}, "0_name":"Generator 1"}
    # Shape C - LSTM timeseries:       ["PV"]

    if request.analysis == "timeseries":
        output_labels = [lbl.label for lbl in request.output_labels]
    else:
        output_labels = {}
        for lbl in request.output_labels:
            if lbl.label_1:
                output_labels[lbl.index] = {
                    "0": lbl.label,
                    "1": lbl.label_1,
                }
                if lbl.name:
                    output_labels[f"{lbl.index}_name"] = lbl.name
            else:
                output_labels[lbl.index] = lbl.label

    target_index = [int(lbl.index) for lbl in request.output_labels
                    if lbl.index.isdigit()]

# Takes data from proto request and transform them into config dict - exactly as to read
# config.json immediately

    config = {
        # Routing
        "analysis":       request.analysis,
        "package":        request.package,
        "model_type":     request.model_type,
        "explainer_type": request.explainer_type or "tree",

        # Paths
        "model_path":           model_path,
        "dataset_path":         dataset_path,
        "background_data_path": background_path,
        "test_data_path":       test_path,
        "output_dir":           tmp_dir,

        # Features / targets
        "feature_names": list(request.feature_names),
        "output_labels": output_labels,
        "target_index":  target_index,

        # Fixed values - NOT sent by client
        "dataset_scope":     "whole",
        "subset_end":        0,
        "save_excel":        True,
        "generate_notebook": True,

        # Timeseries only
        "input_dim":   request.input_dim   if request.input_dim   else None,
        "hidden_size": request.hidden_size if request.hidden_size else None,
        "look_back":   request.look_back   if request.look_back   else None,
        "look_ahead":  request.look_ahead  if request.look_ahead  else None,
    }
    return config


# Collect ShapRows

def _build_shap_rows(explainer, config: dict) -> list:
    rows          = []
    output_labels = config.get("output_labels", {})
    analysis_type = config.get("analysis", "tabular")
    # this config is NOT config.json from our code before
    # is the dict which _build_config() made
    # from the data that client provides through .proto

    for target_idx, shap_arr in explainer.all_shap_values.items():

        # GradientExplainer returns an Explanation object, not a plain array
        if hasattr(shap_arr, 'values'):
            shap_arr = shap_arr.values

        idx_str = str(target_idx)

        if isinstance(output_labels, list):
            t_name = output_labels[target_idx] if target_idx < len(output_labels) \
                     else f"Output_{target_idx}"
        else:
            name_key = f"{idx_str}_name"
            if name_key in output_labels:
                # Shape B — multioutput classification
                t_name = output_labels[name_key]
            elif isinstance(output_labels.get(idx_str), dict):
                # Shape B fallback
                t_name = output_labels.get(name_key, f"Output_{target_idx}")
            else:
                # Shape A — regression
                t_name = str(output_labels.get(idx_str, f"Output_{target_idx}"))

        # Safety: never allow empty name
        if not t_name or not t_name.strip():
            t_name = f"Output_{target_idx}"

        # ── Get raw data values ──────────────────────────────────────────────
        # Tabular: explainer.raw_data.values  (2D: N x features)
        # LSTM:    explainer.raw_data_values  (3D: N x lookback x features)
        if analysis_type == "timeseries":
            raw_vals = explainer.raw_data_values
        else:
            raw_vals = explainer.raw_data.values

        # Get predictions
        is_classification = False
        if analysis_type == "timeseries":
            import torch
            explainer.model.eval()
            with torch.no_grad():
                test_tensor = torch.tensor(raw_vals).float()
                # .tolist() converts to plain Python floats — no numpy scalars
                preds_col = explainer.model(test_tensor).numpy().flatten().tolist()
        else:
            is_classification = hasattr(explainer.model, "predict_proba")
            preds     = explainer.model.predict(explainer.raw_data)
            preds_col = preds[:, target_idx] if (hasattr(preds, "ndim") and preds.ndim > 1) \
                        else preds

        # Build ShapRows
        # Always flatten per sample - handles both 2D and 3D shap arrays
        for i in range(len(shap_arr)):
            shap_flat    = [float(x) for x in shap_arr[i].flatten()]
            feature_flat = [float(x) for x in raw_vals[i].flatten()]

            row = pb2.ShapRow(
                target_index   = target_idx,
                target_name    = t_name,
                sample_index   = i,
                feature_values = feature_flat,
                shap_values    = shap_flat,
            )
            if is_classification:
                row.prediction = int(preds_col[i])
            else:
                row.prediction_cont = float(preds_col[i])
            rows.append(row)

    return rows


# Read output files

def _read_output_files(tmp_dir: str):
    excel_bytes    = b""
    notebook_bytes = b""
    for fname in os.listdir(tmp_dir):
        fpath = os.path.join(tmp_dir, fname)
        if fname.endswith(".xlsx") and not excel_bytes:
            with open(fpath, "rb") as f:
                excel_bytes = f.read()
        elif fname.endswith(".ipynb") and not notebook_bytes:
            with open(fpath, "rb") as f:
                notebook_bytes = f.read()
    return excel_bytes, notebook_bytes


# Service

class ShapExplainerServicer(pb2_grpc.ShapExplainerServiceServicer):

    def HealthCheck(self, request, context):
        log.info("HealthCheck called")
        return pb2.HealthResponse(status="OK")

    def Explain(self, request, context):
        log.info(
            "Explain request | analysis=%s package=%s model_type=%s features=%d",
            request.analysis, request.package,
            request.model_type, len(request.feature_names),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            try:
                # 1. Build config from request
                config = _build_config(request, tmp_dir)

                # 2. Get the right explainer class
                analysis_type = config["analysis"]
                model_type    = config["model_type"]

                from analysis.tabular import TABULAR_MAP
                from analysis.timeseries import MODEL_MAP as TS_MAP

                if analysis_type == "tabular":
                    ExplainerClass = TABULAR_MAP.get(model_type)
                else:
                    ExplainerClass = TS_MAP.get(model_type)

                if ExplainerClass is None:
                    raise ValueError(
                        f"No explainer for analysis='{analysis_type}' "
                        f"model_type='{model_type}'"
                    )

                # 3. Run the pipeline (identical to main.py)
                explainer = ExplainerClass(config)
                explainer.load_model()
                explainer.explain()
                explainer.save_results_to_excel()
                explainer.plot_results()

                # 4. Collect results
                shap_rows = _build_shap_rows(explainer, config)
                excel_bytes, notebook_bytes = _read_output_files(tmp_dir)

                log.info("Explain completed — %d ShapRows", len(shap_rows))
                return pb2.ExplainResponse(
                    success       = True,
                    message       = f"OK — {len(shap_rows)} rows computed",
                    shap_rows     = shap_rows,
                    excel_file    = excel_bytes,
                    notebook_file = notebook_bytes,
                )

            except Exception as exc:
                log.error("Explain failed:\n%s", traceback.format_exc())
                context.set_code(grpc.StatusCode.INTERNAL)
                context.set_details(str(exc))
                return pb2.ExplainResponse(success=False, message=str(exc))


# Entry point

def serve(port: int = 50051, max_workers: int = 4):
    # max_workers up to 4 requests from 4 different clients
    options = [
        ("grpc.max_send_message_length",    256 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers), options=options
    )
    pb2_grpc.add_ShapExplainerServiceServicer_to_server(
        ShapExplainerServicer(), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    log.info("Server started on port %d (max_workers=%d)", port, max_workers)
    server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",        type=int, default=50051)
    parser.add_argument("--max_workers", type=int, default=4)
    args = parser.parse_args()
    serve(port=args.port, max_workers=args.max_workers)