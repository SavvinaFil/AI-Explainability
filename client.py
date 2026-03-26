"""
gRPC Client - SHAP Explainability Service

Reads an existing config.json + model + dataset files,
translates them into a proto ExplainRequest, and sends to server.

NOTE: dataset_scope, save_excel, generate_notebook are NOT sent through client but
they are always fixed on the server side (whole / True / True).

Usage examples

  # Tabular trees
  python client.py
      --config  examples/tabular/multioutput_classify/config.json
      --model   source/models/multioutput_classify.pkl
      --dataset source/data/multioutput_classify.csv

  # LSTM timeseries
  python client.py
      --config     examples/timeseries/lstm/config.json
      --model      source/models/lstm_model.pth
      --dataset    source/data/lstm_background_data.pt
      --background source/data/lstm_background_data.pt
      --test_data  source/data/lstm_data_to_explain.pt
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import grpc

import AI_explainability_pb2      as pb2
import AI_explainability_pb2_grpc as pb2_grpc

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CLIENT] %(levelname)s — %(message)s",
)
log = logging.getLogger(__name__)


# Build OutputLabel list from config output_labels

def _parse_output_labels(config: dict) -> list:
    """
    Converts config.json output_labels into OutputLabel proto messages.

    Shape A - regression / binary classification:
        {"0": "Power Forecast", "1": "Load Forecast"}
        → OutputLabel(index="0", label="Power Forecast")

    Shape B - multioutput classification (ON/OFF):
        {"0": {"0": "OFF", "1": "ON"}, "0_name": "Generator 1"}
        → OutputLabel(index="0", label="OFF", label_1="ON", name="Generator 1")

    Shape C - LSTM timeseries:
        ["PV"]
        → OutputLabel(index="0", label="PV")
    """
    raw    = config.get("output_labels", {})
    result = []

    # Shape C - list (LSTM)
    if isinstance(raw, list):
        for i, label in enumerate(raw):
            result.append(pb2.OutputLabel(
                index = str(i),
                label = str(label),
            ))
        return result

    # Shape A & B - dict, collect only numeric keys
    indices = sorted([k for k in raw.keys() if k.isdigit()], key=int)

    for idx_str in indices:
        value    = raw[idx_str]
        name_key = f"{idx_str}_name"

        if isinstance(value, dict):
            # Shape B - multioutput classification
            result.append(pb2.OutputLabel(
                index   = idx_str,
                label   = value.get("0", ""),
                label_1 = value.get("1", ""),
                name    = raw.get(name_key, ""),
            ))
        else:
            # Shape A - regression or simple label
            result.append(pb2.OutputLabel(
                index = idx_str,
                label = str(value),
                name  = raw.get(name_key, ""),
            ))

    return result


# Core function

def run_explain(
    config_path:     str,
    model_path:      str,
    dataset_path:    str,
    background_path: str = "",
    test_path:       str = "",
    host:            str = "localhost",
    port:            int = 50051,
):
    # 1. Load config.json
    with open(config_path) as f:
        config = json.load(f)

    output_dir = config.get("output_dir", "output")
    feature_names = config.get("feature_names", [])
    look_back     = config.get("look_back") or None

    # 2. Read files as bytes
    log.info("Reading model   : %s", model_path)
    log.info("Reading dataset : %s", dataset_path)
    with open(model_path,   "rb") as f: model_bytes   = f.read()
    with open(dataset_path, "rb") as f: dataset_bytes = f.read()

    background_bytes = b""
    if background_path and os.path.exists(background_path):
        log.info("Reading background : %s", background_path)
        with open(background_path, "rb") as f: background_bytes = f.read()

    test_bytes = b""
    if test_path and os.path.exists(test_path):
        log.info("Reading test data  : %s", test_path)
        with open(test_path, "rb") as f: test_bytes = f.read()

    # 3. Build ExplainRequest
    # NOTE: dataset_scope, save_excel, generate_notebook are NOT included —
    # they are always fixed on the server (whole / True / True)
    request = pb2.ExplainRequest(
        # Routing (from config.json)
        analysis   = config.get("analysis",   "tabular"),
        package    = config.get("package",    "sklearn"),
        model_type = config.get("model_type", "random_forest"),

        # Artifacts
        model_file           = model_bytes,
        dataset_file         = dataset_bytes,
        background_data_file = background_bytes,
        test_data_file       = test_bytes,

        # Feature and output metadata (from config.json)
        feature_names = feature_names,
        output_labels = _parse_output_labels(config),

        # Timeseries-specific (from config.json)
        explainer_type = config.get("explainer_type", "") or "",
        input_dim      = config.get("input_dim")   or 0,
        hidden_size    = config.get("hidden_size") or 0,
        look_back      = config.get("look_back")   or 0,
        look_ahead     = config.get("look_ahead")  or 0,
    )

    # 4. Connect and call
    options = [
        ("grpc.max_send_message_length",    256 * 1024 * 1024),
        ("grpc.max_receive_message_length", 256 * 1024 * 1024),
    ]
    address = f"{host}:{port}"
    log.info("Connecting to %s …", address)

    with grpc.insecure_channel(address, options=options) as channel:
        stub = pb2_grpc.ShapExplainerServiceStub(channel)

        try:
            health = stub.HealthCheck(pb2.HealthRequest(), timeout=5)
            log.info("Server health: %s", health.status)
        except grpc.RpcError:
            log.warning("HealthCheck timed out — continuing anyway")

        log.info("Sending Explain request …")
        response: pb2.ExplainResponse = stub.Explain(request, timeout=300)

    # 5. Handle response
    if not response.success:
        log.error("Server error: %s", response.message)
        sys.exit(1)

    log.info("Success: %s", response.message)
    log.info("Received %d ShapRows", len(response.shap_rows))

    # 6. Print summary with correct feature names
    _print_summary(response, feature_names=feature_names, look_back=look_back)

    # 7. Save artifacts
    os.makedirs(output_dir, exist_ok=True)

    if response.excel_file:
        path = os.path.join(output_dir, "shap_results.xlsx")
        with open(path, "wb") as f: f.write(response.excel_file)
        log.info("Excel saved → %s", path)

    if response.notebook_file:
        path = os.path.join(output_dir, "shap_report.ipynb")
        with open(path, "wb") as f: f.write(response.notebook_file)
        log.info("Notebook saved → %s", path)

    return response


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SHAP Explainability gRPC Client")
    parser.add_argument("--config",     required=True, help="Path to config.json")
    parser.add_argument("--model",      required=True, help="Model file (.pkl or .pth)")
    parser.add_argument("--dataset",    required=True, help="Dataset file (.csv or .pt)")
    parser.add_argument("--background", default="",    help="Background data (.pt) — LSTM only")
    parser.add_argument("--test_data",  default="",    help="Test data (.pt) — LSTM only")
    parser.add_argument("--host",       default="localhost")
    parser.add_argument("--port",       type=int, default=50051)

    args = parser.parse_args()

    run_explain(
        config_path     = args.config,
        model_path      = args.model,
        dataset_path    = args.dataset,
        background_path = args.background,
        test_path       = args.test_data,
        host            = args.host,
        port            = args.port,
    )