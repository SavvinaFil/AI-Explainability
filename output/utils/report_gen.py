import nbformat as nbf
import os

from . import report_content as rc
import numpy as np
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _resolve_kernel_name():
    """Return a usable Jupyter kernel name, registering one if necessary.

    Hardcoding ``kernel_name='python3'`` fails in environments where no kernel
    spec is registered under that name (common with fresh installs or named
    conda envs). We prefer ``python3`` when present, fall back to any registered
    kernel, and as a last resort register the current interpreter. Returns
    ``None`` if no kernel can be made available (the notebook is then saved
    unexecuted rather than crashing).
    """
    try:
        from jupyter_client.kernelspec import KernelSpecManager

        specs = KernelSpecManager().find_kernel_specs()
    except Exception:
        specs = {}

    if "python3" in specs:
        return "python3"
    if specs:
        return next(iter(specs))

    # Nothing registered — best-effort: register the running interpreter.
    try:
        import subprocess

        subprocess.run(
            [sys.executable, "-m", "ipykernel", "install", "--user", "--name", "python3"],
            check=True,
            capture_output=True,
        )
        return "python3"
    except Exception:
        return None


def _execute_notebook_isolated(nb_path, timeout=600):
    """Execute a notebook in a SEPARATE process and update it in place.

    Running execution out-of-process (via the ``jupyter nbconvert`` CLI) means a
    native crash in the kernel/zmq stack — e.g. the libzmq access violation seen
    on some Windows setups — is contained to the child process instead of killing
    the caller. Returns ``True`` on success; on any failure the on-disk notebook
    is left unexecuted (but still runnable) and a short message is printed.
    """
    import subprocess

    kernel = _resolve_kernel_name()
    cmd = [
        sys.executable, "-m", "jupyter", "nbconvert",
        "--to", "notebook", "--execute", "--inplace", nb_path,
        f"--ExecutePreprocessor.timeout={timeout}",
    ]
    if kernel:
        cmd += ["--ExecutePreprocessor.kernel_name", kernel]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout + 120
        )
    except Exception as e:  # launch failure / timeout
        print(f"Warning: could not execute the notebook ({type(e).__name__}: {e}).")
        return False

    if proc.returncode == 0:
        print("Notebook executed (plots rendered).")
        return True

    # Non-zero / abnormal exit — includes a contained native crash.
    detail = ""
    if proc.stderr:
        lines = [ln for ln in proc.stderr.strip().splitlines() if ln.strip()]
        if lines:
            detail = f" Last line: {lines[-1][:160]}"
    print(
        "Warning: notebook execution failed in an isolated subprocess "
        f"(exit code {proc.returncode}).{detail}"
    )
    return False


def generate_notebook(explainer_inst, shap_values=None, raw_data=None, output_path=None, all_shap_values=None):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    nb = nbf.v4.new_notebook()

    # 1. Extract Metadata from Config
    config = explainer_inst.config
    model_type = config.get("model_type", "Unknown").upper()
    explainer_type = config.get("explainer_type", "SHAP")
    features = config.get("feature_names", [])
    look_back = config.get("look_back", 1)
    output_labels = config.get("output_labels", {})
    top_features = config.get("top_features", None)
    select_features = config.get("select_features", None)
    is_timeseries = (
        config.get("analysis") == "timeseries"
        and raw_data is not None
        and getattr(raw_data, "ndim", None) == 3
    )

    # 2. Dynamic Explainer Description
    if "gradient" in explainer_type.lower():
        explainer_desc = (
            "**Gradient SHAP** is designed for deep learning models. It explains predictions by "
            "computing the gradients of the output with respect to the inputs, integrated over "
            "various reference points (baselines). It is highly efficient for neural networks."
        )
    elif "kernel" in explainer_type.lower():
        explainer_desc = (
            "**Kernel SHAP** is a model-agnostic method that uses a specially weighted local linear "
            "regression to estimate SHAP values. It treats the model as a black box and is "
            "effective for any architecture, though it can be computationally intensive."
        )
    elif "tree" in explainer_type.lower():
        explainer_desc = (
            "**Tree SHAP** is an optimized algorithm for tree-based models (like Random Forest or XGBoost). "
            "It leverages the internal structure of the trees to calculate exact SHAP values "
            "significantly faster than model-agnostic methods."
        )
    else:
        explainer_desc = f"The **{explainer_type}** explainer was used to attribute feature importance."

    text_intro = f"""# SHAP Interpretation Report: {model_type}
This notebook provides a post-hoc explanation of the model's predictions using **{explainer_type}** SHAP.

---

### What are SHAP Values?
**SHAP (SHapley Additive exPlanations)** decomposes a model's prediction into the contribution of each individual feature.
* **Magnitude:** A larger absolute SHAP value means the feature had a bigger impact on the output.
* **Direction:** A positive SHAP value means the feature pushed the prediction *higher*, while a negative value pushed it *lower*.
* **Interpretation:** For any given sample, the sum of SHAP values plus the base value (average model output) equals the actual model prediction.

### Methodology
{explainer_desc}

---

### Metadata
**Model Architecture:** {model_type}
**Analysis Context:** {config.get("analysis", "General Interpretation")}
**Dataset Scope:** {config.get("dataset_scope", "N/A")}

---
"""
    cells = [nbf.v4.new_markdown_cell(text_intro)]

    # Windows asyncio fix — runs inside the kernel before anything else
    cells.append(nbf.v4.new_code_cell("""import asyncio, sys
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
"""))

    # 3. Save data to .npz file instead of embedding inline
    # This avoids MemoryError when datasets are large (e.g. 1500 samples x 6 features x 6 targets)
    if all_shap_values is None and shap_values is not None:
        all_shap_data = {0: shap_values}
    else:
        all_shap_data = {k: np.array(v) for k, v in all_shap_values.items()}

    # Use absolute path so the notebook always finds the .npz file regardless of working directory
    data_file = os.path.abspath(output_path.replace(".ipynb", "_data.npz"))

    np.savez(
        data_file,
        raw_data=raw_data,
        **{f"shap_{k}": v for k, v in all_shap_data.items()}
    )

    # Notebook loads data from .npz — no inline literals, no MemoryError
    code_setup = f"""import shap
import numpy as np
import matplotlib.pyplot as plt
# Shared plot helpers — the HTML report (save_report_html) uses these exact
# functions too, so both reports produce an identical layout.
from output.utils.report_content import fig_beeswarm, fig_bar, fig_temporal, fig_top5
from output.utils.report_content import select_feature_indices

# Load data from companion .npz file — avoids MemoryError for large datasets
_data = np.load(r"{data_file}", allow_pickle=True)
data_raw = _data["raw_data"]
all_shap_dict = {{
    int(k.replace("shap_", "")): _data[k]
    for k in _data.files
    if k.startswith("shap_")
}}

feature_names = {features}
look_back = {look_back}
is_timeseries = {is_timeseries}
output_labels = {output_labels}
top_features = {top_features!r}
select_features = {select_features!r}

def get_flattened_data(shap_data):
    if is_timeseries:
        shap_flat = shap_data.reshape(shap_data.shape[0], -1)
        data_flat = data_raw.reshape(data_raw.shape[0], -1)
        flat_names = [f"{{feat}}_t-{{look_back - 1 - i}}" for i in range(look_back) for feat in feature_names]
    else:
        shap_flat = shap_data
        data_flat = data_raw
        flat_names = list(feature_names)
    # Trim to selected / top features (per output) — keeps big outputs readable.
    sel = select_feature_indices(shap_flat, flat_names, top_features=top_features, select_features=select_features)
    if len(sel) != len(flat_names):
        shap_flat = shap_flat[:, sel]
        data_flat = data_flat[:, sel] if data_flat is not None else data_flat
        flat_names = [flat_names[i] for i in sel]
    return shap_flat, data_flat, flat_names

print(f"Setup complete. Targets to explain: {{list(all_shap_dict.keys())}}")
"""
    cells.append(nbf.v4.new_code_cell(code_setup))

    # 4. Loop Through Targets
    for target_idx in all_shap_data.keys():
        # Resolve correct label for each target — handles all 3 shapes
        if isinstance(output_labels, list):
            # Shape C — LSTM: ["PV"]
            label = output_labels[target_idx] if target_idx < len(output_labels) else f"Target {target_idx}"
        elif isinstance(output_labels, dict):
            # Shape B — multioutput classification: use _name key if available
            name_key = f"{target_idx}_name"
            if name_key in output_labels:
                label = output_labels[name_key]  # "Generator 1"
            else:
                raw = output_labels.get(str(target_idx), f"Target {target_idx}")
                # Shape A — regression: plain string e.g. "Power Forecast"
                # Shape B fallback: dict e.g. {"0":"OFF","1":"ON"} → use generic name
                label = raw if isinstance(raw, str) else f"Target {target_idx}"
        else:
            label = f"Target {target_idx}"

        # Section Headers
        cells.append(nbf.v4.new_markdown_cell(f"---\n# Analysis for: **{label}**"))
        cells.append(nbf.v4.new_markdown_cell(f"## Analysis for Target: `{label}`\n---"))

        # PLOT 1: BEESWARM
        cells.append(nbf.v4.new_markdown_cell(f"### {rc.BEESWARM_TITLE}\n\n{rc.BEESWARM_TEXT}"))

        cells.append(nbf.v4.new_code_cell(
            f"current_shap_raw = np.array(all_shap_dict[{repr(target_idx)}])\n"
            f"s_flat, d_flat, names = get_flattened_data(current_shap_raw)\n"
            f'fig_beeswarm(s_flat, d_flat, names, "Impact Distribution: {label}")\n'
            f"plt.show()"
        ))

        # PLOT 2: BAR CHART
        cells.append(nbf.v4.new_markdown_cell(f"### {rc.BAR_TITLE}\n\n{rc.BAR_TEXT}"))

        cells.append(nbf.v4.new_code_cell(
            f'fig_bar(s_flat, d_flat, names, "Mean Influence: {label}")\n'
            f"plt.show()"
        ))

        # PLOT 3: TEMPORAL (Optional — LSTM only)
        if is_timeseries:
            cells.append(nbf.v4.new_markdown_cell(f"### {rc.TEMPORAL_TITLE}\n\n{rc.TEMPORAL_TEXT}"))

            cells.append(nbf.v4.new_code_cell(
                f"current_shap_raw = np.array(all_shap_dict[{repr(target_idx)}])\n"
                f'fig_temporal(current_shap_raw, {look_back}, "Temporal Relevance: {label}")\n'
                f"plt.show()"
            ))

        # PLOT 4: TOP 5 DRIVERS
        cells.append(nbf.v4.new_markdown_cell(f"### {rc.TOP5_TITLE}\n\n{rc.TOP5_TEXT}"))

        cells.append(nbf.v4.new_code_cell(
            f'fig_top5(s_flat, names, "Top 5 Drivers: {label}")\n'
            f"plt.show()"
        ))

    # 5. Save first, then (optionally) execute OUT-OF-PROCESS.
    #
    # Writing the notebook before any execution guarantees the file exists even
    # if execution fails. Execution spins up a Jupyter kernel over pyzmq, which
    # on some platforms (notably Windows) can crash the *host* process with a
    # native fault (EXCEPTION_ACCESS_VIOLATION in libzmq) that Python cannot
    # catch. So execution is (a) opt-in via config['execute_notebook'] and
    # (b) run in an isolated subprocess, so such a crash is contained and the
    # unexecuted — but perfectly runnable — notebook is still delivered.
    nb['cells'] = cells
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

    if explainer_inst.config.get("execute_notebook", False):
        ok = _execute_notebook_isolated(output_path)
        if not ok:
            print(
                "  (The notebook was saved WITHOUT rendered plots — open it in "
                "Jupyter and 'Run All' to render them.)"
            )
    else:
        print(
            "Notebook saved (code cells, not executed). Open it and 'Run All' to "
            "render plots, or pass execute_notebook=True to pre-render."
        )

    print(f"Notebook generated: {output_path}")