import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
import os
import numpy as np
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def generate_notebook(explainer_inst, shap_values=None, raw_data=None, output_path=None, all_shap_values=None):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    nb = nbf.v4.new_notebook()
    
    # --- 1. Extract Metadata from Config ---
    config = explainer_inst.config
    model_type = config.get("model_type", "Unknown").upper()
    explainer_type = config.get("explainer_type", "SHAP")
    features = config.get("feature_names", [])
    look_back = config.get("look_back", 1)
    output_labels = config.get("output_labels", {})
    is_timeseries = config.get("analysis") == "timeseries" and raw_data.ndim == 3

    # --- 2. Dynamic Explainer Description ---
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

    text_intro = f"""# 📊 SHAP Interpretation Report: {model_type}
This notebook provides a post-hoc explanation of the model's predictions using **{explainer_type}** SHAP.

---

###  What are SHAP Values?
**SHAP (SHapley Additive exPlanations)** decomposes a model's prediction into the contribution of each individual feature. 
* **Magnitude:** A larger absolute SHAP value means the feature had a bigger impact on the output.
* **Direction:** A positive SHAP value means the feature pushed the prediction *higher*, while a negative value pushed it *lower*.
* **Interpretation:** For any given sample, the sum of SHAP values plus the base value (average model output) equals the actual model prediction.

###  Methodology
{explainer_desc}

---

###  Metadata
**Model Architecture:** {model_type}  
**Analysis Context:** {config.get("analysis", "General Interpretation")}  
**Dataset Scope:** {config.get("dataset_scope", "N/A")}

---
"""
    # !!!cells = [nbf.v4.new_markdown_cell(text_intro)]
    cells = [nbf.v4.new_markdown_cell(text_intro)]
    cells.append(nbf.v4.new_code_cell("""
    import asyncio, sys
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    """))

    # --- 3. Setup Code (Handles Single or Multi-Target) ---
    # If all_shap_values is provided, we use it. Otherwise, we wrap shap_values in a dict.
    if all_shap_values is None and shap_values is not None:
        # Backward compatibility: treat single output as index 0
        all_shap_data = {0: shap_values.tolist()}
    else:
        all_shap_data = {k: v.tolist() for k, v in all_shap_values.items()}

    code_setup = f"""import shap
import numpy as np
import matplotlib.pyplot as plt

# Data provided by the explainer
all_shap_dict = {all_shap_data}
data_raw = np.array({raw_data.tolist()})
feature_names = {features}
look_back = {look_back}
is_timeseries = {is_timeseries}
output_labels = {output_labels}

def get_flattened_data(shap_data):
    if is_timeseries:
        shap_flat = shap_data.reshape(shap_data.shape[0], -1)
        data_flat = data_raw.reshape(data_raw.shape[0], -1)
        flat_names = [f"{{feat}}_t-{{look_back - 1 - i}}" for i in range(look_back) for feat in feature_names]
    else:
        shap_flat = shap_data
        data_flat = data_raw
        flat_names = feature_names
    return shap_flat, data_flat, flat_names

print(f"Setup complete. Targets to explain: {{list(all_shap_dict.keys())}}")
"""
    cells.append(nbf.v4.new_code_cell(code_setup))

    # --- 4. Loop Through Targets ---
    for target_idx in all_shap_data.keys():
        # Handle different label formats (Dictionary for Tabular, List for TS)
        if isinstance(output_labels, dict):
            label = output_labels.get(str(target_idx), f"Target {target_idx}")
        elif isinstance(output_labels, list) and target_idx < len(output_labels):
            label = output_labels[target_idx]
        else:
            label = f"Target {target_idx}"
        
        # Section Header
        cells.append(nbf.v4.new_markdown_cell(f"--- \n# Analysis for: **{label}**"))

        # Section Header with "Target" highlight
        cells.append(nbf.v4.new_markdown_cell(f"## 🎯 Analysis for Target: `{label}`\n---"))

        # --- PLOT 1: BEESWARM ---
        cells.append(nbf.v4.new_markdown_cell(f"""### 1. Feature Impact Distribution (Beeswarm)
**What is this?** A distribution of SHAP values for every sample in the dataset.  
**What to focus on:** * **Horizontal Position:** Points to the right increase the model output; points to the left decrease it.
* **Color:** Represents the feature value (**Red** is high, **Blue** is low). 
* **Insight:** If Red points are on the right, the feature has a positive correlation with the target."""))
        
        cells.append(nbf.v4.new_code_cell(f"""
current_shap_raw = np.array(all_shap_dict[{repr(target_idx)}])
s_flat, d_flat, names = get_flattened_data(current_shap_raw)
plt.figure(figsize=(10, 6))
shap.summary_plot(s_flat, d_flat, feature_names=names, show=False)
plt.title(f"Impact Distribution: {label}", fontsize=14, pad=20)
plt.show()"""))

        # --- PLOT 2: BAR CHART ---
        cells.append(nbf.v4.new_markdown_cell(f"""### 2. Global Feature Importance
**What is this?** The mean absolute SHAP value for each feature.  
**What to focus on:** The length of the bar. It represents the "global" influence of a feature—how much it moves the prediction on average, regardless of direction."""))
        
        cells.append(nbf.v4.new_code_cell(f"""
plt.figure(figsize=(10, 6))
shap.summary_plot(s_flat, d_flat, feature_names=names, plot_type='bar', show=False, color='#34495e')
plt.title(f"Mean Influence: {label}", fontsize=14, pad=20)
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()"""))

        # --- PLOT 3: TEMPORAL (Optional) ---
        if is_timeseries:
            cells.append(nbf.v4.new_markdown_cell(f"""### 3. Temporal Relevance
**What is this?** A look at which time-steps in the `{look_back}` window are most influential.  
**What to focus on:** Does the model care more about the immediate past (`t-0`, `t-1`) or older history? A peak at `t-0` suggests a reactive model; a peak at `t-5` suggests a dependency on long-term trends."""))
            
            cells.append(nbf.v4.new_code_cell(f"""
current_shap_raw = np.array(all_shap_dict[{repr(target_idx)}])
importance_per_step = np.abs(current_shap_raw).mean(axis=(0, 2)).flatten()
time_labels = [f"t-{{i}}" for i in range(look_back-1, -1, -1)]
time_axis = list(range(len(time_labels)))

plt.figure(figsize=(10, 4))
plt.plot(time_axis, importance_per_step, marker='o', linewidth=2, color='#e67e22')
plt.xticks(time_axis, time_labels)
plt.title(f"Temporal Relevance: {label}", fontsize=14)
plt.xlabel("Timeline (History -> Present)")
plt.fill_between(time_axis, importance_per_step, alpha=0.1, color='#e67e22')
plt.grid(True, alpha=0.3)
plt.show()"""))

        # --- PLOT 4: TOP 5 DRIVERS ---
        cells.append(nbf.v4.new_markdown_cell(f"""### 4. Focused View: Top 5 Drivers
**What is this?** A high-precision look at the five most critical variables for `{label}`.  
**What to focus on:** The gap between the 1st and 5th feature. If the 1st is much larger, the model is heavily reliant on a single "smoking gun" variable."""))
        
        cells.append(nbf.v4.new_code_cell(f"""
mean_shap = np.abs(s_flat).mean(axis=0)
sorted_idx = np.argsort(mean_shap)[-5:]
plt.figure(figsize=(10, 5))
plt.barh([names[i] for i in sorted_idx], mean_shap[sorted_idx], color='#2c3e50')
plt.title(f"Top 5 Drivers: {label}", fontsize=14)
for i, v in enumerate(mean_shap[sorted_idx]):
    plt.text(v, i, f"  {{v:.4f}}", va='center', fontweight='bold')
plt.tight_layout()
plt.show()"""))

    # --- 5. Execution and Saving ---
    nb['cells'] = cells
    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    try:
        ep.preprocess(nb, {'metadata': {'path': os.path.dirname(output_path)}})
    except Exception as e:
        print(f"Warning: Execution failed. Error: {e}")

    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    
    print(f"Notebook generated successfully: {output_path}")