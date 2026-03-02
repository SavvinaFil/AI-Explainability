## 📄 Configuration & Data Flow

To ensure the toolbox operates correctly, your directory structure should mirror the paths defined in your `config.json`. 

### 1. Storage & Paths
* **Model Storage (`model_path`)**: Your trained model (e.g. PyTorch weights (`.pth`), or trained sklearn model) should be stored in `source/models/`. This allows the explainer to reload the architecture and weights before performing the analysis.
* **Data (`background_data_path` & `test_data_path`)**: Storing these as pre-processed subsets (separate from the raw dataset) ensures that the explainer uses the exact same normalization and windowing as your training pipeline.
* **Output Directory (`output_dir`)**: All generated SHAP plots, summary statistics, and optional reports will be saved here.

**Recommended Structure:**
```text
project_root/
├── source/
│   ├── models/
│   │   └── lstm_model.pth          # Trained Model
│   └── data/
│       ├── background_data.csv     # Reference set
│       └── data_to_explain.csv     # Target samples
└── output/                         # Resulting SHAP plots
```
---

## 🧪 Background vs. Test Data

In SHAP analysis, there is a critical distinction between the **Background Data** and the **Data to Explain**.

### Background Data (`background_data_path`)
* **What it is:** A representative subset of your training data.
* **What it’s used for:** SHAP explains predictions by comparing the current input to a "baseline." The background data calculates this baseline by "integrating out" features—effectively replacing a feature with values from the background set to measure the impact on the prediction.
* **Why it matters:** It defines the Base Value ($\phi_0$). If the background set is biased or contains outliers, the resulting Shapley values will be skewed. A well-chosen background set ensures that the "contributions" identified are relative to a truly "typical" state of the system.

### Data to Explain (`test_data_path`)
* **What it is:** The specific input instances (observations) you wish to deconstruct.
* **What it’s used for:** These are the targets for "local" explanations. The explainer deconstructs the model's output for these specific samples, assigning a numeric value (the SHAP value) to each feature. This value represents how much that feature pushed the prediction higher or lower compared to the background average.

---

### ⚠️ A Note on Background Data & Explainer Types

The role of the background data changes significantly depending on your `explainer_type`:

#### The Kernel Explainer (The Primary User)
For `explainer_type: "kernel"`, the background data is **mandatory**. 
* **The Mechanism:** KernelSHAP is "black-box"; it only observes inputs and outputs. To "ignore" a feature, it replaces it with samples from the **Background Data**.
* **The Impact:** Feature importance is measured *relative* to this set. If your background data only contains summer months, the explainer cannot accurately attribute importance for winter predictions.

#### Tree and Gradient Explainers (The Optimized Users)
* **Tree Explainer (`rf_regressor`, `rf_classifier`):** Uses the internal tree structure. It is significantly less sensitive to the background set size but still uses it to define the "expected value" of the model.
* **Gradient/Deep Explainer (`lstm`):** Uses model gradients. The background data (or "reference") serves as the starting point for the integration path. For energy data, a common baseline is the average "clear sky" or "zero-input" profile.



---

### Tips Selecting Background Data
1. **Size:** Select a sufficient amount of samples to fully represent your data, but not too many to keep low computational time. 
2. **Diversity:** Use a **K-Means summarized** version of your training set rather than the first 100 rows to ensure you capture the full range of variability.
3. **Consistency:** The background data must have the exact same `normalization` and `look_back` windowing as your `test_data`.

---

## ⚙️ Configuration Parameter Breakdown

| Parameter | Purpose |
| :--- | :--- |
| `analysis` | The class of problem, such as `timeseries` for forecasting with LSTMs. |
| `explainer_type` | Specifies the algorithm (e.g., `gradient` for NNs, `tree` for Random Forest, `kernel` for black-box). |
| `dataset_scope` | Usually set to `whole` to ensure scaling context is maintained across the entire project. |
| `generate_notebook` | If `true`, exports a `.ipynb` file for interactive post-run analysis. |

---

## 📊 Feature & Label Setup
The toolbox maps raw tensor dimensions back to human-readable names:
* **`feature_names`**: Maps the input dimensions (e.g., Physics: `ghi`, Temporal: `hour_sin`, Historical: `PV_lag_24`).
* **`output_labels`**: Defines the target variable being explained (e.g., `PV` power production).