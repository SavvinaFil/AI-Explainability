import torch
import shap
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from .base import TimeseriesExplainerBase
from datetime import datetime
from output.utils.report_gen import generate_notebook

class LSTMForecaster(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = torch.nn.Linear(hidden_dim, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

class LSTMExplainer(TimeseriesExplainerBase):
    def __init__(self, config):
        super().__init__(config)
        # Dynamically attach the function as a method
        self.generate_notebook = generate_notebook.__get__(self)
    
    
    def load_model(self):
        
        """Reconstructs the model from state_dict for better stability."""
        # 1. Get parameters from config
        self.input_dim = self.config.get("input_dim", 12)
        self.hidden_dim = self.config.get("hidden_size", 16)
        self.model_type = self.config.get("model_type", "lstm")
        self.output_dim = 1
        self.output_labels = self.config.get("output_labels", 0)
        
        # 2. Reconstruct the 'Skeleton' (Architecture)
        # This works because the class LSTMForecaster is defined in this same file
        self.model = LSTMForecaster(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        
        # 3. Load the 'Muscles' (Weights)
        model_full_path = self.get_path("model_path")
        
        # Load the weights (state_dict)
        state_dict = torch.load(model_full_path, map_location=torch.device('cpu'))
        
        # Apply weights to the skeleton
        self.model.load_state_dict(state_dict)
        
        # 4. Set to evaluation mode
        self.model.eval()
        torch.set_grad_enabled(True) 
        print("Model state_dict loaded successfully.")

    def explain(self):
        """Agnostic explanation: Uses pre-processed tensors."""
        # Load tensors directly
        bg_path = self.get_path("background_data_path")
        test_path = self.get_path("test_data_path")
        
        background = torch.load(bg_path)
        test_data = torch.load(test_path)
        
        # Determine the subset to explain to save time
        explain_len = min(len(test_data), 50)
        test_subset = test_data[:explain_len]

        # Initialize Explainer
        if self.config["explainer_type"] == "gradient":
            explainer = shap.GradientExplainer(self.model, background)
            self.shap_values = explainer(test_subset)
        
        elif self.config["explainer_type"] == "deep":
            explainer = shap.DeepExplainer(self.model, background)
            self.shap_values = explainer.shap_values(test_subset, check_additivity=False)

        # FIX: Define the numpy versions consistently
        # Use test_subset (the exact 50 samples explained)
        self.raw_data_values = test_subset.numpy() 
        
        # Extract values for the dictionary
        if hasattr(self.shap_values, 'values'):
            # GradientExplainer returns an Explanation object
            val_to_plot = self.shap_values.values
        else:
            # DeepExplainer returns a list of arrays (one per output)
            val_to_plot = self.shap_values[0] if isinstance(self.shap_values, list) else self.shap_values

        # Align with the Multi-Target format for generate_notebook
        self.all_shap_values = {0: val_to_plot}
        
        print(f"SHAP explanation complete. Data shape: {self.raw_data_values.shape}")

    def plot_results(self):
        from datetime import datetime
        output_dir = self.config.get("output_dir", "output/")
        os.makedirs(output_dir, exist_ok=True)

        # 1. Generate the timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 2. Create the path
        # explainer_type = self.config['explainer_type']
        nb_name = f"report_lstm_{timestamp}.ipynb"
        nb_path = os.path.join(output_dir, nb_name)
        
        # 3. Call the updated utility
        generate_notebook(
            explainer_inst=self,
            all_shap_values=self.all_shap_values, # Uses the dict from explain()
            raw_data=self.raw_data_values,
            output_path=nb_path
        )
        
    def save_results_to_excel(self):
        """Flattens 3D LSTM SHAP values and saves to Excel with timestamps."""
        # 1. Setup Paths
        output_dir = self.get_path("output_dir")
        os.makedirs(output_dir, exist_ok=True)
        
        # 2. Extract Data
        # Handle GradientExplainer (Explanation object) vs DeepExplainer (list)
        if hasattr(self.shap_values, 'values'):
            shap_array = self.shap_values.values
        else:
            shap_array = self.shap_values[0] if isinstance(self.shap_values, list) else self.shap_values

        # raw_data shape: (Samples, Lookback, Features)
        look_back = self.config["look_back"]
        features = self.config["feature_names"]
        
        # 3. Flatten 3D to 2D
        # We create column names like 'FeatureA_t-0', 'FeatureA_t-1', etc.
        flat_cols_shap = [f"SHAP_{feat}_t-{look_back - 1 - i}" for i in range(look_back) for feat in features]
        flat_cols_data = [f"Val_{feat}_t-{look_back - 1 - i}" for i in range(look_back) for feat in features]
        
        # Reshape: (N, Time, Feat) -> (N, Time * Feat)
        shap_flat = shap_array.reshape(shap_array.shape[0], -1)
        #data_flat = self.raw_data.reshape(self.raw_data.shape[0], -1)
        data_flat = self.raw_data_values.reshape(self.raw_data_values.shape[0], -1)

        # 4. Create DataFrames
        shap_df = pd.DataFrame(shap_flat, columns=flat_cols_shap)
        data_df = pd.DataFrame(data_flat, columns=flat_cols_data)
        
        # 5. Add Predictions (if available)
        # We get the model output for these samples to show alongside the SHAP values
        import torch
        self.model.eval()
        with torch.no_grad():
            #test_tensor = torch.tensor(self.raw_data).float()
            test_tensor = torch.tensor(self.raw_data_values).float()
            preds = self.model(test_tensor).numpy().flatten()
        
        pred_df = pd.DataFrame({"Model_Prediction": preds})
        
        # 6. Concatenate everything: [Inputs] + [Prediction] + [SHAP Values]
        output_df = pd.concat([data_df, pred_df, shap_df], axis=1)
        
        # 7. Save with Timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"shap_audit_{timestamp}.xlsx")
        
        try:
            output_df.to_excel(output_path, index=False)
            print(f"Excel audit saved: {output_path}")
        except Exception as e:
            csv_path = output_path.replace(".xlsx", ".csv")
            output_df.to_csv(csv_path, index=False)
            print(f"Excel failed, saved CSV instead: {csv_path}")