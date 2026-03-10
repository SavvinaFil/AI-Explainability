import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

# --- 1. Path Setup ---
root_dir = Path(__file__).resolve().parents[3]
model_dir = root_dir / "source" / "models"
data_dir = root_dir / "source" / "data"

model_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)

# --- Define your specific names here ---
model_filename = "rf_classify.pkl"
data_filename = "rf_classify_data.csv"

# Create full file paths
model_save_path = model_dir / model_filename
data_save_path = data_dir / data_filename

# --- 2. Data Generation (Keep as is) ---
n_samples = 1000
ages = np.random.randint(18, 70, size=n_samples)

# Calculate the max years of employment per person (we suppose they start working after 18 years old or later)
max_work_years = ages - 18
years_of_employment = np.array([
    np.random.randint(0, max(1, max_years + 1))
    for max_years in max_work_years
])
base_income = np.random.randint(15000, 50000, size=n_samples)
experience_bonus = years_of_employment * np.random.randint(500, 2000, size=n_samples)
income_per_year = np.clip(base_income + experience_bonus, 10000, 150000)

data = {
    "Age": ages,
    "Income per Year": income_per_year,
    "Years of Employment": years_of_employment
}

df = pd.DataFrame(data)
df["Loan Approved"] = (
    (df["Income per Year"] > 40000) &
    (df["Age"] >= 30)&
    (df["Age"] <= 56) &
    (df["Years of Employment"] >= 2) &
    (
        ((df["Age"] > 35) & (df["Age"] <= 50) & (df["Years of Employment"] > 5)) |
        ((df["Income per Year"] > 60000) & (df["Years of Employment"] > 3))
    )
).astype(int)

# --- 3. Save Dataset ---
# Use the full path including the filename
df.to_csv(data_save_path, index=False)
print(f"Dataset saved to: {data_save_path}")

# --- 4. Train Model ---
features = ["Age", "Income per Year", "Years of Employment"]

X = df[features]
y = df["Loan Approved"]

model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
model.fit(X, y)

# --- 5. Save Model ---
# Save the model
with open(model_save_path, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to: {model_save_path}")