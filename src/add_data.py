import pandas as pd
import numpy as np

# Path to dataset
dataset_path = "data/raw/dataset.csv"

# Load dataset
df = pd.read_csv(dataset_path)

# Number of features (excluding target column)
num_features = len(df.columns) - 1

# Generate 5 new instances
# We'll sample values randomly based on the mean and std of existing data
new_data = []

for _ in range(5):
    new_row = []
    for col in df.columns[:-1]:  # all features (x0..x19)
        mean = df[col].mean()
        std = df[col].std()
        value = np.random.normal(mean, std)  # random value based on distribution
        new_row.append(value)
    # Assign random target value (0 or 1)
    y_value = np.random.choice([0, 1])
    new_row.append(y_value)
    new_data.append(new_row)

# Convert to DataFrame
new_df = pd.DataFrame(new_data, columns=df.columns)

# Append new data to existing dataset
df_updated = pd.concat([df, new_df], ignore_index=True)

# Save updated dataset
df_updated.to_csv(dataset_path, index=False)

print("✅ Se agregaron 5 nuevas instancias al dataset.")
print(f"Nueva cantidad total de filas: {len(df_updated)}")
