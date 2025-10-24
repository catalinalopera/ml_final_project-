import pandas as pd

# Load the dataset
df = pd.read_csv("data/raw/dataset.csv")

# Remove the last 10 rows
df = df.iloc[:-10]

# Save the updated dataset
df.to_csv("data/raw/dataset.csv", index=False)

print("✅ Últimas 10 filas eliminadas y dataset actualizado.")