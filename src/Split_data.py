import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
raw_path = "data/raw/dataset.csv"
processed_dir = "data/processed/"

# Load dataset
df = pd.read_csv(raw_path)

# Split: 70% train, 15% validation, 15% test
train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["y"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["y"])

# Save splits
train_df.to_csv(processed_dir + "train.csv", index=False)
val_df.to_csv(processed_dir + "val.csv", index=False)
test_df.to_csv(processed_dir + "test.csv", index=False)

print("✅ Dataset dividido exitosamente:")
print(f"  Entrenamiento: {train_df.shape[0]} filas")
print(f"  Validación: {val_df.shape[0]} filas")
print(f"  Prueba: {test_df.shape[0]} filas")
