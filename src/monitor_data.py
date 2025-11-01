# src/monitor_data.py
import pandas as pd
import os

from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, DataQualityTab, CatTargetDriftTab

# 📂 Paths
DATA_DIR = os.path.join("data", "processed")
OUTPUT_DIR = os.path.join("reports")

# 📁 Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🧾 Cargar datasets
train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

reference_data = pd.read_csv(train_path)
current_data = pd.read_csv(test_path)

# ✅ Aseguramos que el target esté presente
target_col = "y"
if target_col not in reference_data.columns or target_col not in current_data.columns:
    raise ValueError(f"La columna '{target_col}' no se encuentra en los datasets.")

# ==========================================
# 1️⃣ Reporte de Data Drift
# ==========================================
print("🔍 Generando reporte de Data Drift...")

data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
data_drift_dashboard.calculate(reference_data, current_data)
data_drift_dashboard.save(os.path.join(OUTPUT_DIR, "data_drift_report.html"))

print("✅ Reporte de Data Drift guardado.")

# ==========================================
# 2️⃣ Reporte de Target Drift
# ==========================================
print("🎯 Generando reporte de Target Drift...")

target_drift_dashboard = Dashboard(tabs=[
    CatTargetDriftTab()
])

target_drift_dashboard.calculate(reference_data, current_data)
target_drift_dashboard.save(os.path.join(OUTPUT_DIR, "target_drift_report.html"))

print("✅ Reporte de Target Drift guardado correctamente.")

# ==========================================
# 3️⃣ Reporte de Calidad de Datos
# ==========================================
print("🧹 Generando reporte de Calidad de Datos...")

data_quality_dashboard = Dashboard(tabs=[DataQualityTab()])
data_quality_dashboard.calculate(reference_data, current_data)
data_quality_dashboard.save(os.path.join(OUTPUT_DIR, "data_quality_report.html"))

print("✅ Reporte de Calidad guardado.")

print("\n🎉 Todos los reportes fueron generados exitosamente en la carpeta 'reports/'.")

