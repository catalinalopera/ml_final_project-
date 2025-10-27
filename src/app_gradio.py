import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os

# 📂 Paths
MODELS_DIR = "models"
DATA_DIR = os.path.join("data", "processed")
RAW_DATA_PATH = os.path.join("data", "raw", "dataset.csv")

# Load feature names
df_raw = pd.read_csv(RAW_DATA_PATH)
feature_names = df_raw.columns[:-1] #documentacion
target_col = df_raw.columns[-1] #documentacion

def list_models():
    return [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]

def list_datasets():
    return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]

# Manual prediction
def predict_manual(model_name, *inputs):
    model_path = os.path.join(MODELS_DIR, model_name)
    model = joblib.load(model_path)
    X = np.array(inputs).reshape(1, -1)
    pred = model.predict(X)[0]
    return f"✅ Modelo: {model_name}\nPredicción: {pred}"

# Dataset prediction
def predict_dataset(model_name, dataset_name):
    model_path = os.path.join(MODELS_DIR, model_name)
    dataset_path = os.path.join(DATA_DIR, dataset_name)

    try:
        model = joblib.load(model_path)
        df = pd.read_csv(dataset_path)
        X = df[feature_names]
        preds = model.predict(X)
        df["prediction"] = preds

        output_path = os.path.join(DATA_DIR, "predictions.csv")
        df.to_csv(output_path, index=False)

        acc = None
        if target_col in df.columns:
            acc = (df[target_col] == df["prediction"]).mean()

        msg = f"✅ Predicciones generadas para {dataset_name} usando {model_name}\n"
        if acc is not None:
            msg += f"📈 Accuracy: {acc:.4f}"
        else:
            msg += "⚠️ No se encontró la columna target para evaluar."
        return msg, df.head(10)
    except Exception as e:
        return f"❌ Error: {str(e)}", None


# -----------------------------
# INTERFAZ GRADIO CON TABS
# -----------------------------
with gr.Blocks(title="ML Project App") as app:
    gr.Markdown("# 🧠 Interfaz de Predicción de Modelos Entrenados")

    with gr.Tab("🔢 Modo Manual"):
        model_selector_m = gr.Dropdown(choices=list_models(), label="Selecciona el modelo")
        inputs = [gr.Number(label=col) for col in feature_names]
        output_m = gr.Textbox(label="Resultado de la predicción", lines=10, max_lines=20, interactive=False)
        btn_m = gr.Button("Predecir")
        btn_m.click(fn=predict_manual, inputs=[model_selector_m] + inputs, outputs=output_m)

    with gr.Tab("📂 Modo Dataset"):
        model_selector_d = gr.Dropdown(choices=list_models(), label="Selecciona el modelo")
        dataset_selector = gr.Dropdown(choices=list_datasets(), label="Selecciona dataset")
        output_text = gr.Textbox(label="Resumen", lines=10, max_lines=20, interactive=False)
        output_table = gr.Dataframe(label="Vista previa de predicciones")
        btn_d = gr.Button("Generar predicciones")
        btn_d.click(fn=predict_dataset, inputs=[model_selector_d, dataset_selector], outputs=[output_text, output_table])

if __name__ == "__main__":
    app.launch()
