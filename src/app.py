import gradio as gr
import joblib
import numpy as np
import pandas as pd
import os

# 📂 Paths - IMPORTANTE: Usar paths relativos
MODELS_DIR = "models"
DATA_DIR = os.path.join("data", "processed")
RAW_DATA_PATH = os.path.join("data", "raw", "dataset.csv")

# Cargar nombres de features con manejo de errores
try:
    df_raw = pd.read_csv(RAW_DATA_PATH)
    feature_names = df_raw.columns[:-1]  # documentación
    target_col = df_raw.columns[-1]      # documentación
except Exception as e:
    # Valores por defecto en caso de error
    feature_names = [f"feature_{i}" for i in range(10)]  # Ajusta según tus datos
    target_col = "target"
    print(f"⚠️ No se pudo cargar dataset.csv: {e}")

def list_models():
    """Lista los modelos disponibles"""
    try:
        return [f for f in os.listdir(MODELS_DIR) if f.endswith(".joblib")]
    except FileNotFoundError:
        return ["modelo_ejemplo.joblib"]  # Placeholder

def list_datasets():
    """Lista los datasets disponibles"""
    try:
        return [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    except FileNotFoundError:
        return ["dataset_ejemplo.csv"]  # Placeholder

# Manual prediction
def predict_manual(model_name, *inputs):
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        model = joblib.load(model_path)
        X = np.array(inputs).reshape(1, -1)
        pred = model.predict(X)[0]
        return f"✅ Modelo: {model_name}\nPredicción: {pred}"
    except Exception as e:
        return f"❌ Error en predicción manual: {str(e)}"

# Dataset prediction
def predict_dataset(model_name, dataset_name):
    try:
        model_path = os.path.join(MODELS_DIR, model_name)
        dataset_path = os.path.join(DATA_DIR, dataset_name)

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
# INTERFAZ GRADIO
# -----------------------------
with gr.Blocks(title="ML Project App", theme=gr.themes.Soft()) as app:
    gr.Markdown("# 🧠 Interfaz de Predicción de Modelos Entrenados")
    gr.Markdown("Carga tus modelos y datasets en las carpetas correspondientes")

    with gr.Tab("🔢 Modo Manual"):
        gr.Markdown("### Predicción manual por características")
        model_selector_m = gr.Dropdown(
            choices=list_models(), 
            label="Selecciona el modelo",
            value=list_models()[0] if list_models() else None
        )
        
        # Crear inputs dinámicamente
        inputs = []
        for i, col in enumerate(feature_names):
            inputs.append(gr.Number(
                label=col, 
                value=0.0,  # Valor por defecto
                info=f"Feature {i+1}"
            ))
        
        output_m = gr.Textbox(
            label="Resultado de la predicción", 
            lines=5, 
            interactive=False
        )
        btn_m = gr.Button("🎯 Predecir", variant="primary")
        btn_m.click(
            fn=predict_manual, 
            inputs=[model_selector_m] + inputs, 
            outputs=output_m
        )

    with gr.Tab("📂 Modo Dataset"):
        gr.Markdown("### Predicción por lotes con dataset")
        model_selector_d = gr.Dropdown(
            choices=list_models(), 
            label="Selecciona el modelo",
            value=list_models()[0] if list_models() else None
        )
        dataset_selector = gr.Dropdown(
            choices=list_datasets(), 
            label="Selecciona dataset",
            value=list_datasets()[0] if list_datasets() else None
        )
        output_text = gr.Textbox(
            label="Resumen", 
            lines=4, 
            interactive=False
        )
        output_table = gr.Dataframe(
            label="Vista previa de predicciones (primeras 10 filas)",
            max_rows=10
        )
        btn_d = gr.Button("📊 Generar predicciones", variant="primary")
        btn_d.click(
            fn=predict_dataset, 
            inputs=[model_selector_d, dataset_selector], 
            outputs=[output_text, output_table]
        )

# Para Hugging Face Spaces
if __name__ == "__main__":
    app.launch(share=False)  # Important: share=False para Spaces