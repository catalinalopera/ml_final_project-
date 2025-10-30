import joblib
import numpy as np
import gradio as gr

# Model info
model_path = "models/SupportVectorMachine.joblib"
model_name = model_path.split("/")[-1]

# Load model
model = joblib.load(model_path)

# Features
feature_names = [f"x{i}" for i in range(20)]

def predict_svm(real_value, *inputs):
    features = np.array(inputs, dtype=float).reshape(1, -1)
    pred = model.predict(features)[0]
    return model_name, float(pred), real_value

with gr.Blocks(title="Predicción Manual") as demo:
    gr.Markdown("## Predicción Manual con SVM")
    gr.Markdown("Ingrese valores para x0–x19")

    inputs_list = []
    for row in range(4):
        with gr.Row():
            for col in range(5):
                idx = row * 5 + col
                num = gr.Number(label=f"x{idx}", interactive=True)
                inputs_list.append(num)

    # Campo Real al final (después de la predicción)
    real_input = gr.Number(label="Valor real (si lo tienes)")

    model_out = gr.Textbox(label="Modelo", interactive=False)
    pred_out = gr.Textbox(label="Predicción", interactive=False)
    real_out = gr.Textbox(label="Real", interactive=False)

    btn = gr.Button("Predecir", variant="primary")
    btn.click(
        fn=predict_svm,
        inputs=[real_input] + inputs_list,
        outputs=[model_out, pred_out, real_out]
    )

demo.launch()
