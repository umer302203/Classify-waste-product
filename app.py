import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the fine‑tuned model (now from root directory)
model = tf.keras.models.load_model("O_R_tlearn_fine_tune_vgg16.keras")

# Class names
CLASS_NAMES = ["Organic (O)", "Recyclable (R)"]

def predict_image(image):
    """
    image: PIL Image or numpy array (H, W, 3)
    Returns: label string and confidence score
    """
    # Resize to 150x150 (the model's input size)
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0          # rescale as during training
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    pred = model.predict(img_array)[0][0]      # sigmoid output
    confidence = pred if pred > 0.5 else 1 - pred
    label = CLASS_NAMES[0] if pred < 0.5 else CLASS_NAMES[1]
    return f"{label} (confidence: {confidence:.2f})"

# Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Waste Classifier (Organic vs Recyclable)",
    description="Upload an image of waste to classify it as Organic (O) or Recyclable (R)."
)

if __name__ == "__main__":
    iface.launch()
