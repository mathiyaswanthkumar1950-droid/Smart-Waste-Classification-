import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the trained waste classification model
model = tf.keras.models.load_model("smart_waste_model.keras")

class_names = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]

def classify_waste(image):
    if image is None:
        return {"Please upload an image": 1.0}

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)[0]

    return {
        class_names[i]: float(predictions[i])
        for i in range(len(class_names))
    }

demo = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="pil", label="Upload a waste image"),
    outputs=gr.Label(num_top_classes=3, label="Predicted waste category"),
    title="Smart Waste Classification Using Deep Learning",
    description="Upload an image of waste to predict its category."
)

if __name__ == "__main__":
    demo.launch()
