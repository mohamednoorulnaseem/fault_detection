# app.py (paste into top / replace existing model load + predict code)
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# Load class mapping
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)
index_to_class = {int(v): k for k, v in class_indices.items()}  # ensure int keys

# Cache model load so it doesn't reload on every interaction
@st.cache_resource
def load_model(path="model.h5"):
    model = tf.keras.models.load_model(path)
    return model

model = load_model("model.h5")

st.title("🔎 Metal Surface Fault Detection")
st.write("Upload an image to classify it as **crack**, **corrosion**, or **normal**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_arr = np.array(img_resized).astype(np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  # shape (1,224,224,3)

    # Predict (show spinner while doing heavy work)
    with st.spinner("Predicting..."):
        preds = model.predict(img_arr)
        pred_idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds) * 100)

    st.subheader("Prediction Result")
    st.write(f"**Class:** {index_to_class[pred_idx]}")
    st.write(f"**Confidence:** {confidence:.2f}%")

# below the prediction block, show the confusion matrix if file exists
import os
if os.path.exists("confusion_matrix.png"):
    st.subheader("Model evaluation (confusion matrix)")
    st.image("confusion_matrix.png", use_column_width=True)
