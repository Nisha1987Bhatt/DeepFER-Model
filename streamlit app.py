"""
DeepFER — Streamlit frontend
"""

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import os
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def get_model():
    return load_model(str(MODEL_PATH), compile=False)
# ---- CONFIG ----
MODEL_PATH = BASE_DIR / "Emotion_detection_model.h5"
IMG_SIZE = (48, 48)
CLASSES = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# ---- LOAD MODEL (cached so it only loads once, not on every interaction) ----
@st.cache_resource
def get_model():
    return load_model(MODEL_PATH, compile=False)

model = get_model()

# ---- PAGE LAYOUT ----
st.set_page_config(page_title="DeepFER", page_icon="🙂")
st.title("DeepFER: Facial Emotion Recognition")
st.write("Upload a clear, front-facing photo of a face to predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---- PREPROCESS (must match training pipeline exactly) ----
    img = image.convert("L")                   # grayscale
    img = img.resize(IMG_SIZE)                  # 48x48
    img_array = np.array(img) / 255.0           # normalize
    img_array = img_array.reshape(1, 48, 48, 1)  # batch + channel dims

    # ---- PREDICT ----
    with st.spinner("Predicting..."):
        predictions = model.predict(img_array)[0]

    predicted_class = CLASSES[np.argmax(predictions)]
    confidence = float(np.max(predictions)) * 100

    st.subheader(f"Predicted Emotion: **{predicted_class.upper()}**")
    st.write(f"Confidence: {confidence:.2f}%")

    # ---- SHOW ALL CLASS PROBABILITIES AS A BAR CHART ----
    st.bar_chart({CLASSES[i]: float(predictions[i]) for i in range(len(CLASSES))})
else:
    st.info("Please upload an image to get started.")
    
