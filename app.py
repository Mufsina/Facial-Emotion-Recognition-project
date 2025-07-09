import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image


# 1️⃣ App Title and Description

st.set_page_config(page_title="Facial Emotion Detector", layout="centered")
st.title("Facial Emotion Detection App")
st.markdown("Upload a face image and detect the facial emotion using a deep learning model.")


# 2️⃣ Load Model and Class Labels

@st.cache_resource
def load_emotion_model():
    model = load_model("face_emotion_classificationnn.keras")
    return model

model = load_emotion_model()

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


# 3️⃣ Image Preprocessing

def preprocess_image(img: Image.Image) -> np.ndarray:
    image = img.convert("L")  # Convert to grayscale
    image = image.resize((48, 48))  # Resize
    image = np.array(image)
    image = np.invert(image)  # Invert (like training set)
    image = image / 255.0  # Normalize if needed
    image = image.reshape(1, 48, 48, 1)
    return image


# 4️⃣ Main UI

uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict Emotion"):
        with st.spinner("Analyzing image..."):
            processed_img = preprocess_image(img)
            predictions = model.predict(processed_img)
            predicted_class = np.argmax(predictions)
            confidence = np.max(predictions)

            st.subheader(f"🔍 Predicted Emotion: {emotions[predicted_class]}")
            st.progress(int(confidence * 100))
            st.write(f"Confidence: {confidence:.2f}")
