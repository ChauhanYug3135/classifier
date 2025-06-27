import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import json
import os

# Page configuration
st.set_page_config(page_title="🤖 AI Image Classifier Hub", layout="centered")

# ---------------------------
# Load Models and Metadata
# ---------------------------

@st.cache_resource
def load_dog_cat_model():
    return load_model("dogs_vs_cats_model.h5")

@st.cache_resource
def load_plant_model():
    model = load_model("plant_model.h5")
    with open("preprocessor.pkl", "rb") as f:
        class_names = pickle.load(f)
    if isinstance(class_names, dict):
        class_names = [v for k, v in sorted(class_names.items(), key=lambda x: int(x[0]))]
    return model, class_names

@st.cache_data
def load_plant_description():
    with open("description.json", "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------
# Image Preprocessing
# ---------------------------

def preprocess_image(image, size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(size)
    img_array = np.array(image).astype('float32') / 255.0
    return np.expand_dims(img_array, axis=0)

# ---------------------------
# Prediction Functions
# ---------------------------

def predict_dog_cat(model, image):
    input_img = preprocess_image(image, size=(150, 150))
    prob = model.predict(input_img)[0][0]
    label = "Dog" if prob > 0.5 else "Cat"
    confidence = prob if prob > 0.5 else 1 - prob
    return label, confidence * 100

def predict_plant(model, class_names, image):
    input_img = preprocess_image(image, size=(100, 100))
    preds = model.predict(input_img)[0]
    idx = np.argmax(preds)
    return class_names[idx], preds[idx] * 100

# ---------------------------
# App UI
# ---------------------------

st.title("🤖 AI Image Classifier Hub")
st.markdown("Upload an image and choose the model to classify Dog vs Cat or Plant Disease.")

task = st.sidebar.selectbox("Select Task", ["Dog vs Cat", "Plant Disease"])

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)
    st.markdown("---")

    if task == "Dog vs Cat":
        st.subheader("🐶 Dog vs 🐱 Cat Classification")
        model = load_dog_cat_model()
        label, confidence = predict_dog_cat(model, image)
        st.markdown(f"### 🧠 Prediction: **{label}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        if label == "Dog":
            st.info("🐕 This looks like a dog! Loyal, energetic, and friendly.")
        else:
            st.info("🐱 This looks like a cat! Calm, independent, and graceful.")

    elif task == "Plant Disease":
        st.subheader("🌿 Plant Disease Detection")
        model, class_names = load_plant_model()
        description_data = load_plant_description()
        label, confidence = predict_plant(model, class_names, image)
        st.markdown(f"### 🧠 Detected Disease: **{label}**")
        st.markdown(f"**Confidence:** {confidence:.2f}%")

        if label in description_data:
            info = description_data[label]
            st.markdown("### 📝 Disease Description")
            st.markdown(f"**🔬 Scientific Name:** {info.get('scientific_name', 'N/A')}")
            st.markdown(f"**🧫 Category:** {info.get('category', 'N/A')}")
            st.markdown(f"**📖 Description:** {info.get('description', 'N/A')}")

            symptoms = info.get('symptoms', [])
            if symptoms:
                st.markdown("**🩺 Symptoms:**")
                for symptom in symptoms:
                    st.markdown(f"- {symptom}")

            st.markdown(f"**📉 Impact:** {info.get('impact', 'N/A')}")

            controls = info.get('prevention_control', [])
            if controls:
                st.markdown("**🛡️ Prevention & Control:**")
                for step in controls:
                    st.markdown(f"- {step}")
        else:
            st.warning("No description available for this disease.")

else:
    st.info("👆 Upload an image to begin classification.")
