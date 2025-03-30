import streamlit as st
import tensorflow as tf
import gdown
from PIL import Image
import numpy as np
import os
import io

# 🎨 Page Configuration
st.set_page_config(
    page_title="🌾 Farm Disease Detection",
    page_icon="🦠",
    layout="centered"
)

# Custom CSS for Styling
st.markdown("""
    <style>
        body {
            background-color: #f7f7f7;
        }
        .title {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            color: #2E8B57;
        }
        .sub-title {
            font-size: 18px;
            text-align: center;
            color: #555;
        }
        .uploaded-image {
            display: flex;
            justify-content: center;
            margin-bottom: 10px;
        }
        .prediction-box {
            font-size: 20px;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
            background-color: #f4f4f4;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        }
        .confidence-score {
            font-size: 16px;
            font-weight: bold;
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# 📍 Model Paths
MODEL_DIR = "./models"
POULTRY_MODEL_PATH = os.path.join(MODEL_DIR, "poultry_disease_model.h5")
POTATO_MODEL_PATH = os.path.join(MODEL_DIR, "potato_model.h5")
CROP_MODEL_PATH = os.path.join(MODEL_DIR, "crop_disease_model.h5")
CROP_CLASS_INDEX_PATH = os.path.join(MODEL_DIR, "class_indices.pkl")

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# 🌍 Google Drive File ID for Poultry Model
POULTRY_MODEL_DRIVE_ID = "1isifj4_xUzXfUe9qbKLuqNQm5ldYvglo"

# Download Poultry Model if Not Present
def download_poultry_model():
    if not os.path.exists(POULTRY_MODEL_PATH):
        st.info("📥 Downloading Poultry Disease Model...")
        gdown.download(f"https://drive.google.com/uc?id={POULTRY_MODEL_DRIVE_ID}", POULTRY_MODEL_PATH, quiet=False)
        st.success("✅ Poultry Disease Model downloaded successfully!")

download_poultry_model()  # Auto-download if missing

# 🏗️ Load All Models
st.info("📦 Loading Models...")
potato_model = tf.keras.models.load_model(POTATO_MODEL_PATH)
poultry_model = tf.keras.models.load_model(POULTRY_MODEL_PATH)
crop_model = tf.keras.models.load_model(CROP_MODEL_PATH)

# 🏷️ Disease Classes
POTATO_CLASSES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
POULTRY_CLASSES = ['Bumblefoot', 'Fowlpox', 'Healthy', 'Coryza', 'CRD']

# Load Crop Disease Classes
import pickle
with open(CROP_CLASS_INDEX_PATH, "rb") as f:
    CROP_CLASSES = pickle.load(f)
CROP_CLASSES = {v: k for k, v in CROP_CLASSES.items()}  # Reverse mapping

st.success("✅ Models Loaded Successfully!")

# 📸 Image Preprocessing
def preprocess_image(image):
    img = Image.open(io.BytesIO(image))
    img = img.resize((224, 224))  # Resize for Model
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add Batch Dimension
    return img_array

# 🌍 Sidebar Navigation
st.sidebar.title("🔍 Select Model")
model_choice = st.sidebar.radio("", ["Poultry Disease", "Potato Disease", "Crop Disease"])

# 🎯 Main UI
st.markdown("<p class='title'>🌾 Farm Disease Detection</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Upload an image to detect Poultry, Potato, or Crop Diseases.</p>", unsafe_allow_html=True)

# 📤 Image Upload
uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # 🌟 Display Image in a Centered Card
    st.markdown("<div class='uploaded-image'>", unsafe_allow_html=True)
    st.image(image, caption="📷 Uploaded Image", use_column_width=False, width=250)
    st.markdown("</div>", unsafe_allow_html=True)

    # 🚀 Prediction Button
    if st.button("🔍 Predict Disease"):
        with st.spinner("⏳ Analyzing... Please wait"):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes = img_bytes.getvalue()
            img_array = preprocess_image(img_bytes)

            if model_choice == "Poultry Disease":
                prediction = poultry_model.predict(img_array)
                predicted_class = POULTRY_CLASSES[np.argmax(prediction)]
            elif model_choice == "Potato Disease":
                prediction = potato_model.predict(img_array)
                predicted_class = POTATO_CLASSES[np.argmax(prediction)]
            else:
                prediction = crop_model.predict(img_array)
                predicted_class = CROP_CLASSES[np.argmax(prediction)]

            confidence = round(np.max(prediction) * 100, 2)

            # 🎨 Color-code Confidence Score
            if confidence > 85:
                color = "green"
            elif confidence > 60:
                color = "orange"
            else:
                color = "red"

            # 🏷️ Show Prediction Results
            st.markdown(f"<div class='prediction-box'>🩺 <b>Predicted Disease:</b> {predicted_class}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='confidence-score' style='color:{color};'>🔥 Confidence Score: <b>{confidence:.2f}%</b></div>", unsafe_allow_html=True)