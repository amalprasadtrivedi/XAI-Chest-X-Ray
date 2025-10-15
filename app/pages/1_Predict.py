# app/pages/1_Predict.py
"""
🔍 Prediction Page - Explainable AI for Chest X-Ray Classification

This Streamlit page allows users to upload chest X-ray images
and get AI-based classification results (Normal / Pneumonia)
along with confidence levels.

Sections:
1️⃣ Sidebar Information and Developer Links
2️⃣ Upload and Display Image
3️⃣ Model Information and Preprocessing
4️⃣ Prediction Results
5️⃣ System Transparency and Next Steps
"""

# ------------------------------
# Imports
# ------------------------------
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import sys
from pathlib import Path

# ------------------------------
# Add project root to path
# ------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_trained_model

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="🔍 Chest X-Ray Prediction",
    page_icon="🩺",
    layout="wide"
)

# ------------------------------
# Sidebar Section
# ------------------------------
st.sidebar.title("ℹ️ About the System")
st.sidebar.markdown("""
This module performs **AI-powered medical diagnosis** using **Deep Learning** and **Explainable AI (XAI)**.  
Upload a **Chest X-Ray image**, and the system will classify it as either:

- 🟢 **Normal**
- 🔴 **Pneumonia**
""")

# External Links
st.sidebar.markdown("---")
st.sidebar.markdown("### 👨‍💻 Creator Info")
st.sidebar.markdown("""
**Amal Prasad Trivedi**  
🎓 B.Tech CSE (AI & ML)  
💡 Researcher | ML Developer  
""")
st.sidebar.markdown("### 🔍📬  Connect with Developer")
st.sidebar.link_button("🌐 Portfolio", "https://amalprasadtrivediportfolio.vercel.app/")
st.sidebar.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/amalprasadtrivedi-aiml-engineer")
st.sidebar.link_button("💻 GitHub", "https://github.com/amalprasadtrivedi")
st.sidebar.markdown("---")

st.sidebar.info("👨‍💻 **Developed by:** Amal Prasad Trivedi\nB.Tech – Computer Science (AI & ML)")
st.sidebar.markdown("---")
st.sidebar.caption("⚙️ Developed as part of the AI-Based Radiology Assistant Project.")
st.sidebar.caption("⚙️ Built as part of Final Year Major Project - Explainable AI for Radiology")

# ------------------------------
# Title and Introduction
# ------------------------------
st.title("🩻 Chest X-Ray Disease Prediction")
st.subheader("Deep Learning Meets Explainability for Reliable Diagnosis")

st.markdown("""
Welcome to the **Prediction Interface** of our **Explainable AI Chest X-Ray System**.  
This intelligent diagnostic tool identifies **Pneumonia** from chest X-rays with high accuracy while maintaining **transparency and interpretability**.

---

### 💡 About the System
This module:
- Accepts an uploaded X-ray image (JPG/PNG)
- Preprocesses it into a model-compatible format
- Predicts the medical condition (Normal/Pneumonia)
- Displays **confidence** and **visual feedback**
""")

st.markdown("---")

# ------------------------------
# Page 1: Model Overview
# ------------------------------
st.header("🧠 Model Information")

st.markdown("""
The model is built using **Convolutional Neural Networks (CNNs)**, which mimic how the human brain processes visual information.

- **Input size:** 224×224 pixels  
- **Model type:** Transfer Learning (e.g., VGG16 / ResNet-based CNN)  
- **Output:** Binary classification → Normal / Pneumonia  
- **Training dataset:** 200 chest X-ray samples  
- **Loss function:** Binary Crossentropy  
- **Optimizer:** Adam  

These configurations ensure both high performance and clinical trustworthiness.
""")

model_img = Path("assets/system_overview.jpg")
if model_img.exists():
    st.image(str(model_img), caption="CNN Model Architecture", use_container_width=True)
else:
    st.info("🖼️ Add 'model_architecture.png' to assets/ for better illustration.")

st.markdown("---")

# ------------------------------
# Page 2: Upload and Preprocessing
# ------------------------------
st.header("📤 Upload a Chest X-Ray Image")

st.markdown("""
Please upload a **Chest X-Ray image** in JPG or PNG format.  
The image will be automatically resized, normalized, and prepared for prediction.
""")

# Define constants
MODEL_PATH = os.path.join(project_root, "models", "cnn_model.h5")
IMG_SIZE = (224, 224)

# Load trained model (cached for performance)
@st.cache_resource
def load_model():
    model = load_trained_model(MODEL_PATH)
    return model

model = load_model()

# Preprocessing function
def preprocess_image(img):
    """
    Preprocess uploaded chest X-ray image for prediction.
    Steps:
    1. Resize to model input size.
    2. Convert to numpy array and normalize pixel values.
    3. Add batch dimension for Keras compatibility.
    """
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# File upload widget
uploaded_file = st.file_uploader("📁 Upload a Chest X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🩻 Uploaded Chest X-Ray", use_container_width=True)

    # Preprocess and predict
    processed_img = preprocess_image(img)

    st.markdown("### ⏳ Running Model Inference...")
    prediction = model.predict(processed_img)[0][0]

    # Interpret prediction
    if prediction < 0.5:
        label = "🟢 NORMAL"
        confidence = (1 - prediction) * 100
    else:
        label = "🔴 PNEUMONIA"
        confidence = prediction * 100

    st.markdown("---")

    # ------------------------------
    # Page 3: Prediction Result
    # ------------------------------
    st.header("📊 Prediction Result")

    st.markdown(f"### **Prediction:** {label}")
    st.metric(label="Confidence Level", value=f"{confidence:.2f}%")

    if label == "🔴 PNEUMONIA":
        st.warning("""
        ⚠️ **Potential Pneumonia Detected!**  
        Please consult a radiologist or medical professional for verification.
        """)
    else:
        st.success("""
        ✅ **Normal X-Ray Detected!**  
        No significant abnormalities observed.
        """)

    st.markdown("---")

    # ------------------------------
    # Page 4: System Transparency
    # ------------------------------
    st.header("🔎 System Transparency")

    st.markdown("""
    While AI can predict diseases effectively, **explainability** is crucial to
    understand **why** the model made a decision.  

    Our system supports:
    - **Grad-CAM (Gradient-weighted Class Activation Mapping)** → highlights affected regions.
    - **SHAP (SHapley Additive exPlanations)** → shows feature contribution to model output.

    Navigate to the **Explainability Page** to visualize the AI’s reasoning.
    """)

    explain_img = Path("assets/explainability.jpg")
    if explain_img.exists():
        st.image(str(explain_img), caption="Explainability Visualization Example", use_container_width=True)
    else:
        st.info("🖼️ Add 'explainability.png' to assets/ for better visual impact.")

    st.markdown("---")

    # ------------------------------
    # Page 5: Next Steps
    # ------------------------------
    st.header("🧭 Next Steps")

    st.markdown("""
    - Proceed to **Explainability Page** to visualize model attention areas (Grad-CAM, SHAP).  
    - Visit **Results Page** for evaluation metrics, confusion matrix, and performance summary.  
    - Re-upload different scans for comparative diagnosis.

    💬 *This tool is intended for educational and research purposes — not as a medical diagnostic substitute.*
    """)

else:
    st.info("📸 Please upload a chest X-ray image to start prediction.")

st.markdown("---")

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
<div style="text-align: center; padding: 20px; font-size: 16px;">
    <b>🩻 Explainable AI Chest X-Ray Classification System</b><br>
    Developed by <b>Amal Prasad Trivedi</b><br>
    B.Tech – Computer Science (AI & ML)
</div>
""", unsafe_allow_html=True)
