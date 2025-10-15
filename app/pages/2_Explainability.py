# app/pages/2_Explainability.py
"""
🧠 Explainability Page for Chest X-Ray Disease Classification

Enhanced Version — Features:
1. Sidebar with author info and external profile buttons.
2. Multi-section educational layout (5–6 sections).
3. Beautifully commented and structured code.
4. Grad-CAM and SHAP explanations for deep insight.
5. Improved UI/UX using icons, emojis, and section headers.

Author: Amal Prasad Trivedi
"""

# ------------------------------
# Imports
# ------------------------------
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import shap
import matplotlib.pyplot as plt
import cv2
import os
import sys

# ------------------------------
# Project Imports
# ------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_trained_model

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="Explainability", page_icon="🧠", layout="wide")

# ==============================
# 🔹 SIDEBAR SECTION
# ==============================
with st.sidebar:
    st.title("ℹ️ About This Page")
    st.markdown("""
    This module provides **Explainable AI (XAI)** visualization
    for **Chest X-Ray Classification**.

    It helps radiologists and researchers understand *why* a model predicted
    a particular disease outcome (Normal or Pneumonia).
    """)


    # External Links
    st.markdown("---")
    st.markdown("### 👨‍💻 Creator Info")
    st.markdown("""
            **Amal Prasad Trivedi**  
            🎓 B.Tech CSE (AI & ML)  
            💡 Researcher | ML Developer  
            """)

    st.markdown("### 🔍📬  Connect with Developer")
    st.link_button("🌐 Portfolio", "https://amalprasadtrivediportfolio.vercel.app/")
    st.link_button("🔗 LinkedIn", "https://www.linkedin.com/in/amalprasadtrivedi-aiml-engineer")
    st.link_button("💻 GitHub", "https://github.com/amalprasadtrivedi")
    st.markdown("---")

    st.sidebar.info("👨‍💻 Developed by **Amal Prasad Trivedi**\nB.Tech Final Year • Computer Science (AI & ML)")
    st.markdown("---")
    st.caption("⚙️ Developed as part of the AI-Based Radiology Assistant Project.")
    st.caption("⚙️ Built as part of Final Year Major Project - Explainable AI for Radiology")

# ==============================
# 🔹 MAIN TITLE
# ==============================
st.title("🧠 Explainable AI for Chest X-Ray Disease Detection")
st.markdown("""
This system demonstrates how **AI can detect Pneumonia or Normal lungs**
from Chest X-rays — and more importantly, **explain the reasoning** behind its predictions.
""")

st.info("Upload a Chest X-Ray image below to visualize model explainability using Grad-CAM 🔥 and SHAP 🧠.")

# ==============================
# 🔹 MODEL LOADING
# ==============================
MODEL_PATH = os.path.join(project_root, "models", "cnn_model.h5")
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    """Load trained CNN model for Chest X-ray classification."""
    return load_trained_model(MODEL_PATH)

model = load_model()

# ==============================
# 🔹 IMAGE PREPROCESSING
# ==============================
def preprocess_image(img):
    """Resize and normalize uploaded image."""
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0

# ==============================
# 🔹 GRAD-CAM IMPLEMENTATION
# ==============================
def generate_gradcam(img_array, model, last_conv_layer_name=None):
    """
    Generates Grad-CAM heatmap to highlight regions influencing the model.
    """
    _ = model(img_array)

    # Automatically find last Conv2D layer if not given
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    if not last_conv_layer_name:
        raise ValueError("No Conv2D layer found in model!")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = np.zeros(conv_outputs.shape[1:3], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[0, :, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam) if np.max(cam) != 0 else cam
    return cv2.resize(np.uint8(255 * cam), IMG_SIZE)

# ==============================
# 🔹 SHAP IMPLEMENTATION
# ==============================
def generate_shap_explanation(img_array, model):
    """Generate SHAP explanation (feature importance heatmap)."""
    _ = model(img_array)
    explainer = shap.GradientExplainer(model, img_array)
    shap_values = explainer.shap_values(img_array)

    fig, _ = plt.subplots(figsize=(5, 5))
    shap.image_plot(shap_values, img_array, show=False)
    plt.title("🧠 SHAP Feature Attribution", fontsize=14)

    # Convert SHAP map to overlay
    shap_map = np.abs(shap_values[0][0]).mean(axis=-1)
    shap_map = (shap_map - shap_map.min()) / (shap_map.max() + 1e-8)
    shap_map = np.uint8(plt.cm.jet(shap_map)[:, :, :3] * 255)

    original_img = np.uint8(img_array[0] * 255)
    overlay_img = Image.fromarray((0.6 * original_img + 0.4 * shap_map).astype("uint8"))
    return fig, overlay_img

# ==============================
# 🩻 PAGE 1: Introduction to Disease
# ==============================
st.header("🩺 Understanding Pneumonia and X-Ray Diagnosis")
st.markdown("""
**Pneumonia** is an infection that inflames the air sacs in one or both lungs.
It can be caused by bacteria, viruses, or fungi.  
Chest X-rays are crucial in detecting:
- Lung opacity (whitish regions)
- Inflammation or fluid accumulation
- Asymmetry in lung fields

However, interpreting X-rays manually is time-consuming and requires expertise.
AI models can **assist doctors** by providing fast and accurate classification results.
""")

# ==============================
# 🧩 PAGE 2: Upload & Predict
# ==============================
st.header("📂 Upload and Analyze Chest X-Ray")
uploaded_file = st.file_uploader("Upload a Chest X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📸 Uploaded Chest X-Ray", use_container_width=True)

    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)[0][0]

    if prediction < 0.5:
        label = "✅ NORMAL"
        confidence = (1 - prediction) * 100
    else:
        label = "❌ PNEUMONIA"
        confidence = prediction * 100

    st.success(f"**Prediction:** {label}")
    st.metric(label="Confidence Level", value=f"{confidence:.2f}%")

    # ==============================
    # 🔥 PAGE 3: Grad-CAM Visualization
    # ==============================
    st.header("🔥 Grad-CAM Visualization")
    try:
        gradcam = generate_gradcam(processed_img, model)
        img_resized = img.resize(IMG_SIZE)
        heatmap = np.uint8(plt.cm.jet(gradcam / np.max(gradcam))[:, :, :3] * 255)
        overlay = Image.fromarray((0.5 * np.array(img_resized) + 0.5 * heatmap).astype("uint8"))

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="🩻 Original X-ray")
        with col2:
            st.image(overlay, caption="🔥 Grad-CAM Heatmap")

        st.success("Grad-CAM highlights the regions most influential in prediction.")
    except Exception as e:
        st.error(f"Grad-CAM could not be generated: {e}")

    # ==============================
    # 🧠 PAGE 4: SHAP Explainability
    # ==============================
    st.header("🧠 SHAP Explainability")
    try:
        shap_fig, shap_overlay = generate_shap_explanation(processed_img, model)
        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(shap_fig)
        with col2:
            st.image(shap_overlay, caption="🧠 SHAP Overlay")
        st.success("SHAP provides pixel-level insight into model’s reasoning.")
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

# ==============================
# 📚 PAGE 5: Conclusion
# ==============================
st.header("📚 Conclusion & Future Scope")
st.markdown("""
Explainable AI (XAI) bridges the gap between **AI predictions** and **medical trust**.
By combining Grad-CAM and SHAP, this system ensures:
- Transparency of decision-making  
- Support for radiologists  
- Enhanced safety in healthcare AI deployment

🩶 Future versions can integrate multi-disease detection, patient metadata, and interactive 3D visualization.
""")

st.success("✅ System successfully demonstrates Explainable AI in Medical Imaging.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 20px; font-size: 16px;">
    <b>🩻 Explainable AI Chest X-Ray Classification System</b><br>
    Developed by <b>Amal Prasad Trivedi</b><br>
    B.Tech – Computer Science (AI & ML)
</div>
""", unsafe_allow_html=True)